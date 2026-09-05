from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict

import pandas as pd

from lib.tad_bulk import (
    TAD_BULK_COLUMNS,
    canonical_account,
    normalize_legal,
    target_house_numbers,
    target_legal_descriptions,
    _filter_to_targets,
    _read_pipe,
)

_PART_RE = re.compile(r"(?i)(.+\.txt)\.part(\d+)$")
_SUPPORTED = (".txt", ".csv", ".xlsx")


def _norm_col(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _find_column(df: pd.DataFrame, candidates: list[str]):
    names = {_norm_col(c): c for c in df.columns}
    for candidate in candidates:
        key = _norm_col(candidate)
        if key in names:
            return names[key]
    return None


def _classify_columns(columns) -> str:
    names = {_norm_col(c) for c in columns}

    improvement_markers = {
        "improvementdetail",
        "improvementdetaildescription",
    }
    comp_markers = {
        "marketarea",
        "improvementtype",
        "style",
        "quality",
        "condition",
        "actualyearbuilt",
        "effectiveyearbuilt",
        "mainarea",
    }
    account_markers = {
        "accountnum",
        "accountnumber",
        "account",
        "apn",
        "pin",
        "parcelid",
        "propertyid",
    }
    property_markers = {
        "situsaddress",
        "legaldescription",
        "ownername",
        "propertyclass",
        "stateusecode",
    }

    if names & improvement_markers and names & account_markers:
        return "improvement"
    if len(names & comp_markers) >= 2 and names & account_markers:
        return "comp"
    if names & account_markers and names & property_markers:
        return "property"
    return "unknown"


def _classify_frame(df: pd.DataFrame) -> str:
    return _classify_columns(df.columns)


def _part_members(uploaded_file):
    raw = uploaded_file.getvalue()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return None, []

    found = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        member_name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        match = _PART_RE.fullmatch(member_name)
        if match:
            found.append((match.group(1), int(match.group(2)), info))
    return zf, found


def _rows_from_split_parts(
    parts,
    house_targets: set[str],
    legal_targets: set[str],
) -> pd.DataFrame:
    """Stream raw byte-split pipe-delimited PropertyData files in part order."""
    wanted = set(TAD_BULK_COLUMNS)
    header = None
    wanted_positions = None
    situs_pos = None
    legal_pos = None
    rows: list[dict[str, str]] = []
    carry = b""

    def should_keep(fields: list[str]) -> bool:
        if not house_targets and not legal_targets:
            return True

        if situs_pos is not None and situs_pos < len(fields) and house_targets:
            match = re.match(r"^\s*(\d+)", fields[situs_pos] or "")
            if match and match.group(1) in house_targets:
                return True

        if legal_pos is not None and legal_pos < len(fields) and legal_targets:
            legal = normalize_legal(fields[legal_pos] or "")
            if legal and legal in legal_targets:
                return True
        return False

    for zf, info in parts:
        with zf.open(info, "r") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break

                data = carry + block
                lines = data.split(b"\n")
                carry = lines.pop()

                for raw_line in lines:
                    line = raw_line.rstrip(b"\r")
                    if not line:
                        continue

                    text = line.decode("utf-8", errors="replace")
                    if header is None:
                        header = text.split("|")
                        wanted_positions = [
                            (i, col) for i, col in enumerate(header) if col in wanted
                        ]
                        situs_pos = header.index("Situs_Address") if "Situs_Address" in header else None
                        legal_pos = header.index("LegalDescription") if "LegalDescription" in header else None
                        if situs_pos is None and legal_pos is None:
                            raise ValueError(
                                "Split TAD PropertyData file does not contain Situs_Address or LegalDescription."
                            )
                        continue

                    fields = text.split("|")
                    if not should_keep(fields):
                        continue
                    rows.append({
                        col: fields[i] if i < len(fields) else ""
                        for i, col in wanted_positions
                    })

    if carry.strip():
        text = carry.rstrip(b"\r").decode("utf-8", errors="replace")
        if header is None:
            header = text.split("|")
        else:
            fields = text.split("|")
            if should_keep(fields):
                rows.append({
                    col: fields[i] if i < len(fields) else ""
                    for i, col in wanted_positions
                })

    if not rows:
        return pd.DataFrame(columns=TAD_BULK_COLUMNS)
    return pd.DataFrame(rows)


def _read_frame_from_bytes(name: str, raw: bytes, house_targets: set[str], legal_targets: set[str]):
    lower = name.lower()

    if lower.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl", dtype=object)
        kind = _classify_frame(df)
        if kind == "property":
            df = _filter_to_targets(df, house_targets, legal_targets)
        return df, kind

    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
        kind = _classify_frame(df)
        if kind == "property":
            df = _filter_to_targets(df, house_targets, legal_targets)
        return df, kind

    if lower.endswith(".txt"):
        first_line = raw.splitlines()[0].decode("utf-8", errors="replace") if raw else ""
        kind = _classify_columns(first_line.split("|"))
        if kind == "property":
            return _read_pipe(io.BytesIO(raw), house_targets, legal_targets), kind
        df = pd.read_csv(
            io.BytesIO(raw),
            sep="|",
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
        )
        return df, _classify_frame(df)

    raise ValueError("Unsupported TAD file type.")


def _account_column(df: pd.DataFrame):
    return _find_column(
        df,
        ["Account_Num", "Account Number", "Account", "APN", "PIN", "Parcel ID", "Property ID"],
    )


def _prepare_property_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    account_col = _account_column(result)
    if not account_col:
        return pd.DataFrame()
    result["__account"] = result[account_col].map(canonical_account)
    result = result[result["__account"] != ""].copy()
    result["Account_Num"] = result["__account"]
    return result


def _latest_comp_rows(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    account_col = _account_column(result)
    if not account_col:
        return pd.DataFrame()

    result["__account"] = result[account_col].map(canonical_account)
    result = result[result["__account"] != ""].copy()

    tax_col = _find_column(result, ["Tax Year", "TaxYear", "Appraisal Year", "Appraisal_Year"])
    if tax_col:
        result["__tax_sort"] = pd.to_numeric(result[tax_col], errors="coerce")
        result = result.sort_values(["__account", "__tax_sort"], na_position="first")
    result = result.drop_duplicates(subset=["__account"], keep="last")
    return result


def _comp_payload(comp: pd.DataFrame, target_accounts: set[str]) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame(columns=["__account"])

    comp = _latest_comp_rows(comp)
    if comp.empty:
        return pd.DataFrame(columns=["__account"])
    comp = comp[comp["__account"].isin(target_accounts)].copy()

    aliases = {
        "Market Area": ["Market Area", "MarketArea"],
        "Improvement Type": ["Improvement Type", "ImprovementType"],
        "Style": ["Style"],
        "Quality": ["Quality"],
        "Condition": ["Condition"],
        "Actual Year Built": ["Actual Year Built", "ActualYearBuilt"],
        "Effective Year Built": ["Effective Year Built", "EffectiveYearBuilt"],
        "Main Area": ["Main Area", "MainArea", "Living Area"],
        "Stories": ["Stories", "Story", "Story Height"],
    }

    payload = pd.DataFrame({"__account": comp["__account"]})
    for output, candidates in aliases.items():
        source = _find_column(comp, candidates)
        if source:
            payload[output] = comp[source]
    return payload


def _improvement_payload(details: pd.DataFrame, target_accounts: set[str]):
    if details.empty:
        return pd.DataFrame(columns=["__account"]), 0

    work = details.copy()
    account_col = _account_column(work)
    if not account_col:
        return pd.DataFrame(columns=["__account"]), 0

    work["__account"] = work[account_col].map(canonical_account)
    work = work[
        (work["__account"] != "") & work["__account"].isin(target_accounts)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["__account"]), 0

    detail_col = _find_column(
        work,
        ["ImprovementDetail", "Improvement Detail", "Improvement Detail Description", "Description"],
    )
    value_col = _find_column(
        work,
        ["ImprovementValue", "Improvement Value", "Value"],
    )

    rows = []
    for account, group in work.groupby("__account", sort=False):
        item = {
            "__account": account,
            "Improvement Detail Count": int(len(group)),
        }

        if detail_col:
            values = []
            for value in group[detail_col].tolist():
                text = str(value or "").strip()
                if text and text.lower() != "nan" and text not in values:
                    values.append(text)
            if values:
                item["Improvement Details"] = " | ".join(values)

        if value_col:
            numeric = pd.to_numeric(
                group[value_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce",
            )
            if numeric.notna().any():
                item["Improvement Detail Value Total"] = float(numeric.sum())

        rows.append(item)

    return pd.DataFrame(rows), len(work)


def _merge_fill(base: pd.DataFrame, payload: pd.DataFrame) -> pd.DataFrame:
    if payload.empty or "__account" not in payload.columns:
        return base

    merged = base.merge(payload, on="__account", how="left", suffixes=("", "__enrich"))
    for column in payload.columns:
        if column == "__account":
            continue
        enrich_col = f"{column}__enrich"
        if enrich_col not in merged.columns:
            continue

        if column in merged.columns:
            existing = merged[column]
            blank = existing.isna() | existing.astype(str).str.strip().isin(["", "nan", "None"])
            merged.loc[blank, column] = merged.loc[blank, enrich_col]
            merged = merged.drop(columns=[enrich_col])
        else:
            merged = merged.rename(columns={enrich_col: column})
    return merged


def _join_tad_sources(property_frames, comp_frames, improvement_frames):
    diagnostics = {
        "property_rows": 0,
        "property_accounts": 0,
        "comp_rows": sum(len(df) for df in comp_frames),
        "comp_accounts_joined": 0,
        "improvement_rows": sum(len(df) for df in improvement_frames),
        "improvement_rows_joined": 0,
        "improvement_accounts_joined": 0,
    }

    prepared = [_prepare_property_frame(df) for df in property_frames]
    prepared = [df for df in prepared if not df.empty]
    if not prepared:
        return pd.DataFrame(), diagnostics

    master = pd.concat(prepared, ignore_index=True, sort=False)
    master = master.drop_duplicates(subset=["__account"], keep="first")
    diagnostics["property_rows"] = len(master)
    diagnostics["property_accounts"] = master["__account"].nunique()
    accounts = set(master["__account"])

    if comp_frames:
        comp = pd.concat(comp_frames, ignore_index=True, sort=False)
        payload = _comp_payload(comp, accounts)
        diagnostics["comp_accounts_joined"] = payload["__account"].nunique() if not payload.empty else 0
        master = _merge_fill(master, payload)

    if improvement_frames:
        details = pd.concat(improvement_frames, ignore_index=True, sort=False)
        payload, matched_rows = _improvement_payload(details, accounts)
        diagnostics["improvement_rows_joined"] = matched_rows
        diagnostics["improvement_accounts_joined"] = payload["__account"].nunique() if not payload.empty else 0
        master = _merge_fill(master, payload)

    return master.drop(columns=["__account"], errors="ignore"), diagnostics


def load_tad_uploads(uploaded_files, target_addresses: list[str], max_mb: int = 200):
    """Load, classify, and relationally join TAD PropertyData + companion files.

    PropertyData/BigBoy is the master. Residential Comp Attribute and
    Improvement Details are joined by canonical Account_Num/PIN. ZIP archives
    are inspected member-by-member instead of silently choosing only the
    largest file.
    """
    sources: list[str] = []
    errors: list[str] = []
    house_targets = target_house_numbers(target_addresses)
    legal_targets = target_legal_descriptions(target_addresses)

    property_frames: list[pd.DataFrame] = []
    comp_frames: list[pd.DataFrame] = []
    improvement_frames: list[pd.DataFrame] = []

    multipart_groups = defaultdict(list)
    opened = []

    def register(df: pd.DataFrame, kind: str, source: str):
        if df.empty:
            sources.append(f"{source} [{kind}: 0 matching rows]")
            return
        if kind == "property":
            property_frames.append(df)
        elif kind == "comp":
            comp_frames.append(df)
        elif kind == "improvement":
            improvement_frames.append(df)
        else:
            errors.append(f"{source}: unrecognized TAD layout; no enrichment join was attempted")
            return
        sources.append(f"{source} [{kind}]")

    try:
        for uploaded in uploaded_files or []:
            raw = uploaded.getvalue()
            size_mb = len(raw) / 1024 / 1024
            if size_mb > max_mb:
                errors.append(
                    f"{uploaded.name}: upload is {size_mb:.1f} MB; maximum is {max_mb} MB"
                )
                continue

            if uploaded.name.lower().endswith(".zip"):
                try:
                    zf = zipfile.ZipFile(io.BytesIO(raw))
                except zipfile.BadZipFile as exc:
                    errors.append(f"{uploaded.name}: {exc}")
                    continue

                opened.append(zf)
                for info in zf.infolist():
                    if info.is_dir():
                        continue

                    member_name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                    part_match = _PART_RE.fullmatch(member_name)
                    if part_match:
                        multipart_groups[part_match.group(1)].append(
                            (int(part_match.group(2)), zf, info, uploaded.name)
                        )
                        continue

                    if not member_name.lower().endswith(_SUPPORTED):
                        continue

                    try:
                        with zf.open(info, "r") as stream:
                            member_raw = stream.read()
                        df, kind = _read_frame_from_bytes(
                            member_name, member_raw, house_targets, legal_targets
                        )
                        register(df, kind, f"{uploaded.name} → {info.filename}")
                    except Exception as exc:
                        errors.append(f"{uploaded.name} → {info.filename}: {exc}")
                continue

            try:
                df, kind = _read_frame_from_bytes(
                    uploaded.name, raw, house_targets, legal_targets
                )
                register(df, kind, uploaded.name)
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")

        for base, members in multipart_groups.items():
            members.sort(key=lambda item: item[0])
            numbers = [item[0] for item in members]
            expected = list(range(1, max(numbers) + 1)) if numbers else []
            if numbers != expected:
                missing = sorted(set(expected) - set(numbers))
                errors.append(
                    f"{base}: missing split part(s) {', '.join(map(str, missing))}"
                )
                continue

            try:
                ordered = [(zf, info) for _, zf, info, _ in members]
                df = _rows_from_split_parts(ordered, house_targets, legal_targets)
                zip_names = []
                for _, _, _, upload_name in members:
                    if upload_name not in zip_names:
                        zip_names.append(upload_name)
                register(
                    df,
                    "property",
                    f"{' + '.join(zip_names)} → {base}.part1..part{numbers[-1]}",
                )
            except Exception as exc:
                errors.append(f"{base}: {exc}")

        combined, diagnostics = _join_tad_sources(
            property_frames,
            comp_frames,
            improvement_frames,
        )
        if combined.empty:
            return [], sources, errors, diagnostics
        return [combined], sources, errors, diagnostics
    finally:
        for zf in opened:
            try:
                zf.close()
            except Exception:
                pass
