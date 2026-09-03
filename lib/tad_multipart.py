from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict

import pandas as pd

from lib.tad_bulk import (
    TAD_BULK_COLUMNS,
    load_tad_dataframe,
    normalize_legal,
    target_house_numbers,
    target_legal_descriptions,
)

_PART_RE = re.compile(r"(?i)(.+\.txt)\.part(\d+)$")


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
    """Stream raw byte-split pipe-delimited files in part order.

    Rows are retained when either their situs house number matches a lead or
    their normalized legal description matches a legal-description-only lead.
    Split boundaries may occur inside a row, so trailing bytes are carried into
    the next part before parsing.
    """
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
                                "Split TAD property file does not contain Situs_Address or LegalDescription."
                            )
                        continue

                    fields = text.split("|")
                    if not should_keep(fields):
                        continue

                    row = {
                        col: fields[i] if i < len(fields) else ""
                        for i, col in wanted_positions
                    }
                    rows.append(row)

    if carry.strip():
        text = carry.rstrip(b"\r").decode("utf-8", errors="replace")
        if header is None:
            header = text.split("|")
        else:
            fields = text.split("|")
            if should_keep(fields):
                rows.append(
                    {
                        col: fields[i] if i < len(fields) else ""
                        for i, col in wanted_positions
                    }
                )

    if not rows:
        return pd.DataFrame(columns=TAD_BULK_COLUMNS)
    return pd.DataFrame(rows)


def load_tad_uploads(uploaded_files, target_addresses: list[str], max_mb: int = 200):
    """Load regular TAD files plus .txt.partN members spread across ZIP uploads."""
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    errors: list[str] = []
    house_targets = target_house_numbers(target_addresses)
    legal_targets = target_legal_descriptions(target_addresses)

    multipart_groups = defaultdict(list)
    regular_files = []
    opened = []

    try:
        for uploaded in uploaded_files:
            size_mb = len(uploaded.getvalue()) / 1024 / 1024
            if size_mb > max_mb:
                errors.append(
                    f"{uploaded.name}: upload is {size_mb:.1f} MB; maximum is {max_mb} MB"
                )
                continue

            if uploaded.name.lower().endswith(".zip"):
                zf, found = _part_members(uploaded)
                if found:
                    opened.append(zf)
                    for base, number, info in found:
                        multipart_groups[base].append(
                            (number, zf, info, uploaded.name)
                        )
                    continue
                if zf is not None:
                    zf.close()

            regular_files.append(uploaded)

        for uploaded in regular_files:
            try:
                df, source = load_tad_dataframe(
                    uploaded, target_addresses, max_mb
                )
                if not df.empty:
                    frames.append(df)
                sources.append(source)
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
                if not df.empty:
                    frames.append(df)

                zip_names = []
                for _, _, _, upload_name in members:
                    if upload_name not in zip_names:
                        zip_names.append(upload_name)

                sources.append(
                    f"{' + '.join(zip_names)} → {base}.part1..part{numbers[-1]}"
                )
            except Exception as exc:
                errors.append(f"{base}: {exc}")

        return frames, sources, errors
    finally:
        for zf in opened:
            try:
                zf.close()
            except Exception:
                pass
