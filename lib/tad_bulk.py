from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Optional

import pandas as pd
from rapidfuzz import fuzz, process

TAD_BULK_COLUMNS = [
    "RP", "Appraisal_Year", "Account_Num", "Record_Type", "Sequence_No", "PIDN",
    "Owner_Name", "Owner_Address", "Owner_CityState", "Owner_Zip", "Owner_Zip4",
    "Owner_CRRT", "Situs_Address", "Property_Class", "TAD_Map", "MAPSCO",
    "Exemption_Code", "State_Use_Code", "LegalDescription", "Notice_Date",
    "County", "City", "School", "Num_Special_Dist", "Spec1", "Spec2", "Spec3",
    "Spec4", "Spec5", "Deed_Date", "Deed_Book", "Deed_Page", "Land_Value",
    "Improvement_Value", "Total_Value", "Garage_Capacity", "Num_Bedrooms",
    "Num_Bathrooms", "Year_Built", "Living_Area", "Swimming_Pool_Ind",
    "ARB_Indicator", "Ag_Code", "Land_Acres", "Land_SqFt", "Ag_Acres",
    "Ag_Value", "Central_Heat_Ind", "Central_Air_Ind", "Structure_Count",
    "From_Accts", "Appraisal_Date", "Appraised_Value", "GIS_Link",
    "Instrument_No", "Overlap_Flag", "Gross_Building_Area",
    "Total_Net_Rentable_Area",
]


def canonical_account(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return ""
    if len(digits) > 8:
        digits = digits[:8]
    return digits.zfill(8)


def normalize_address(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).upper().strip()
    replacements = {
        r"\bSTREET\b": "ST", r"\bAVENUE\b": "AVE", r"\bROAD\b": "RD",
        r"\bDRIVE\b": "DR", r"\bBOULEVARD\b": "BLVD", r"\bLANE\b": "LN",
        r"\bCOURT\b": "CT", r"\bCIRCLE\b": "CIR", r"\bPLACE\b": "PL",
        r"\bPARKWAY\b": "PKWY", r"\bHIGHWAY\b": "HWY",
        r"\bNORTH\b": "N", r"\bSOUTH\b": "S", r"\bEAST\b": "E", r"\bWEST\b": "W",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s#-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_legal(value: Any) -> str:
    """Normalize county/TAD legal descriptions without turning them into street addresses."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).upper().strip()
    # County foreclosure rows often append city/state after the legal description.
    text = text.split(",", 1)[0]
    replacements = {
        r"\bBLOCK\b": "BLK",
        r"\bSECTION\b": "SEC",
        r"\bADDITION\b": "ADDN",
        r"\bSUBDIVISION\b": "SUBD",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^A-Z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_legal_description(value: Any) -> bool:
    text = str(value or "").upper().strip()
    return bool(re.search(r"\b(LOT|BLOCK|BLK|TRACT|ABSTRACT|SURVEY|ADDITION|SUBDIVISION|ESTATES?)\b", text))


def address_score(query: str, candidate: str) -> float:
    q, c = normalize_address(query), normalize_address(candidate)
    if not q or not c:
        return 0.0
    return round(fuzz.ratio(q, c) * 0.55 + fuzz.token_set_ratio(q, c) * 0.45, 1)


def legal_score(query: str, candidate: str) -> float:
    q, c = normalize_legal(query), normalize_legal(candidate)
    if not q or not c:
        return 0.0
    return round(fuzz.ratio(q, c) * 0.35 + fuzz.token_set_ratio(q, c) * 0.65, 1)


def target_house_numbers(addresses: list[str]) -> set[str]:
    out: set[str] = set()
    for address in addresses:
        m = re.match(r"^\s*(\d+)", str(address or ""))
        if m:
            out.add(m.group(1))
    return out


def target_legal_descriptions(addresses: list[str]) -> set[str]:
    return {
        normalize_legal(value)
        for value in addresses
        if looks_like_legal_description(value) and normalize_legal(value)
    }


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    names = {re.sub(r"[\s_]+", "", str(c).strip().lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[\s_]+", "", candidate.strip().lower())
        if key in names:
            return names[key]
    return None


def _filter_to_targets(df: pd.DataFrame, house_targets: set[str], legal_targets: set[str]) -> pd.DataFrame:
    if df.empty or (not house_targets and not legal_targets):
        return df

    keep = pd.Series(False, index=df.index)
    address_col = _find_column(
        df,
        ["Situs_Address", "Situs Address", "SitusAddress", "Property Address", "Address", "Site Address"],
    )
    if address_col and house_targets:
        house = df[address_col].fillna("").astype(str).str.extract(r"^\s*(\d+)", expand=False)
        keep |= house.isin(house_targets)

    legal_col = _find_column(df, ["LegalDescription", "Legal Description", "Legal_Description", "LEGAL_DESCRIPTION"])
    if legal_col and legal_targets:
        legal_norm = df[legal_col].fillna("").astype(str).map(normalize_legal)
        keep |= legal_norm.isin(legal_targets)

    return df.loc[keep].copy()


def _read_pipe(stream, house_targets: set[str], legal_targets: set[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    wanted = set(TAD_BULK_COLUMNS)
    for chunk in pd.read_csv(
        stream,
        sep="|",
        dtype=str,
        usecols=lambda c: c in wanted,
        chunksize=75000,
        low_memory=False,
        on_bad_lines="skip",
    ):
        if "Situs_Address" not in chunk.columns and "LegalDescription" not in chunk.columns:
            raise ValueError("TAD property file does not contain Situs_Address or LegalDescription.")
        chunk = _filter_to_targets(chunk, house_targets, legal_targets)
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=TAD_BULK_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def _read_excel(raw_or_stream, house_targets: set[str], legal_targets: set[str]) -> pd.DataFrame:
    df = pd.read_excel(raw_or_stream, engine="openpyxl", dtype=object)
    return _filter_to_targets(df, house_targets, legal_targets)


def load_tad_dataframe(uploaded_file, target_addresses: list[str], max_mb: int = 200) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    size_mb = len(raw) / 1024 / 1024
    if size_mb > max_mb:
        raise ValueError(f"TAD upload is {size_mb:.1f} MB; maximum is {max_mb} MB.")
    name = uploaded_file.name
    lower = name.lower()
    house_targets = target_house_numbers(target_addresses)
    legal_targets = target_legal_descriptions(target_addresses)

    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            supported = (".txt", ".csv", ".xlsx")
            members = [x for x in zf.infolist() if x.filename.lower().endswith(supported)]
            if not members:
                raise ValueError("ZIP does not contain a supported TXT, CSV, or XLSX property file.")

            member = max(members, key=lambda x: x.file_size)
            member_lower = member.filename.lower()
            with zf.open(member) as stream:
                if member_lower.endswith(".xlsx"):
                    data = _read_excel(io.BytesIO(stream.read()), house_targets, legal_targets)
                elif member_lower.endswith(".csv"):
                    data = pd.read_csv(stream, dtype=str, low_memory=False)
                    data = _filter_to_targets(data, house_targets, legal_targets)
                else:
                    data = _read_pipe(stream, house_targets, legal_targets)
            return data, f"{name} → {member.filename}"

    if lower.endswith(".txt"):
        return _read_pipe(io.BytesIO(raw), house_targets, legal_targets), name
    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
        return _filter_to_targets(df, house_targets, legal_targets), name
    if lower.endswith(".xlsx"):
        return _read_excel(io.BytesIO(raw), house_targets, legal_targets), name
    raise ValueError("TAD data must be ZIP, TXT, CSV, or XLSX.")


class BulkTADIndex:
    ADDRESS_CANDIDATES = [
        "Situs_Address", "Situs Address", "SitusAddress", "Property Address",
        "Address", "Site Address", "SITUS_ADDRESS",
    ]
    LEGAL_CANDIDATES = [
        "LegalDescription", "Legal Description", "Legal_Description", "LEGAL_DESCRIPTION",
    ]
    ACCOUNT_CANDIDATES = [
        "Account_Num", "Account Number", "Account", "APN", "ACCOUNT_NUM",
        "PIN", "Pin", "Parcel ID", "Property ID",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.address_col = self._find(self.ADDRESS_CANDIDATES)
        self.legal_col = self._find(self.LEGAL_CANDIDATES)
        self.account_col = self._find(self.ACCOUNT_CANDIDATES)
        if not self.account_col or (not self.address_col and not self.legal_col):
            raise ValueError(
                "Could not identify TAD account and property-location columns. "
                "Expected Account_Num/PIN plus Situs_Address and/or LegalDescription."
            )

        working = self.df.copy()
        if self.address_col:
            working["__norm"] = working[self.address_col].map(normalize_address)
        else:
            working["__norm"] = ""
        if self.legal_col:
            working["__legal_norm"] = working[self.legal_col].map(normalize_legal)
        else:
            working["__legal_norm"] = ""
        working["__house"] = working["__norm"].str.extract(r"^(\d+)", expand=False).fillna("")
        working["__account"] = working[self.account_col].map(canonical_account)
        working = working[
            (working["__account"] != "")
            & ((working["__norm"] != "") | (working["__legal_norm"] != ""))
        ]
        self.working = working.reset_index(drop=True)

        self.exact: dict[str, int] = {}
        self.exact_legal: dict[str, int] = {}
        self.by_house: dict[str, list[int]] = {}
        self.by_account: dict[str, int] = {}
        for i, row in self.working.iterrows():
            if row["__norm"]:
                self.exact.setdefault(row["__norm"], i)
            if row["__legal_norm"]:
                self.exact_legal.setdefault(row["__legal_norm"], i)
            if row["__house"]:
                self.by_house.setdefault(row["__house"], []).append(i)
            self.by_account.setdefault(row["__account"], i)

    def _find(self, candidates: list[str]) -> Optional[str]:
        names = {re.sub(r"[\s_]+", "", str(c).strip().lower()): c for c in self.df.columns}
        for candidate in candidates:
            key = re.sub(r"[\s_]+", "", candidate.strip().lower())
            if key in names:
                return names[key]
        return None

    def _result(self, row_index: int, score: float) -> tuple[str, str, float]:
        matched_address = ""
        if self.address_col:
            matched_address = str(self.working.at[row_index, self.address_col] or "")
        return (
            str(self.working.at[row_index, "__account"]),
            matched_address,
            float(score),
        )

    def lookup(self, address_or_legal: str) -> tuple[str, str, float]:
        q_address = normalize_address(address_or_legal)
        q_legal = normalize_legal(address_or_legal)
        if not q_address and not q_legal:
            return "", "", 0.0

        exact_i = self.exact.get(q_address)
        if exact_i is not None:
            return self._result(exact_i, 100.0)

        exact_legal_i = self.exact_legal.get(q_legal)
        if exact_legal_i is not None:
            return self._result(exact_legal_i, 100.0)

        m = re.match(r"^(\d+)", q_address)
        house = m.group(1) if m else ""
        idxs = self.by_house.get(house, []) if house else []
        if idxs:
            choices = {i: self.working.at[i, "__norm"] for i in idxs}
            hit = process.extractOne(q_address, choices, scorer=fuzz.token_set_ratio)
            if hit:
                _, score, row_index = hit
                if float(score) >= 72.0:
                    return self._result(row_index, float(score))

        if q_legal and looks_like_legal_description(address_or_legal):
            choices = {
                i: self.working.at[i, "__legal_norm"]
                for i in self.working.index
                if self.working.at[i, "__legal_norm"]
            }
            hit = process.extractOne(q_legal, choices, scorer=fuzz.token_set_ratio)
            if hit:
                _, score, row_index = hit
                if float(score) >= 88.0:
                    return self._result(row_index, float(score))

        return "", "", 0.0

    def get_record(self, account: str) -> dict[str, Any]:
        i = self.by_account.get(canonical_account(account))
        if i is None:
            return {}
        row = self.working.loc[i]
        out = {}
        for key, value in row.items():
            if str(key).startswith("__"):
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            out[str(key)] = value
        return out


def parse_money(value: Any):
    if value in (None, ""):
        return None
    text = re.sub(r"[^0-9.-]", "", str(value))
    try:
        return int(round(float(text)))
    except (TypeError, ValueError):
        return None


def yes_no(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"Y", "YES", "1", "TRUE", "T"}:
        return "Yes"
    if text in {"N", "NO", "0", "FALSE", "F"}:
        return "No"
    return str(value or "").strip()


def _first(record: dict[str, Any], *keys: str) -> Any:
    normalized = {re.sub(r"[\s_]+", "", str(k).lower()): v for k, v in record.items()}
    for key in keys:
        value = normalized.get(re.sub(r"[\s_]+", "", key.lower()))
        if value not in (None, "") and not (isinstance(value, float) and pd.isna(value)):
            return value
    return ""


def map_record(record: dict[str, Any], pdf_template: str) -> dict[str, Any]:
    account = canonical_account(_first(record, "Account_Num", "PIN", "Account Number", "APN"))
    owner_mail = " ".join(
        str(_first(record, k) or "").strip()
        for k in ("Owner_Address", "Owner_CityState", "Owner_Zip")
        if str(_first(record, k) or "").strip()
    )
    out = {
        "Matched Address": _first(record, "Situs_Address", "Situs Address", "SitusAddress", "Property Address"),
        "Tax Account/APN": account,
        "TAD Account Number": account,
        "Current Owner": _first(record, "Owner_Name", "Owner Name", "Owner"),
        "TAD Owner Mailing Address": owner_mail,
        "Land Use Code": _first(record, "State_Use_Code", "State Use Code"),
        "Land Use Category": _first(record, "Property_Class", "Property Class", "Improvement Type", "Style"),
        "Exemptions": _first(record, "Exemption_Code", "Exemption Code"),
        "Legal Description": _first(record, "LegalDescription", "Legal Description"),
        "TAD Deed Date": _first(record, "Deed_Date", "Deed Date"),
        "Latest Recording Date": _first(record, "Deed_Date", "Deed Date"),
        "TAD Instrument Number": _first(record, "Instrument_No", "Instrument Number"),
        "Latest Document ID": _first(record, "Instrument_No", "Instrument Number"),
        "Land Value": parse_money(_first(record, "Land_Value", "Land Value")),
        "Improvement Value": parse_money(_first(record, "Improvement_Value", "Improvement Value")),
        "Total Assessed Value": parse_money(_first(record, "Total_Value", "Total Value", "Equity Indicated Value")),
        "Market Value": parse_money(_first(record, "Appraised_Value", "Market Indicated Value", "Total_Value", "Total Value")),
        "Parking": _first(record, "Garage_Capacity", "Garage Capacity", "Garage"),
        "Bedrooms": _first(record, "Num_Bedrooms", "Bedrooms"),
        "Bathrooms": _first(record, "Num_Bathrooms", "Bathrooms"),
        "Year Built": _first(record, "Year_Built", "Actual Year Built", "Year Built"),
        "Year Updated": _first(record, "Effective Year Built", "Effective_Year_Built"),
        "Total Structure Area": parse_money(_first(record, "Living_Area", "Main Area", "Gross_Building_Area", "Gross Building Area", "Total_Net_Rentable_Area", "Total Net Rentable Area")),
        "Stories": _first(record, "Stories", "Story", "Story Height"),
        "Pool": yes_no(_first(record, "Swimming_Pool_Ind", "Pool", "Pool Indicator")),
        "Lot Acres": _first(record, "Land_Acres", "Land Acres", "Acreage"),
        "Lot Area Sq Ft": parse_money(_first(record, "Land_SqFt", "Land Sq Ft", "Land Area")),
        "Heating": yes_no(_first(record, "Central_Heat_Ind", "Central Heat")),
        "Air Conditioning": yes_no(_first(record, "Central_Air_Ind", "Central Air")),
        "Units": _first(record, "Structure_Count", "Structure Count"),
        "Structure Quality": _first(record, "Quality"),
        "Structure Condition": _first(record, "Condition"),
        "Improvements": _first(record, "Improvement Details", "Improvements"),
        "Current Tax Year": _first(record, "Appraisal_Year", "Appraisal Year", "Tax Year", "TaxYear"),
        "TAD Verified": "YES" if account else "REVIEW",
        "TAD Property URL": pdf_template.format(account=account) if account else "",
        "County": _first(record, "County") or "Tarrant",
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def make_bulk_source_class(core):
    class BulkTADPropertySource(core.PropertySource):
        name = "Tarrant Appraisal District"

        def __init__(self, tad_index=None):
            self.tad_index = tad_index

        def search(self, address: str, account_hint: str = ""):
            account = canonical_account(account_hint)
            matched_address = ""
            index_score = 0.0
            if not account and self.tad_index is not None:
                account, matched_address, index_score = self.tad_index.lookup(address)
            if not account:
                return core.SourceResult(
                    source=self.name,
                    query_address=address,
                    status="needs_account",
                    note="No TAD account match was found in the uploaded bulk property data by situs address or legal description.",
                )
            record = self.tad_index.get_record(account) if self.tad_index is not None else {}
            if record:
                data = map_record(record, core.TAD_PROPERTY_PDF_TEMPLATE)
                candidate = data.get("Matched Address") or matched_address
                score = index_score
                if candidate and not looks_like_legal_description(address):
                    score = address_score(address, candidate)
                return core.SourceResult(
                    source=self.name,
                    source_url=data.get("TAD Property URL", ""),
                    query_address=address,
                    matched_address=candidate,
                    data=data,
                    confidence=score,
                    status="found",
                    note="Matched from uploaded TAD bulk data by situs address or legal description.",
                )
            fallback = core._ORIGINAL_TAD_PROPERTY_SOURCE(tad_index=None)
            return fallback.search(address, account_hint=account)

    return BulkTADPropertySource
