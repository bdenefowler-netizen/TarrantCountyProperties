from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Optional

import pandas as pd
from rapidfuzz import fuzz, process

TAD_BULK_COLUMNS = [
    "Appraisal_Year", "Account_Num", "Owner_Name", "Owner_Address",
    "Owner_CityState", "Owner_Zip", "Situs_Address", "Property_Class",
    "State_Use_Code", "LegalDescription", "Deed_Date", "Land_Value",
    "Improvement_Value", "Total_Value", "Garage_Capacity", "Num_Bedrooms",
    "Num_Bathrooms", "Year_Built", "Living_Area", "Swimming_Pool_Ind",
    "Land_Acres", "Land_SqFt", "Central_Heat_Ind", "Central_Air_Ind",
    "Structure_Count", "Appraised_Value", "Instrument_No",
    "Gross_Building_Area",
]


def canonical_account(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(8) if digits else ""


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


def address_score(query: str, candidate: str) -> float:
    q, c = normalize_address(query), normalize_address(candidate)
    if not q or not c:
        return 0.0
    return round(fuzz.ratio(q, c) * 0.55 + fuzz.token_set_ratio(q, c) * 0.45, 1)


def target_house_numbers(addresses: list[str]) -> set[str]:
    out: set[str] = set()
    for address in addresses:
        m = re.match(r"^\s*(\d+)", str(address or ""))
        if m:
            out.add(m.group(1))
    return out


def _read_pipe(stream, targets: set[str]) -> pd.DataFrame:
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
        if "Situs_Address" not in chunk.columns:
            raise ValueError("TAD property file does not contain Situs_Address.")
        if targets:
            house = chunk["Situs_Address"].fillna("").str.extract(r"^\s*(\d+)", expand=False)
            chunk = chunk[house.isin(targets)]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=TAD_BULK_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def load_tad_dataframe(uploaded_file, target_addresses: list[str], max_mb: int = 200) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    size_mb = len(raw) / 1024 / 1024
    if size_mb > max_mb:
        raise ValueError(f"TAD upload is {size_mb:.1f} MB; maximum is {max_mb} MB.")
    name = uploaded_file.name
    lower = name.lower()
    targets = target_house_numbers(target_addresses)
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [x for x in zf.infolist() if x.filename.lower().endswith((".txt", ".csv"))]
            if not members:
                raise ValueError("ZIP does not contain a TXT/CSV property file.")
            member = max(members, key=lambda x: x.file_size)
            with zf.open(member) as stream:
                return _read_pipe(stream, targets), f"{name} → {member.filename}"
    if lower.endswith(".txt"):
        return _read_pipe(io.BytesIO(raw), targets), name
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False), name
    if lower.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl", dtype=object), name
    raise ValueError("TAD data must be ZIP, TXT, CSV, or XLSX.")


class BulkTADIndex:
    ADDRESS_CANDIDATES = ["Situs_Address", "Situs Address", "Property Address", "Address", "SITUS_ADDRESS"]
    ACCOUNT_CANDIDATES = ["Account_Num", "Account Number", "Account", "APN", "ACCOUNT_NUM"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.address_col = self._find(self.ADDRESS_CANDIDATES)
        self.account_col = self._find(self.ACCOUNT_CANDIDATES)
        if not self.address_col or not self.account_col:
            raise ValueError("Could not identify TAD situs-address/account columns.")
        working = self.df.dropna(subset=[self.address_col, self.account_col]).copy()
        working["__norm"] = working[self.address_col].map(normalize_address)
        working["__house"] = working["__norm"].str.extract(r"^(\d+)", expand=False).fillna("")
        working["__account"] = working[self.account_col].map(canonical_account)
        working = working[(working["__norm"] != "") & (working["__account"] != "")]
        self.working = working.reset_index(drop=True)
        self.exact: dict[str, int] = {}
        self.by_house: dict[str, list[int]] = {}
        self.by_account: dict[str, int] = {}
        for i, row in self.working.iterrows():
            self.exact.setdefault(row["__norm"], i)
            self.by_house.setdefault(row["__house"], []).append(i)
            self.by_account.setdefault(row["__account"], i)

    def _find(self, candidates: list[str]) -> Optional[str]:
        names = {str(c).strip().lower(): c for c in self.df.columns}
        for candidate in candidates:
            if candidate.lower() in names:
                return names[candidate.lower()]
        return None

    def lookup(self, address: str) -> tuple[str, str, float]:
        q = normalize_address(address)
        if not q:
            return "", "", 0.0
        exact_i = self.exact.get(q)
        if exact_i is not None:
            return (
                str(self.working.at[exact_i, "__account"]),
                str(self.working.at[exact_i, self.address_col]),
                100.0,
            )
        m = re.match(r"^(\d+)", q)
        house = m.group(1) if m else ""
        idxs = self.by_house.get(house, [])
        if not idxs:
            return "", "", 0.0
        choices = {i: self.working.at[i, "__norm"] for i in idxs}
        hit = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
        if not hit:
            return "", "", 0.0
        _, score, row_index = hit
        return (
            str(self.working.at[row_index, "__account"]),
            str(self.working.at[row_index, self.address_col]),
            float(score),
        )

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


def map_record(record: dict[str, Any], pdf_template: str) -> dict[str, Any]:
    account = canonical_account(record.get("Account_Num"))
    owner_mail = " ".join(
        str(record.get(k) or "").strip()
        for k in ("Owner_Address", "Owner_CityState", "Owner_Zip")
        if str(record.get(k) or "").strip()
    )
    out = {
        "Matched Address": record.get("Situs_Address", ""),
        "Tax Account/APN": account,
        "TAD Account Number": account,
        "Current Owner": record.get("Owner_Name", ""),
        "TAD Owner Mailing Address": owner_mail,
        "Land Use Code": record.get("State_Use_Code", ""),
        "Land Use Category": record.get("Property_Class", ""),
        "Legal Description": record.get("LegalDescription", ""),
        "TAD Deed Date": record.get("Deed_Date", ""),
        "Latest Recording Date": record.get("Deed_Date", ""),
        "TAD Instrument Number": record.get("Instrument_No", ""),
        "Latest Document ID": record.get("Instrument_No", ""),
        "Land Value": parse_money(record.get("Land_Value")),
        "Improvement Value": parse_money(record.get("Improvement_Value")),
        "Total Assessed Value": parse_money(record.get("Total_Value")),
        "Market Value": parse_money(record.get("Appraised_Value")) or parse_money(record.get("Total_Value")),
        "Parking": record.get("Garage_Capacity", ""),
        "Bedrooms": record.get("Num_Bedrooms", ""),
        "Bathrooms": record.get("Num_Bathrooms", ""),
        "Year Built": record.get("Year_Built", ""),
        "Total Structure Area": parse_money(record.get("Living_Area")) or parse_money(record.get("Gross_Building_Area")),
        "Pool": yes_no(record.get("Swimming_Pool_Ind")),
        "Lot Acres": record.get("Land_Acres", ""),
        "Lot Area Sq Ft": parse_money(record.get("Land_SqFt")),
        "Heating": yes_no(record.get("Central_Heat_Ind")),
        "Air Conditioning": yes_no(record.get("Central_Air_Ind")),
        "Units": record.get("Structure_Count", ""),
        "Current Tax Year": record.get("Appraisal_Year", ""),
        "TAD Verified": "YES" if account else "REVIEW",
        "TAD Property URL": pdf_template.format(account=account) if account else "",
        "County": "Tarrant",
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
                    note="No TAD account match was found in the uploaded bulk property data.",
                )
            record = self.tad_index.get_record(account) if self.tad_index is not None else {}
            if record:
                data = map_record(record, core.TAD_PROPERTY_PDF_TEMPLATE)
                candidate = data.get("Matched Address") or matched_address
                score = address_score(address, candidate) if candidate else index_score
                return core.SourceResult(
                    source=self.name,
                    source_url=data.get("TAD Property URL", ""),
                    query_address=address,
                    matched_address=candidate,
                    data=data,
                    confidence=score,
                    status="found",
                    note="Matched directly from uploaded TAD bulk data.",
                )
            fallback = core._ORIGINAL_TAD_PROPERTY_SOURCE(tad_index=None)
            return fallback.search(address, account_hint=account)

    return BulkTADPropertySource
