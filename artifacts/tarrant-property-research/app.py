from __future__ import annotations

import io
import json
import os
import re
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz, process


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Tarrant County Property Research"
MAX_UPLOAD_MB = 200
REQUEST_TIMEOUT = 20
APP_VERSION = "2.0"
TAD_PROPERTY_PDF_TEMPLATE = "https://www.tad.org/export/property-pdf?account={account}"
PUBRECORD_PROPERTY_URL = "https://www.pubrecord.org/property-records/?q={address}"

REQUIRED_COLUMNS = [
    "Grantor/Owner",
    "Sale Date",
    "Filed Date",
    "Property Address",
]

# The application appends these fields. Original input fields remain untouched.
RESEARCH_COLUMNS = [
    "Matched Address",
    "Address Match Score",
    "Match Status",
    "Research Status",
    "Research Timestamp",
    "Sources",
    "Source URLs",
    "Missing Data Flag",
    "Uncertain Match Flag",

    "Carrier Code",
    "Census Tract",
    "Lot Area Sq Ft",
    "Lot Acres",
    "Land Use Code",
    "Land Use Category",
    "County",
    "Subdivision",
    "Legal Description",
    "Tax Account/APN",
    "Lot Number",

    "Year Built",
    "Year Updated",
    "Total Structure Area",
    "Stories",
    "Bedrooms",
    "Bathrooms",
    "Units",
    "Parking",
    "Structure Quality",
    "Structure Condition",
    "Improvements",
    "Pool",
    "Construction",
    "Heating",
    "Air Conditioning",

    "Current Owner",
    "Buyer Information",

    "Latest Document ID",
    "Latest Recording Date",
    "Latest Contract Date",
    "Latest Sale Price",
    "Latest Sale Type",

    "Lender",
    "Loan Amount",
    "Loan Type",
    "Loan Due Date",

    "Current Tax Year",
    "Property Tax Amount",
    "Exemptions",
    "Land Value",
    "Improvement Value",
    "Total Assessed Value",
    "Market Value",

    # JSON fields retain multiple historical records without losing detail.
    "Sales History JSON",
    "Mortgage/Deed/Loan History JSON",
    "Property Tax History JSON",
    "Assessment History JSON",

    "Research Notes",
    "TAD Account Number",
    "TAD Owner Mailing Address",
    "TAD Deed Date",
    "TAD Instrument Number",
    "TAD Notice Value",
    "TAD Property URL",
    "TAD Verified",
    "Foreclosure Verified",
    "Foreclosure Status",
    "Scheduled Auction",
    "Later Sale Found",
    "Likely Resolved",
    "Lead Action",
    "County Sale Date",
    "County Cause Number",
    "County Purchaser",
    "County Sale Amount",
    "County Status Source",
    "Distress Score",
    "Research Priority",
]


# ============================================================
# Data models
# ============================================================

@dataclass
class SourceResult:
    source: str
    source_url: str = ""
    query_address: str = ""
    matched_address: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    status: str = "not_found"
    note: str = ""


# ============================================================
# Helpers
# ============================================================

def clean_scalar(value: Any) -> Any:
    """Convert source values to safe scalar/JSON-friendly forms."""
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)

    return value


def normalize_address(value: Any) -> str:
    """
    Normalize addresses for comparison only.
    Does NOT modify the user's original Property Address field.
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).upper().strip()

    replacements = {
        r"\bSTREET\b": "ST",
        r"\bAVENUE\b": "AVE",
        r"\bROAD\b": "RD",
        r"\bDRIVE\b": "DR",
        r"\bBOULEVARD\b": "BLVD",
        r"\bLANE\b": "LN",
        r"\bCOURT\b": "CT",
        r"\bCIRCLE\b": "CIR",
        r"\bPLACE\b": "PL",
        r"\bPARKWAY\b": "PKWY",
        r"\bHIGHWAY\b": "HWY",
        r"\bNORTH\b": "N",
        r"\bSOUTH\b": "S",
        r"\bEAST\b": "E",
        r"\bWEST\b": "W",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^\w\s#-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def address_score(query: str, candidate: str) -> float:
    q = normalize_address(query)
    c = normalize_address(candidate)

    if not q or not c:
        return 0.0

    return round(
        (
            fuzz.ratio(q, c) * 0.55
            + fuzz.token_set_ratio(q, c) * 0.45
        ),
        1,
    )


def classify_match(score: float) -> str:
    if score >= 92:
        return "High confidence"
    if score >= 80:
        return "Review"
    if score > 0:
        return "Low confidence"
    return "No match"


def safe_json(value: Any) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def create_source_links(address: str) -> dict[str, str]:
    """
    Review/search links rather than undocumented scraping endpoints.

    If a site changes its public search URL, modify it here after checking
    the source's current terms and technical documentation.
    """
    encoded = quote_plus(address)

    return {
        "Tarrant Appraisal District":
            "https://www.tad.org/",
        "Tarrant County":
            "https://www.tarrantcountytx.gov/",
        "PubRecord":
            PUBRECORD_PROPERTY_URL.format(address=encoded),
    }


def file_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# Source adapters
# ============================================================

class PropertySource:
    name = "Base source"

    def search(self, address: str) -> SourceResult:
        raise NotImplementedError


# Input-header aliases let the app accept original county exports and cleaned workbooks.
COLUMN_ALIASES = {
    "Grantor/Owner": ["Grantor/Owner", "Grantor / Owner", "Grantor", "Owner"],
    "Sale Date": ["Sale Date", "Sale Month", "Auction Date", "Sale"],
    "Filed Date": ["Filed Date", "Filing Date", "Recorded Date"],
    "Property Address": ["Property Address", "Property / Legal Description", "Address", "Situs Address", "Property/Legal Description"],
}


def resolve_input_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        if alias.strip().lower() in normalized:
            return normalized[alias.strip().lower()]
    return None


def canonical_account(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(8) if digits else ""


def parse_money(value: str) -> Optional[int]:
    if not value:
        return None
    digits = re.sub(r"[^0-9.-]", "", value)
    try:
        return int(round(float(digits)))
    except (TypeError, ValueError):
        return None


def extract_tad_pdf_text(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("TAD PDF parsing requires pypdf. Install with: pip install pypdf")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _rx(text: str, pattern: str, flags=re.I | re.M) -> str:
    match = re.search(pattern, text, flags)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def parse_tad_property_pdf(text: str, account: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    compact = text.replace("\r", "")
    address = _rx(compact, r"Address:\s*(.+?)\s*(?:\n|City:)")
    city = _rx(compact, r"City:\s*(.+?)\s*(?:\n|Georeference:|Subdivision:)")
    matched = ", ".join(x for x in [address, city, "TX"] if x)
    if matched:
        out["Matched Address"] = matched
    out["Tax Account/APN"] = canonical_account(_rx(compact, r"Account Number:\s*([0-9-]+)") or account)
    out["TAD Account Number"] = out["Tax Account/APN"]
    out["Subdivision"] = _rx(compact, r"Subdivision:\s*(.+?)\s*(?:\n|Neighborhood Code:)")
    out["Legal Description"] = _rx(compact, r"Legal Description:\s*(.+?)\s*(?:\n|Jurisdictions:)", re.I | re.M | re.S)
    out["Land Use Code"] = _rx(compact, r"State Code:\s*([^\n]+)")
    out["Land Use Category"] = _rx(compact, r"Site Class:\s*([^\n]+)")
    out["Year Built"] = _rx(compact, r"Year Built:\s*([0-9]{4})")
    out["Lot Area Sq Ft"] = parse_money(_rx(compact, r"Land Sqft[^:]*:\s*([0-9,]+)"))
    acres = _rx(compact, r"Land Acres[^:]*:\s*([0-9.]+)")
    if acres:
        try: out["Lot Acres"] = float(acres)
        except ValueError: pass
    pool = _rx(compact, r"Pool:\s*([YN])")
    if pool: out["Pool"] = "Yes" if pool.upper() == "Y" else "No"
    owner = _rx(compact, r"Current Owner:\s*(.+?)\s*(?:Primary Owner Address:|Deed Date:)", re.I | re.M | re.S)
    if owner: out["Current Owner"] = owner
    owner_addr = _rx(compact, r"Primary Owner Address:\s*(.+?)\s*(?:Deed Date:|$)", re.I | re.M | re.S)
    if owner_addr: out["TAD Owner Mailing Address"] = owner_addr
    deed_date = _rx(compact, r"Deed Date:\s*([^\n]+)")
    instrument = _rx(compact, r"Instrument:\s*([^\n]+)")
    notice_value = parse_money(_rx(compact, r"Notice Value:\s*\$?([0-9,]+)"))
    if deed_date:
        out["TAD Deed Date"] = deed_date
        out["Latest Recording Date"] = deed_date
    if instrument:
        out["TAD Instrument Number"] = instrument
        out["Latest Document ID"] = instrument
    if notice_value is not None: out["TAD Notice Value"] = notice_value
    value_match = re.search(r"(?:VALUES.*?)?\b(20\d{2})\s+\$?([0-9,]+)\s+\$?([0-9,]+)\s+\$?([0-9,]+)\s+\$?([0-9,]+)", compact, re.I | re.S)
    if value_match:
        out["Current Tax Year"] = int(value_match.group(1))
        out["Improvement Value"] = parse_money(value_match.group(2))
        out["Land Value"] = parse_money(value_match.group(3))
        out["Market Value"] = parse_money(value_match.group(4))
        out["Total Assessed Value"] = parse_money(value_match.group(5))
    area = _rx(compact, r"(?:Gross Building Area|Net Leasable Area|Living Area|Building Area)[^:]*:\s*([0-9,]+)")
    if area: out["Total Structure Area"] = parse_money(area)
    out["County"] = "Tarrant"
    out["TAD Verified"] = "YES" if out.get("Tax Account/APN") else "REVIEW"
    out["TAD Property URL"] = TAD_PROPERTY_PDF_TEMPLATE.format(account=account)
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


class TADIndex:
    ADDRESS_CANDIDATES = ["Property Address", "Situs Address", "Address", "Site Address", "Situs", "Location Address", "PROPERTY_ADDRESS", "SITUS_ADDRESS"]
    ACCOUNT_CANDIDATES = ["Account", "Account Number", "Account No", "TAD Account Number", "Tax Account/APN", "APN", "ACCOUNT", "ACCOUNT_NUM"]
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.address_col = self._find(self.ADDRESS_CANDIDATES)
        self.account_col = self._find(self.ACCOUNT_CANDIDATES)
        if not self.address_col or not self.account_col:
            raise ValueError("Could not identify TAD address/account columns.")
        working = self.df[[self.address_col, self.account_col]].dropna().copy()
        working["__norm"] = working[self.address_col].map(normalize_address)
        working["__house"] = working["__norm"].str.extract(r"^(\d+)", expand=False).fillna("")
        working["__account"] = working[self.account_col].map(canonical_account)
        working = working[(working["__norm"] != "") & (working["__account"] != "")]
        self.working = working.reset_index(drop=True)
        self.exact = dict(zip(self.working["__norm"], self.working["__account"]))
        self.by_house: dict[str, list[int]] = {}
        for i, house in enumerate(self.working["__house"]): self.by_house.setdefault(house, []).append(i)
    def _find(self, candidates: list[str]) -> Optional[str]:
        norm = {str(c).strip().lower(): c for c in self.df.columns}
        for c in candidates:
            if c.lower() in norm: return norm[c.lower()]
        return None
    def lookup(self, address: str) -> tuple[str, str, float]:
        q = normalize_address(address)
        if not q: return "", "", 0.0
        if q in self.exact: return self.exact[q], address, 100.0
        m = re.match(r"^(\d+)", q); house = m.group(1) if m else ""
        idxs = self.by_house.get(house, [])
        if not idxs: return "", "", 0.0
        choices = {i: self.working.at[i, "__norm"] for i in idxs}
        hit = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
        if not hit: return "", "", 0.0
        _, score, row_index = hit
        return str(self.working.at[row_index, "__account"]), str(self.working.at[row_index, self.address_col]), float(score)


class TADPropertySource(PropertySource):
    name = "Tarrant Appraisal District"
    def __init__(self, tad_index: Optional[TADIndex] = None):
        self.tad_index = tad_index
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TarrantPropertyResearch/2.0 (public-record research)", "Accept": "application/pdf,text/plain,*/*"})
    def search(self, address: str, account_hint: str = "") -> SourceResult:
        account = canonical_account(account_hint); index_address = ""; index_score = 0.0
        if not account and self.tad_index is not None:
            account, index_address, index_score = self.tad_index.lookup(address)
        if not account:
            return SourceResult(source=self.name, query_address=address, status="needs_account", note="No TAD account number available. Upload a public TAD data export/index or include an APN/account column.")
        url = TAD_PROPERTY_PDF_TEMPLATE.format(account=account)
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT); response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                return SourceResult(source=self.name, source_url=url, query_address=address, status="error", note="TAD did not return a PDF for this account.")
            parsed = parse_tad_property_pdf(extract_tad_pdf_text(response.content), account)
        except Exception as exc:
            return SourceResult(source=self.name, source_url=url, query_address=address, status="error", note=f"{type(exc).__name__}: {exc}")
        candidate = parsed.get("Matched Address") or index_address
        score = address_score(address, candidate) if candidate else index_score
        if not score and index_score: score = index_score
        return SourceResult(source=self.name, source_url=url, query_address=address, matched_address=candidate, data=parsed, confidence=score, status="found" if parsed else "not_found", note="Official TAD property PDF parsed by account number.")


class PubRecordReviewSource(PropertySource):
    name = "PubRecord"
    def search(self, address: str) -> SourceResult:
        url = PUBRECORD_PROPERTY_URL.format(address=quote_plus(address))
        return SourceResult(source=self.name, source_url=url, query_address=address, status="manual_review", note="Manual fallback only; PubRecord currently limits searches to 5/day.")


class ConfigurableJSONSource(PropertySource):
    """
    Generic adapter for a documented/licensed JSON property API.

    Environment variables:
        PROPERTY_API_URL_TEMPLATE
        PROPERTY_API_KEY
        PROPERTY_API_KEY_HEADER

    Example template:
        https://provider.example/api/property?address={address}

    The template must be supplied by the operator based on a provider
    they are authorized to use.
    """

    name = "Configured Property API"

    def __init__(self):
        self.template = os.getenv("PROPERTY_API_URL_TEMPLATE", "").strip()
        self.api_key = os.getenv("PROPERTY_API_KEY", "").strip()
        self.key_header = os.getenv(
            "PROPERTY_API_KEY_HEADER",
            "Authorization",
        ).strip()

    @property
    def enabled(self) -> bool:
        return bool(self.template)

    def search(self, address: str) -> SourceResult:
        if not self.enabled:
            return SourceResult(
                source=self.name,
                query_address=address,
                status="disabled",
                note="PROPERTY_API_URL_TEMPLATE is not configured.",
            )

        url = self.template.format(address=quote_plus(address))

        headers = {
            "User-Agent":
                "TarrantPropertyResearch/1.0 "
                "(authorized public-record research)",
            "Accept": "application/json",
        }

        if self.api_key:
            headers[self.key_header] = self.api_key

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()

        except requests.RequestException as exc:
            return SourceResult(
                source=self.name,
                source_url=url,
                query_address=address,
                status="error",
                note=str(exc),
            )

        except ValueError:
            return SourceResult(
                source=self.name,
                source_url=url,
                query_address=address,
                status="error",
                note="Provider did not return JSON.",
            )

        normalized = map_generic_provider(payload)

        candidate = (
            normalized.get("Matched Address")
            or normalized.get("Property Address")
            or ""
        )

        score = address_score(address, candidate)

        return SourceResult(
            source=self.name,
            source_url=url,
            query_address=address,
            matched_address=candidate,
            data=normalized,
            confidence=score,
            status="found" if normalized else "not_found",
        )


def nested_get(obj: dict, *paths: str) -> Any:
    """
    Try multiple dotted JSON paths.

    This helps connect a JSON provider with minimal changes.
    """
    for path in paths:
        current: Any = obj

        try:
            for key in path.split("."):
                if isinstance(current, dict):
                    current = current[key]
                else:
                    current = None
                    break

            if current not in (None, ""):
                return current
        except (KeyError, TypeError):
            continue

    return None


def map_generic_provider(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Example normalization layer.

    Update aliases here to match the schema of the permitted API/source
    being used. Downstream UI/export code remains unchanged.
    """

    # Some APIs wrap a single property in `property` or `result`.
    record = (
        payload.get("property")
        or payload.get("result")
        or payload.get("data")
        or payload
    )

    if isinstance(record, list):
        record = record[0] if record else {}

    if not isinstance(record, dict):
        return {}

    result = {
        "Matched Address": nested_get(
            record,
            "address",
            "propertyAddress",
            "location.address",
            "siteAddress",
        ),
        "Carrier Code": nested_get(
            record,
            "carrierCode",
            "carrier_route",
        ),
        "Census Tract": nested_get(
            record,
            "censusTract",
            "census.tract",
        ),
        "Lot Area Sq Ft": nested_get(
            record,
            "lot.squareFeet",
            "lotAreaSquareFeet",
            "lotSizeSqFt",
        ),
        "Lot Acres": nested_get(
            record,
            "lot.acres",
            "lotAcres",
        ),
        "Land Use Code": nested_get(
            record,
            "landUse.code",
            "landUseCode",
        ),
        "Land Use Category": nested_get(
            record,
            "landUse.category",
            "landUseDescription",
        ),
        "County": nested_get(
            record,
            "county",
            "location.county",
        ),
        "Subdivision": nested_get(
            record,
            "subdivision",
            "legal.subdivision",
        ),
        "Legal Description": nested_get(
            record,
            "legalDescription",
            "legal.description",
        ),
        "Tax Account/APN": nested_get(
            record,
            "apn",
            "parcelNumber",
            "taxAccount",
            "accountNumber",
        ),
        "Lot Number": nested_get(
            record,
            "lotNumber",
            "legal.lot",
        ),
        "Year Built": nested_get(
            record,
            "yearBuilt",
            "building.yearBuilt",
        ),
        "Year Updated": nested_get(
            record,
            "yearUpdated",
            "building.yearUpdated",
            "building.effectiveYearBuilt",
        ),
        "Total Structure Area": nested_get(
            record,
            "building.squareFeet",
            "buildingArea",
            "livingArea",
        ),
        "Stories": nested_get(
            record,
            "building.stories",
            "stories",
        ),
        "Bedrooms": nested_get(
            record,
            "building.bedrooms",
            "bedrooms",
        ),
        "Bathrooms": nested_get(
            record,
            "building.bathrooms",
            "bathrooms",
        ),
        "Units": nested_get(
            record,
            "building.units",
            "units",
        ),
        "Parking": nested_get(
            record,
            "building.parking",
            "parking",
        ),
        "Structure Quality": nested_get(
            record,
            "building.quality",
            "quality",
        ),
        "Structure Condition": nested_get(
            record,
            "building.condition",
            "condition",
        ),
        "Improvements": nested_get(
            record,
            "improvements",
        ),
        "Pool": nested_get(
            record,
            "building.pool",
            "pool",
        ),
        "Construction": nested_get(
            record,
            "building.construction",
            "construction",
        ),
        "Heating": nested_get(
            record,
            "building.heating",
            "heating",
        ),
        "Air Conditioning": nested_get(
            record,
            "building.cooling",
            "airConditioning",
        ),
        "Current Owner": nested_get(
            record,
            "owner.name",
            "ownerName",
            "currentOwner",
        ),
        "Buyer Information": nested_get(
            record,
            "sale.buyer",
            "buyer",
        ),
        "Latest Document ID": nested_get(
            record,
            "sale.documentId",
            "latestSale.documentNumber",
        ),
        "Latest Recording Date": nested_get(
            record,
            "sale.recordingDate",
            "latestSale.recordingDate",
        ),
        "Latest Contract Date": nested_get(
            record,
            "sale.contractDate",
            "latestSale.saleDate",
        ),
        "Latest Sale Price": nested_get(
            record,
            "sale.price",
            "latestSale.price",
        ),
        "Latest Sale Type": nested_get(
            record,
            "sale.type",
            "latestSale.type",
        ),
        "Lender": nested_get(
            record,
            "mortgage.lender",
            "loan.lender",
        ),
        "Loan Amount": nested_get(
            record,
            "mortgage.amount",
            "loan.amount",
        ),
        "Loan Type": nested_get(
            record,
            "mortgage.type",
            "loan.type",
        ),
        "Loan Due Date": nested_get(
            record,
            "mortgage.dueDate",
            "loan.dueDate",
        ),
        "Current Tax Year": nested_get(
            record,
            "tax.year",
            "assessment.year",
        ),
        "Property Tax Amount": nested_get(
            record,
            "tax.amount",
            "propertyTax",
        ),
        "Exemptions": nested_get(
            record,
            "tax.exemptions",
            "exemptions",
        ),
        "Land Value": nested_get(
            record,
            "assessment.landValue",
            "valuation.land",
        ),
        "Improvement Value": nested_get(
            record,
            "assessment.improvementValue",
            "valuation.improvements",
        ),
        "Total Assessed Value": nested_get(
            record,
            "assessment.totalValue",
            "assessedValue",
        ),
        "Market Value": nested_get(
            record,
            "assessment.marketValue",
            "marketValue",
        ),
        "Sales History JSON": safe_json(
            nested_get(
                record,
                "salesHistory",
                "sales",
            )
        ),
        "Mortgage/Deed/Loan History JSON": safe_json(
            nested_get(
                record,
                "mortgageHistory",
                "loanHistory",
                "deeds",
            )
        ),
        "Property Tax History JSON": safe_json(
            nested_get(
                record,
                "taxHistory",
                "tax.history",
            )
        ),
        "Assessment History JSON": safe_json(
            nested_get(
                record,
                "assessmentHistory",
                "assessments",
            )
        ),
    }

    return {
        key: clean_scalar(value)
        for key, value in result.items()
        if value not in (None, "", [], {})
    }


# ============================================================
# Research engine
# ============================================================

def empty_research_row() -> dict[str, Any]:
    return {column: "" for column in RESEARCH_COLUMNS}


def merge_source_results(
    address: str,
    source_results: list[SourceResult],
) -> dict[str, Any]:
    result = empty_research_row()

    successful = [
        r for r in source_results
        if r.status == "found"
    ]

    if not successful:
        result["Research Status"] = "No automated record found"
        result["Match Status"] = "No match"
        result["Missing Data Flag"] = "YES"
        result["Uncertain Match Flag"] = "YES"
        result["Research Timestamp"] = datetime.now().isoformat(
            timespec="seconds"
        )

        review_links = create_source_links(address)
        result["Sources"] = "; ".join(review_links.keys())
        result["Source URLs"] = "; ".join(review_links.values())
        result["Research Notes"] = (
            "No configured automated source returned a record. "
            "Manual source-review links supplied."
        )
        return result

    # Best match first.
    successful.sort(key=lambda x: x.confidence, reverse=True)
    best = successful[0]

    for source_result in successful:
        for key, value in source_result.data.items():
            if key in result and result[key] in ("", None):
                result[key] = clean_scalar(value)

    result["Matched Address"] = (
        best.matched_address
        or best.data.get("Matched Address", "")
    )
    result["Address Match Score"] = best.confidence
    result["Match Status"] = classify_match(best.confidence)
    result["Uncertain Match Flag"] = (
        "YES" if best.confidence < 92 else "NO"
    )

    result["Sources"] = "; ".join(
        dict.fromkeys(r.source for r in successful)
    )

    result["Source URLs"] = "; ".join(
        dict.fromkeys(
            r.source_url
            for r in successful
            if r.source_url
        )
    )

    result["Research Status"] = "Researched"
    result["Research Timestamp"] = datetime.now().isoformat(
        timespec="seconds"
    )

    important = [
        "Tax Account/APN",
        "Current Owner",
        "Total Assessed Value",
        "Market Value",
        "Year Built",
    ]

    missing = [
        field
        for field in important
        if result.get(field) in ("", None)
    ]

    result["Missing Data Flag"] = "YES" if missing else "NO"

    if missing:
        result["Research Notes"] = (
            "Missing key fields: " + ", ".join(missing)
        )

    return result


def research_property(address: str, sources: list[PropertySource], account_hint: str = "") -> dict[str, Any]:
    source_results = []
    for source in sources:
        try:
            if isinstance(source, TADPropertySource): source_results.append(source.search(address, account_hint=account_hint))
            else: source_results.append(source.search(address))
        except Exception as exc:
            source_results.append(SourceResult(source=source.name, query_address=address, status="error", note=f"{type(exc).__name__}: {exc}"))
        time.sleep(0.05)
    result = merge_source_results(address, source_results)
    extra_urls = [r.source_url for r in source_results if r.source_url]
    if extra_urls:
        current = [u for u in str(result.get("Source URLs", "")).split("; ") if u]
        result["Source URLs"] = "; ".join(dict.fromkeys(current + extra_urls))
    notes = [r.note for r in source_results if r.note and r.status != "found"]
    if notes:
        existing = str(result.get("Research Notes", "")).strip()
        result["Research Notes"] = " | ".join(x for x in [existing, *notes] if x)
    return result


def research_dataframe(input_df: pd.DataFrame, progress_callback=None, tad_index: Optional[TADIndex] = None) -> pd.DataFrame:
    original = input_df.copy(deep=True)
    base_columns = [c for c in original.columns if c not in RESEARCH_COLUMNS]
    original = original[base_columns]
    address_col = resolve_input_column(original, "Property Address")
    account_col = next((c for c in original.columns if str(c).strip().lower() in {"tad account number", "tax account/apn", "account number", "apn"}), None)
    sources: list[PropertySource] = [TADPropertySource(tad_index=tad_index)]
    configured = ConfigurableJSONSource()
    if configured.enabled: sources.append(configured)
    sources.append(PubRecordReviewSource())
    enrichment = []; total = len(original)
    for position, (_, row) in enumerate(original.iterrows(), start=1):
        raw_address = row.get(address_col, "") if address_col else ""
        address = "" if pd.isna(raw_address) else str(raw_address)
        account_hint = ""
        if account_col:
            raw_account = row.get(account_col, "")
            if not pd.isna(raw_account): account_hint = str(raw_account)
        is_legal_only = bool(re.match(r"^\s*(LOT|BEING LOT|TRACT|BLOCK)\b", address, re.I))
        if not address.strip():
            researched = empty_research_row(); researched["Research Status"] = "Missing address"; researched["Missing Data Flag"] = "YES"; researched["Uncertain Match Flag"] = "YES"; researched["Research Timestamp"] = datetime.now().isoformat(timespec="seconds")
        elif is_legal_only and not account_hint:
            researched = empty_research_row(); researched["Research Status"] = "Needs address/account"; researched["Missing Data Flag"] = "YES"; researched["Uncertain Match Flag"] = "YES"; researched["Research Timestamp"] = datetime.now().isoformat(timespec="seconds"); researched["Sources"] = "Tarrant Appraisal District; PubRecord"; researched["Research Notes"] = "Input appears to be a legal description rather than a street address. Resolve the situs address or TAD account before automated matching."
        else:
            researched = research_property(address, sources, account_hint=account_hint)
        enrichment.append(researched)
        if progress_callback: progress_callback(position, total, address, researched["Research Status"])
    enrichment_df = pd.DataFrame(enrichment, columns=RESEARCH_COLUMNS)
    return pd.concat([original.reset_index(drop=True), enrichment_df.reset_index(drop=True)], axis=1)


# ============================================================
# Excel handling
# ============================================================

def validate_upload(uploaded_file) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = uploaded_file.name.lower()

    # Intentionally reject macro-enabled Excel.
    if not filename.endswith(".xlsx"):
        return False, "Only .xlsx workbooks are accepted."

    size_mb = uploaded_file.size / 1024 / 1024

    if size_mb > MAX_UPLOAD_MB:
        return (
            False,
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.",
        )

    return True, ""


def read_excel_safely(raw: bytes) -> pd.DataFrame:
    """
    Reads values only. No macros or formulas are executed by Python.
    """
    if raw[:2] != b"PK":
        raise ValueError("The uploaded file is not a valid .xlsx workbook.")

    stream = io.BytesIO(raw)

    return pd.read_excel(
        stream,
        engine="openpyxl",
        dtype=object,
    )


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if resolve_input_column(df, column) is None]


def excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            sheet_name="Enriched Properties",
            index=False,
        )

        workbook = writer.book
        sheet = writer.sheets["Enriched Properties"]

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

        # Moderate, readable widths.
        for column_cells in sheet.columns:
            letter = column_cells[0].column_letter
            max_len = 0

            for cell in column_cells[:200]:
                if cell.value is not None:
                    max_len = max(
                        max_len,
                        len(str(cell.value)),
                    )

            sheet.column_dimensions[letter].width = min(
                max(max_len + 2, 12),
                45,
            )

        # Optional metadata sheet.
        metadata = workbook.create_sheet("Research Metadata")
        metadata.append(["Exported", datetime.now().isoformat()])
        metadata.append(["County", "Tarrant County, Texas"])
        metadata.append(
            [
                "Notice",
                (
                    "Verify material property, title, tax, sale, and "
                    "mortgage information against authoritative records."
                ),
            ]
        )

    return output.getvalue()


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏠",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1550px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        [data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            padding: 14px 16px;
            border-radius: 10px;
            background: #ffffff;
        }

        .research-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            margin-bottom: 14px;
        }

        .muted {
            color: #667085;
            font-size: 0.9rem;
        }

        .status-good {
            color: #067647;
            font-weight: 600;
        }

        .status-review {
            color: #b54708;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_optional_tad_index(uploaded_file) -> Optional[TADIndex]:
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            tad_df = pd.read_csv(io.BytesIO(raw), dtype=object, low_memory=False)
        elif name.endswith(".xlsx"):
            tad_df = pd.read_excel(io.BytesIO(raw), engine="openpyxl", dtype=object)
        else:
            raise ValueError("TAD index must be CSV or XLSX.")
        return TADIndex(tad_df)
    except Exception as exc:
        raise ValueError(f"Unable to use TAD index: {exc}") from exc


def show_dashboard():
    st.title("Tarrant County Property Research")
    st.caption(f"v{APP_VERSION} • Foreclosure-list cleanup + TAD enrichment + PubRecord review fallback")
    upload_tab, results_tab, help_tab, privacy_tab = st.tabs(["Research", "Results", "Help", "Privacy"])
    with upload_tab:
        st.subheader("Upload property list")
        st.info("Accepts the original county export headers or the cleaned workbook headers. Existing input columns and values are preserved; research fields are appended.")
        uploaded = st.file_uploader("Property / foreclosure workbook", type=["xlsx"], help=f"Maximum size: {MAX_UPLOAD_MB} MB.")
        tad_upload = st.file_uploader("Optional: public TAD data export/index (CSV or XLSX)", type=["csv", "xlsx"], help="Recommended for bulk address-to-account matching. The app then retrieves each official TAD property PDF by account number.")
        tad_index = None
        if tad_upload:
            try:
                tad_index = load_optional_tad_index(tad_upload)
                st.success(f"TAD index loaded: {len(tad_index.working):,} usable address/account rows.")
            except Exception as exc:
                st.error(str(exc)); return
        if uploaded:
            valid, message = validate_upload(uploaded)
            if not valid: st.error(message); return
            raw = uploaded.getvalue()
            try: df = read_excel_safely(raw)
            except Exception as exc: st.error(f"Unable to read workbook: {exc}"); return
            missing = validate_columns(df)
            if missing: st.error("Missing required field(s) or recognized aliases: " + ", ".join(missing)); return
            st.session_state["input_fingerprint"] = file_fingerprint(raw)
            st.markdown("#### File preview"); st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            address_col = resolve_input_column(df, "Property Address")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(df):,}"); c2.metric("Addresses / legal", f"{df[address_col].notna().sum():,}"); c3.metric("TAD bulk index", "Loaded" if tad_index else "Not loaded"); c4.metric("Extra JSON API", "Yes" if ConfigurableJSONSource().enabled else "No")
            if PdfReader is None: st.warning("Install pypdf before running TAD PDF enrichment: pip install pypdf")
            if not tad_index: st.warning("Without a TAD bulk index or an account/APN column, TAD cannot be resolved automatically from address in this version. This avoids depending on an undocumented TAD search endpoint.")
            if st.button("Start research", type="primary", use_container_width=True):
                progress = st.progress(0); status = st.empty()
                def update_progress(current, total, address, research_status):
                    progress.progress(current / max(total, 1)); status.caption(f"{current:,} / {total:,} — {address[:80]} — {research_status}")
                enriched = research_dataframe(df, progress_callback=update_progress, tad_index=tad_index)
                st.session_state["results"] = enriched; progress.progress(1.0); status.success(f"Research completed for {len(enriched):,} rows."); st.rerun()
    with results_tab:
        results = st.session_state.get("results")
        if results is None: st.info("Upload a workbook and run research to populate results.")
        else: render_results(results)
    with help_tab: render_help()
    with privacy_tab: render_privacy()


def render_results(df: pd.DataFrame):
    st.subheader("Research results")
    total = len(df); review = df["Uncertain Match Flag"].eq("YES").sum(); missing = df["Missing Data Flag"].eq("YES").sum(); tad_verified = df["TAD Verified"].eq("YES").sum() if "TAD Verified" in df else 0
    c1, c2, c3, c4 = st.columns(4); c1.metric("Properties", f"{total:,}"); c2.metric("TAD verified", f"{tad_verified:,}"); c3.metric("Needs review", f"{review:,}"); c4.metric("Missing data", f"{missing:,}")
    st.divider(); f1, f2, f3 = st.columns([2,1,1]); search_text = f1.text_input("Search", placeholder="Address, owner, APN, subdivision..."); match_filter = f2.multiselect("Match status", options=sorted(df["Match Status"].dropna().astype(str).unique().tolist())); review_only = f3.checkbox("Needs review only")
    filtered = df.copy()
    if search_text:
        needle = search_text.lower(); searchable_columns = [c for c in ["Grantor/Owner","Grantor / Owner","Property Address","Property / Legal Description","Matched Address","Tax Account/APN","Subdivision","Current Owner"] if c in filtered.columns]; mask = pd.Series(False, index=filtered.index)
        for column in searchable_columns: mask |= filtered[column].fillna("").astype(str).str.lower().str.contains(re.escape(needle), regex=True)
        filtered = filtered[mask]
    if match_filter: filtered = filtered[filtered["Match Status"].isin(match_filter)]
    if review_only: filtered = filtered[filtered["Uncertain Match Flag"] == "YES"]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} properties.")
    priority_columns = ["Grantor/Owner","Grantor / Owner","Property Address","Property / Legal Description","Matched Address","Address Match Score","Match Status","TAD Account Number","Current Owner","Year Built","Total Assessed Value","Market Value","TAD Verified","Foreclosure Status","Lead Action","Later Sale Found","Likely Resolved","County Sale Date","County Purchaser","County Sale Amount","Research Status","Missing Data Flag","Uncertain Match Flag"]
    display_columns = [c for c in priority_columns if c in filtered.columns]; st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True, height=500)
    output = excel_bytes(df); st.download_button("Export enriched Excel", data=output, file_name=f"tarrant_county_property_research_{datetime.now():%Y%m%d_%H%M%S}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)


def render_help():
    st.subheader("Help")
    st.markdown("""
### Property workbook
Recognizes the original headers and aliases such as `Grantor / Owner`, `Sale Month`, and `Property / Legal Description`. Original input columns are not renamed or overwritten.

### TAD enrichment
For dependable bulk address matching, upload a public TAD data download/index containing a situs/address column and account-number column. The app matches the address to the account and then fetches the official TAD property-information PDF from tad.org.

If the property workbook already contains a TAD account/APN, the bulk index is not required for that row. Install PDF support with `pip install pypdf`.

### PubRecord
PubRecord is included as a manual review fallback only. Its public property page currently states that searches are limited to 5 per day, so this app does not automate bulk PubRecord searches or bypass that limit.

### Optional licensed JSON API
The existing `PROPERTY_API_URL_TEMPLATE`, `PROPERTY_API_KEY`, and `PROPERTY_API_KEY_HEADER` settings still work.
""")


def render_privacy():
    st.subheader("Privacy & source handling")
    st.markdown("- Uploaded workbooks are processed in the running Streamlit session.\n- The app does not send uploaded workbooks to PubRecord.\n- TAD requests are made only for resolved public account numbers.\n- Respect source terms, rate limits, and public-record restrictions.\n- Do not use this tool for FCRA-regulated consumer-reporting purposes.")


if __name__ == "__main__":
    show_dashboard()
