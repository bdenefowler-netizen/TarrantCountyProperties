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

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Tarrant County Property Research"
MAX_UPLOAD_MB = 25
REQUEST_TIMEOUT = 20

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
            f"https://www.pubrecord.org/property-records/?q={encoded}",
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


def research_property(
    address: str,
    sources: list[PropertySource],
) -> dict[str, Any]:
    source_results = []

    for source in sources:
        try:
            source_results.append(source.search(address))
        except Exception as exc:
            source_results.append(
                SourceResult(
                    source=source.name,
                    query_address=address,
                    status="error",
                    note=f"{type(exc).__name__}: {exc}",
                )
            )

        # Polite rate pacing. Tune according to source rules.
        time.sleep(0.05)

    return merge_source_results(address, source_results)


def research_dataframe(
    input_df: pd.DataFrame,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Preserve all source DataFrame columns and values, appending research
    columns afterward.
    """
    original = input_df.copy(deep=True)

    # Remove old enrichment fields if user uploads a previously enriched file.
    base_columns = [
        c for c in original.columns
        if c not in RESEARCH_COLUMNS
    ]
    original = original[base_columns]

    source = ConfigurableJSONSource()
    sources: list[PropertySource] = [source]

    enrichment: list[dict[str, Any]] = []

    total = len(original)

    for position, (_, row) in enumerate(original.iterrows(), start=1):
        raw_address = row.get("Property Address", "")
        address = "" if pd.isna(raw_address) else str(raw_address)

        if not address.strip():
            researched = empty_research_row()
            researched["Research Status"] = "Missing address"
            researched["Missing Data Flag"] = "YES"
            researched["Uncertain Match Flag"] = "YES"
            researched["Research Timestamp"] = datetime.now().isoformat(
                timespec="seconds"
            )
        else:
            researched = research_property(address, sources)

        enrichment.append(researched)

        if progress_callback:
            progress_callback(
                position,
                total,
                address,
                researched["Research Status"],
            )

    enrichment_df = pd.DataFrame(
        enrichment,
        columns=RESEARCH_COLUMNS,
    )

    # Index reset guarantees one-to-one row alignment.
    return pd.concat(
        [
            original.reset_index(drop=True),
            enrichment_df.reset_index(drop=True),
        ],
        axis=1,
    )


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
    return [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]


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


def show_dashboard():
    st.title("Tarrant County Property Research")
    st.caption(
        "Bulk property-record research, source attribution, "
        "address matching, review flags, and Excel export."
    )

    upload_tab, results_tab, help_tab, privacy_tab = st.tabs(
        [
            "Research",
            "Results",
            "Help",
            "Privacy",
        ]
    )

    with upload_tab:
        st.subheader("Upload property list")

        st.info(
            "Required columns: Grantor/Owner, Sale Date, Filed Date, "
            "Property Address. Existing values are preserved; research "
            "fields are appended."
        )

        uploaded = st.file_uploader(
            "Excel workbook",
            type=["xlsx"],
            help=f"Maximum size: {MAX_UPLOAD_MB} MB.",
        )

        if uploaded:
            valid, message = validate_upload(uploaded)

            if not valid:
                st.error(message)
                return

            raw = uploaded.getvalue()

            try:
                df = read_excel_safely(raw)
            except Exception as exc:
                st.error(f"Unable to read workbook: {exc}")
                return

            missing = validate_columns(df)

            if missing:
                st.error(
                    "Missing required column(s): "
                    + ", ".join(missing)
                )
                return

            st.session_state["input_fingerprint"] = file_fingerprint(raw)

            st.markdown("#### File preview")
            st.dataframe(
                df.head(100),
                use_container_width=True,
                hide_index=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric(
                "Addresses",
                f"{df['Property Address'].notna().sum():,}",
            )
            c3.metric(
                "Configured API",
                "Yes"
                if ConfigurableJSONSource().enabled
                else "No",
            )

            if not ConfigurableJSONSource().enabled:
                st.warning(
                    "No automated property API is currently configured. "
                    "The app will still prepare review links and flags. "
                    "Configure PROPERTY_API_URL_TEMPLATE for a permitted "
                    "property-record data provider."
                )

            if st.button(
                "Start research",
                type="primary",
                use_container_width=True,
            ):
                progress = st.progress(0)
                status = st.empty()

                def update_progress(
                    current: int,
                    total: int,
                    address: str,
                    research_status: str,
                ):
                    fraction = current / max(total, 1)
                    progress.progress(fraction)

                    status.caption(
                        f"{current:,} / {total:,} — "
                        f"{address[:80]} — {research_status}"
                    )

                enriched = research_dataframe(
                    df,
                    progress_callback=update_progress,
                )

                st.session_state["results"] = enriched

                progress.progress(1.0)
                status.success(
                    f"Research completed for {len(enriched):,} rows."
                )

                st.rerun()

    with results_tab:
        results = st.session_state.get("results")

        if results is None:
            st.info(
                "Upload a workbook and run research to populate results."
            )
        else:
            render_results(results)

    with help_tab:
        render_help()

    with privacy_tab:
        render_privacy()


def render_results(df: pd.DataFrame):
    st.subheader("Research results")

    total = len(df)

    researched = (
        df["Research Status"]
        .eq("Researched")
        .sum()
    )

    review = (
        df["Uncertain Match Flag"]
        .eq("YES")
        .sum()
    )

    missing = (
        df["Missing Data Flag"]
        .eq("YES")
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties", f"{total:,}")
    c2.metric("Researched", f"{researched:,}")
    c3.metric("Needs review", f"{review:,}")
    c4.metric("Missing data", f"{missing:,}")

    st.divider()

    f1, f2, f3 = st.columns([2, 1, 1])

    search_text = f1.text_input(
        "Search",
        placeholder="Address, owner, APN, subdivision...",
    )

    match_filter = f2.multiselect(
        "Match status",
        options=sorted(
            df["Match Status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    review_only = f3.checkbox("Needs review only")

    filtered = df.copy()

    if search_text:
        needle = search_text.lower()

        searchable_columns = [
            c for c in [
                "Grantor/Owner",
                "Property Address",
                "Matched Address",
                "Tax Account/APN",
                "Subdivision",
                "Current Owner",
            ]
            if c in filtered.columns
        ]

        mask = pd.Series(
            False,
            index=filtered.index,
        )

        for column in searchable_columns:
            mask |= (
                filtered[column]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    re.escape(needle),
                    regex=True,
                )
            )

        filtered = filtered[mask]

    if match_filter:
        filtered = filtered[
            filtered["Match Status"].isin(match_filter)
        ]

    if review_only:
        filtered = filtered[
            filtered["Uncertain Match Flag"] == "YES"
        ]

    st.caption(
        f"Showing {len(filtered):,} of {len(df):,} properties."
    )

    priority_columns = [
        "Grantor/Owner",
        "Property Address",
        "Matched Address",
        "Address Match Score",
        "Match Status",
        "Tax Account/APN",
        "Current Owner",
        "Year Built",
        "Total Assessed Value",
        "Market Value",
        "Latest Sale Price",
        "Research Status",
        "Sources",
        "Missing Data Flag",
        "Uncertain Match Flag",
    ]

    display_columns = [
        c for c in priority_columns
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.markdown("### Property detail")

    if len(filtered):
        labels = []

        for idx, row in filtered.iterrows():
            labels.append(
                (
                    idx,
                    f"{row.get('Property Address', '')} "
                    f"— {row.get('Grantor/Owner', '')}"
                )
            )

        selected_label = st.selectbox(
            "Select property",
            options=[label for _, label in labels],
        )

        selected_idx = next(
            idx
            for idx, label in labels
            if label == selected_label
        )

        render_property_detail(
            filtered.loc[selected_idx]
        )

    output = excel_bytes(df)

    st.download_button(
        "Export enriched Excel",
        data=output,
        file_name=(
            "tarrant_county_property_research_"
            f"{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True,
    )


def render_property_detail(row: pd.Series):
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Property")

        fields = [
            "Property Address",
            "Matched Address",
            "Current Owner",
            "Tax Account/APN",
            "Subdivision",
            "Legal Description",
            "Lot Number",
            "Land Use Code",
            "Land Use Category",
            "Lot Area Sq Ft",
            "Lot Acres",
            "Census Tract",
        ]

        display_field_grid(row, fields)

        st.markdown("#### Structure")

        fields = [
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
        ]

        display_field_grid(row, fields)

    with right:
        st.markdown("#### Valuation & tax")

        fields = [
            "Current Tax Year",
            "Property Tax Amount",
            "Exemptions",
            "Land Value",
            "Improvement Value",
            "Total Assessed Value",
            "Market Value",
        ]

        display_field_grid(row, fields)

        st.markdown("#### Sale & financing")

        fields = [
            "Latest Document ID",
            "Latest Recording Date",
            "Latest Contract Date",
            "Latest Sale Price",
            "Latest Sale Type",
            "Buyer Information",
            "Lender",
            "Loan Amount",
            "Loan Type",
            "Loan Due Date",
        ]

        display_field_grid(row, fields)

        st.markdown("#### Research quality")

        fields = [
            "Address Match Score",
            "Match Status",
            "Missing Data Flag",
            "Uncertain Match Flag",
            "Research Status",
            "Sources",
            "Source URLs",
            "Research Notes",
        ]

        display_field_grid(row, fields)

    history_sections = {
        "Sales history": row.get("Sales History JSON", ""),
        "Mortgage / deed / loan history":
            row.get("Mortgage/Deed/Loan History JSON", ""),
        "Property tax history":
            row.get("Property Tax History JSON", ""),
        "Assessment history":
            row.get("Assessment History JSON", ""),
    }

    for title, raw_value in history_sections.items():
        if raw_value not in ("", None):
            with st.expander(title):
                try:
                    parsed = (
                        json.loads(raw_value)
                        if isinstance(raw_value, str)
                        else raw_value
                    )
                    st.json(parsed)
                except Exception:
                    st.write(raw_value)


def display_field_grid(
    row: pd.Series,
    fields: list[str],
):
    available = []

    for field_name in fields:
        value = row.get(field_name, "")

        if value is not None and not (
            isinstance(value, float) and pd.isna(value)
        ):
            if str(value).strip():
                available.append(
                    (field_name, str(value))
                )

    if not available:
        st.caption("No data available.")
        return

    for i in range(0, len(available), 2):
        cols = st.columns(2)

        for position, item in enumerate(
            available[i:i + 2]
        ):
            key, value = item

            cols[position].markdown(
                f"""
                <div class="research-card">
                    <div class="muted">{key}</div>
                    <div><strong>{value}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_help():
    st.subheader("Help")

    st.markdown(
        """
**Input format**

Upload an `.xlsx` workbook containing these exact column names:

- `Grantor/Owner`
- `Sale Date`
- `Filed Date`
- `Property Address`

Additional input columns are retained.

**Data sources**

The application is designed to combine appraisal-district, county,
tax, deed/mortgage, and other legally accessible property-record
sources through source adapters.

For Tarrant County, authoritative records should be verified against
the relevant county office or appraisal district before being relied
on for title, lending, acquisition, tax, or legal decisions.

**Connecting an API**

Set:

```text
PROPERTY_API_URL_TEMPLATE=https://provider.example/property?address={address}
PROPERTY_API_KEY=...
PROPERTY_API_KEY_HEADER=Authorization