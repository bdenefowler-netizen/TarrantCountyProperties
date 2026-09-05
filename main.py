from __future__ import annotations

import hashlib
import io
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.tad_bulk import BulkTADIndex, make_bulk_source_class, target_house_numbers
from lib.tad_multipart import load_tad_uploads

CORE_PATH = Path(__file__).parent / "artifacts" / "tarrant-property-research" / "app.py"
spec = importlib.util.spec_from_file_location("tarrant_core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

core._ORIGINAL_TAD_PROPERTY_SOURCE = core.TADPropertySource
core.TADPropertySource = make_bulk_source_class(core)

MAX_TAD_UPLOAD_MB = 200


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _combined_upload_key(files, addresses: list[str]) -> str:
    file_parts = []
    for uploaded in files:
        raw = uploaded.getvalue()
        file_parts.append(f"{uploaded.name}:{_fingerprint(raw)}")
    houses = ",".join(sorted(target_house_numbers(addresses)))
    return "|".join(file_parts) + ":houses:" + hashlib.sha256(houses.encode("utf-8")).hexdigest()


def _clean_foreclosure_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove report chrome while preserving actual foreclosure/legal-description rows."""
    working = df.copy()
    address_col = core.resolve_input_column(working, "Property Address")
    addresses = working[address_col].fillna("").astype(str).str.strip()

    # Real records in the county export always carry something in the Property
    # Address field. Legal descriptions are also stored there, so they survive.
    keep = addresses.ne("") & addresses.str.upper().ne("PROPERTY ADDRESS")

    try:
        grantor_col = core.resolve_input_column(working, "Grantor")
    except Exception:
        grantor_col = None

    if grantor_col:
        grantors = working[grantor_col].fillna("").astype(str).str.strip()
        keep &= ~grantors.str.startswith("©")
        keep &= grantors.str.upper().ne("GRANTOR")

    cleaned = working.loc[keep].reset_index(drop=True)
    removed = len(working) - len(cleaned)
    return cleaned, removed


def _tad_quick_check(tad_index: BulkTADIndex, addresses: list[str]):
    matches = []
    for address in addresses:
        account, matched_address, score = tad_index.lookup(address)
        if account:
            matches.append((address, account, matched_address, score))
    return matches


def _canonical_tad_account(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return ""
    # County reports sometimes omit leading zeroes and sometimes append a second
    # account in parentheses. The first account is the primary one for matching.
    if len(digits) > 8:
        digits = digits[:8]
    return digits.zfill(8)


def _norm_header(value) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: set[str]):
    normalized = {_norm_header(c): c for c in df.columns}
    for candidate in candidates:
        key = _norm_header(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _canonical_cause(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return "".join(ch for ch in str(value).upper().strip() if ch.isalnum())


def _read_county_report(raw: bytes) -> pd.DataFrame:
    # County exports usually place headers on row 4, but detect them so
    # alternate report layouts do not silently fail.
    preview = pd.read_excel(io.BytesIO(raw), header=None, nrows=12, dtype=object, engine="openpyxl")
    header_row = None
    header_tokens = {
        "sale status", "appraisal dist #", "tad account", "tad #",
        "cause no", "cause number", "receipt #", "bidder id",
    }
    wanted = {_norm_header(x) for x in header_tokens}
    for idx, row in preview.iterrows():
        row_tokens = {_norm_header(v) for v in row.tolist() if v not in (None, "")}
        if row_tokens & wanted:
            header_row = idx
            break
    if header_row is None:
        header_row = 3
    return pd.read_excel(io.BytesIO(raw), header=header_row, dtype=object, engine="openpyxl")


def _load_county_sale_events(files) -> tuple[dict[str, dict], dict[str, dict], list[str]]:
    events_by_account: dict[str, dict] = {}
    events_by_cause: dict[str, dict] = {}
    errors: list[str] = []

    status_candidates = {"sale status", "status"}
    account_candidates = {
        "appraisal dist #", "appraisal dist#", "appraisal district #",
        "tad account", "tad account #", "tad #", "tad#", "tad account number",
        "account num", "account number", "apn", "pin",
    }
    cause_candidates = {"cause no", "cause no.", "cause #", "cause number", "cause"}
    date_candidates = {"sale date"}
    purchaser_candidates = {"purchaser", "buyer"}
    amount_candidates = {"amount", "sale amount"}

    for uploaded in files or []:
        try:
            raw = uploaded.getvalue()
            df = _read_county_report(raw).dropna(how="all")
            account_col = _find_column(df, account_candidates)
            cause_col = _find_column(df, cause_candidates)
            # Aggregate-only reports (for example payment summaries) have no
            # property key and are intentionally skipped.
            if not account_col and not cause_col:
                continue

            status_col = _find_column(df, status_candidates)
            date_col = _find_column(df, date_candidates)
            purchaser_col = _find_column(df, purchaser_candidates)
            amount_col = _find_column(df, amount_candidates)

            inferred_status = ""
            upper_name = uploaded.name.upper()
            if "STRUCK OFF" in upper_name:
                inferred_status = "Struck Off"
            elif "WITHDRAWN" in upper_name:
                inferred_status = "Withdrawn"
            elif "SOLD" in upper_name or "EXCESS PROCEEDS" in upper_name or "EXCESS FUNDS" in upper_name:
                inferred_status = "Sold"

            for _, row in df.iterrows():
                account = _canonical_tad_account(row.get(account_col)) if account_col else ""
                cause = _canonical_cause(row.get(cause_col)) if cause_col else ""
                if not account and not cause:
                    continue
                status = str(row.get(status_col, "") or "").strip() if status_col else inferred_status
                if not status:
                    status = inferred_status
                if not status:
                    continue

                event = {
                    "status": status,
                    "sale_date": row.get(date_col, "") if date_col else "",
                    "cause": str(row.get(cause_col, "") or "").strip() if cause_col else "",
                    "purchaser": str(row.get(purchaser_col, "") or "").strip() if purchaser_col else "",
                    "amount": row.get(amount_col, "") if amount_col else "",
                    "source": uploaded.name,
                }

                def keep_latest(store: dict[str, dict], key: str):
                    if not key:
                        return
                    current = store.get(key)
                    if current is None:
                        store[key] = event
                        return
                    cur_date = pd.to_datetime(current.get("sale_date"), errors="coerce")
                    new_date = pd.to_datetime(event.get("sale_date"), errors="coerce")
                    if pd.isna(cur_date) or (not pd.isna(new_date) and new_date >= cur_date):
                        store[key] = event

                keep_latest(events_by_account, account)
                keep_latest(events_by_cause, cause)
        except Exception as exc:
            errors.append(f"{uploaded.name}: {exc}")

    return events_by_account, events_by_cause, errors


def _apply_county_sale_status(
    df: pd.DataFrame,
    events_by_account: dict[str, dict],
    events_by_cause: dict[str, dict],
) -> pd.DataFrame:
    if not events_by_account and not events_by_cause:
        return df

    result = df.copy()
    for column in [
        "Foreclosure Status", "Later Sale Found", "Likely Resolved", "Lead Action",
        "County Sale Date", "County Cause Number", "County Purchaser",
        "County Sale Amount", "County Status Source",
    ]:
        if column not in result.columns:
            result[column] = ""

    account_columns = [c for c in ["TAD Account Number", "Tax Account/APN", "Account_Num", "APN", "PIN"] if c in result.columns]
    cause_columns = [c for c in result.columns if _norm_header(c) in {
        _norm_header("Cause No"), _norm_header("Cause Number"), _norm_header("Cause #"), _norm_header("Cause")
    }]

    for idx, row in result.iterrows():
        event = None
        for col in account_columns:
            account = _canonical_tad_account(row.get(col))
            if account and account in events_by_account:
                event = events_by_account[account]
                break

        if event is None:
            for col in cause_columns:
                cause = _canonical_cause(row.get(col))
                if cause and cause in events_by_cause:
                    event = events_by_cause[cause]
                    break

        if event is None:
            continue
        status = str(event.get("status", "")).strip()
        status_upper = status.upper()

        result.at[idx, "Foreclosure Status"] = status
        result.at[idx, "County Cause Number"] = event.get("cause", "")
        result.at[idx, "County Purchaser"] = event.get("purchaser", "")
        result.at[idx, "County Sale Amount"] = event.get("amount", "")
        result.at[idx, "County Status Source"] = event.get("source", "")

        sale_date = event.get("sale_date", "")
        parsed_date = pd.to_datetime(sale_date, errors="coerce")
        result.at[idx, "County Sale Date"] = parsed_date.date().isoformat() if not pd.isna(parsed_date) else (sale_date or "")

        if "SOLD" in status_upper:
            result.at[idx, "Later Sale Found"] = "YES"
            result.at[idx, "Likely Resolved"] = "YES"
            result.at[idx, "Lead Action"] = "REMOVE — SOLD"
        elif "STRUCK OFF" in status_upper:
            result.at[idx, "Later Sale Found"] = "NO"
            result.at[idx, "Likely Resolved"] = "YES"
            result.at[idx, "Lead Action"] = "STRUCK OFF — REVIEW"
        elif "WITHDRAWN" in status_upper:
            result.at[idx, "Later Sale Found"] = "NO"
            result.at[idx, "Likely Resolved"] = "NO"
            result.at[idx, "Lead Action"] = "FOLLOW UP / MONITOR"
        else:
            result.at[idx, "Lead Action"] = "REVIEW"

    return result


def run():
    st.title("Tarrant County Property Research")
    st.caption("v2.5 • Foreclosure cleanup • TAD match diagnostics • Split-TXT streaming • Excel export")
    research_tab, results_tab, help_tab, privacy_tab = st.tabs(["Research", "Results", "Help", "Privacy"])

    with research_tab:
        st.subheader("1. Upload foreclosure / distress leads")
        lead_file = st.file_uploader(
            "Property / foreclosure workbook",
            type=["xlsx"],
            max_upload_size=MAX_TAD_UPLOAD_MB,
            help="Uses the original county columns or the cleaned workbook aliases.",
        )
        if not lead_file:
            st.info("Start with the foreclosure/distress workbook, then add one or more TAD data files.")
            return

        valid, message = core.validate_upload(lead_file)
        if not valid:
            st.error(message)
            return
        try:
            lead_raw = lead_file.getvalue()
            df = core.read_excel_safely(lead_raw)
        except Exception as exc:
            st.error(f"Unable to read workbook: {exc}")
            return

        missing = core.validate_columns(df)
        if missing:
            st.error("Missing required field(s): " + ", ".join(missing))
            return

        df, removed_rows = _clean_foreclosure_rows(df)
        if removed_rows:
            st.info(
                f"Cleaned county report: removed {removed_rows:,} page-header/footer/blank row(s); "
                f"{len(df):,} foreclosure lead row(s) remain."
            )

        address_col = core.resolve_input_column(df, "Property Address")
        addresses = df[address_col].fillna("").astype(str).tolist()
        st.dataframe(df.head(75), use_container_width=True, hide_index=True)

        st.subheader("2. Add TAD property data")
        st.caption(
            "Upload multiple TAD TXT, CSV, XLSX, or ZIP files together. Split files named like "
            "PropertyData(Delimited).txt.part1, .part2, etc. may be spread across multiple ZIP uploads; "
            "the app stitches them together in numeric order while streaming and preserves rows split across part boundaries."
        )
        tad_files = st.file_uploader(
            "TAD property data files",
            type=["zip", "txt", "csv", "xlsx"],
            accept_multiple_files=True,
            max_upload_size=MAX_TAD_UPLOAD_MB,
            help=f"Select all related split ZIPs at the same time. Each individual upload may be up to {MAX_TAD_UPLOAD_MB} MB.",
        )

        st.subheader("3. Add county tax-sale results (optional)")
        county_files = st.file_uploader(
            "County tax-sale result files",
            type=["xlsx"],
            accept_multiple_files=True,
            max_upload_size=MAX_TAD_UPLOAD_MB,
            help="Upload Sold, Withdrawn, Struck Off, All Without Exceptions, Sale Receipt, or Excess Proceeds reports. Status is matched by TAD account number.",
        )
        county_events, county_cause_events, county_errors = _load_county_sale_events(county_files)
        if county_errors:
            st.warning("Some county files could not be read:\n\n" + "\n".join(f"• {e}" for e in county_errors))
        if county_events or county_cause_events:
            unique_events = list(county_events.values())
            status_counts = pd.Series([e["status"] for e in unique_events]).value_counts() if unique_events else pd.Series(dtype=object)
            st.success(
                f"County sale status ready: {len(county_events):,} TAD account(s) "
                f"and {len(county_cause_events):,} cause number(s) indexed."
            )
            st.caption(" • ".join(f"{status}: {count:,}" for status, count in status_counts.items()))

        tad_index = None
        source_names: list[str] = []
        tad_matches = []
        if tad_files:
            key = _combined_upload_key(tad_files, addresses)
            cached = st.session_state.get("bulk_tad_cache")
            if cached and cached.get("key") == key:
                tad_index = cached["index"]
                source_names = cached.get("sources", [])
            else:
                status = st.empty()
                progress = st.progress(0)
                status.caption(f"Inspecting {len(tad_files)} TAD upload(s) and assembling split parts...")
                progress.progress(0.15)

                frames, source_names, errors = load_tad_uploads(tad_files, addresses, MAX_TAD_UPLOAD_MB)
                progress.progress(0.8)

                if errors:
                    st.warning("Some TAD data could not be loaded:\n\n" + "\n".join(f"• {message}" for message in errors))

                if not frames:
                    st.error("None of the selected TAD files produced usable property rows.")
                    return

                with st.spinner("Combining all TAD sources into one property index..."):
                    combined = pd.concat(frames, ignore_index=True, sort=False)
                    account_candidates = [c for c in combined.columns if str(c).strip().lower().replace("_", " ") in {
                        "account num", "account number", "account", "apn", "pin", "parcel id", "property id"
                    }]
                    if account_candidates:
                        account_col = account_candidates[0]
                        combined = combined.drop_duplicates(subset=[account_col], keep="first")
                    else:
                        combined = combined.drop_duplicates(keep="first")
                    tad_index = BulkTADIndex(combined)

                st.session_state["bulk_tad_cache"] = {
                    "key": key,
                    "index": tad_index,
                    "sources": source_names,
                }
                progress.progress(1.0)
                progress.empty()
                status.empty()

            tad_matches = _tad_quick_check(tad_index, addresses)
            st.success(
                f"Combined TAD index ready: {len(tad_index.working):,} candidate property rows from {len(source_names):,} source group(s)."
            )

            if tad_matches:
                st.success(
                    f"TAD quick-check: {len(tad_matches):,} of {len(addresses):,} lead row(s) currently resolve to a TAD account before research."
                )
                with st.expander("Preview TAD matches"):
                    preview = pd.DataFrame(
                        tad_matches[:25],
                        columns=["Lead Address", "TAD Account", "Matched Situs Address", "Score"],
                    )
                    st.dataframe(preview, use_container_width=True, hide_index=True)
            else:
                st.error(
                    "TAD quick-check found ZERO lead-address matches. Do not start research yet; "
                    "the uploaded property data/index still needs attention."
                )

            with st.expander("Loaded TAD sources"):
                for source in source_names:
                    st.write(f"• {source}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lead rows", f"{len(df):,}")
        c2.metric("Street-number groups", f"{len(target_house_numbers(addresses)):,}")
        c3.metric("TAD candidates", f"{len(tad_index.working):,}" if tad_index else "0")
        c4.metric("TAD matched leads", f"{len(tad_matches):,}" if tad_index else "0")

        if not tad_index:
            st.warning("Research can run without TAD bulk data, but automatic owner/property enrichment will be limited.")

        if st.button("Start research", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()

            def update_progress(current, total, address, research_status):
                progress.progress(current / max(total, 1))
                status.caption(f"{current:,} / {total:,} — {address[:75]} — {research_status}")

            enriched = core.research_dataframe(df, progress_callback=update_progress, tad_index=tad_index)
            enriched = _apply_county_sale_status(enriched, county_events, county_cause_events)
            st.session_state["results"] = enriched
            st.session_state["tad_source_names"] = source_names
            progress.progress(1.0)
            status.success(f"Research complete for {len(enriched):,} properties.")
            st.rerun()

    with results_tab:
        results = st.session_state.get("results")
        if results is None:
            st.info("Run research first.")
        else:
            sources = st.session_state.get("tad_source_names", [])
            if sources:
                st.caption(f"TAD sources combined: {len(sources)}")
            core.render_results(results)

    with help_tab:
        st.subheader("Bulk TAD workflow")
        st.markdown(
            """
1. Upload the foreclosure/distress `.xlsx` file. County report headers, footers, blank rows, and repeated column headings are removed automatically.
2. Select **all** TAD files you want to use at the same time: `.txt`, `.csv`, `.xlsx`, and/or `.zip`.
3. For split county files, upload **every ZIP containing the related `.txt.partN` pieces together**.
4. The app sorts split parts numerically, stitches rows across part boundaries, filters to candidate lead addresses, and builds one TAD index.
5. Check the **TAD quick-check** before starting research. If it says zero matches, stop there rather than producing an empty export.
6. Regular TAD ZIP/TXT/CSV/XLSX files are still supported and can be combined with the split dataset.
7. Start research, review uncertain matches, then export the enriched workbook.

Adding multiple files does **not** intentionally overwrite the prior source. All files selected in the uploader are combined for that research run.

The importer recognizes TAD account identifiers such as `Account_Num`, `Account Number`, `APN`, and `PIN`, plus common situs-address column variations.

The bulk TAD data is the primary property source. PubRecord remains a manual fallback. The app does not bulk-scrape PubRecord.
"""
        )
        core.render_help()

    with privacy_tab:
        core.render_privacy()


if __name__ == "__main__":
    run()
