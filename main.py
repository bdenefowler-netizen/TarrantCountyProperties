from __future__ import annotations

import hashlib
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


def run():
    st.title("Tarrant County Property Research")
    st.caption("v2.5 • Foreclosure cleanup • TAD match diagnostics • Split-TXT streaming • Excel export")
    research_tab, results_tab, help_tab, privacy_tab = st.tabs(["Research", "Results", "Help", "Privacy"])

    with research_tab:
        st.subheader("1. Upload foreclosure / distress leads")
        lead_file = st.file_uploader(
            "Property / foreclosure workbook",
            type=["xlsx"],
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
            help=f"Select all related split ZIPs at the same time. Each individual upload may be up to {MAX_TAD_UPLOAD_MB} MB.",
        )

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
