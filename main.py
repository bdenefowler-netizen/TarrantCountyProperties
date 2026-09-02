from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import streamlit as st

from lib.tad_bulk import BulkTADIndex, load_tad_dataframe, make_bulk_source_class, target_house_numbers

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


def run():
    st.title("Tarrant County Property Research")
    st.caption("v2.2 • Local TAD bulk matching • Excel/ZIP support • Foreclosure research • Excel export")
    research_tab, results_tab, help_tab, privacy_tab = st.tabs(["Research", "Results", "Help", "Privacy"])

    with research_tab:
        st.subheader("1. Upload foreclosure / distress leads")
        lead_file = st.file_uploader(
            "Property / foreclosure workbook",
            type=["xlsx"],
            help="Uses the original county columns or the cleaned workbook aliases.",
        )
        if not lead_file:
            st.info("Start with the foreclosure/distress workbook, then add TAD bulk property data.")
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

        address_col = core.resolve_input_column(df, "Property Address")
        addresses = df[address_col].fillna("").astype(str).tolist()
        st.dataframe(df.head(75), use_container_width=True, hide_index=True)

        st.subheader("2. Add TAD bulk property data")
        st.caption(
            "Upload TAD TXT, CSV, XLSX, or ZIP files. ZIP archives may contain TXT, CSV, or XLSX data. "
            "Large pipe-delimited property files are streamed in chunks and only candidate rows sharing house numbers "
            "with your leads are kept in memory."
        )
        tad_file = st.file_uploader(
            "TAD property data",
            type=["zip", "txt", "csv", "xlsx"],
            help=f"Supports TAD TXT, CSV, Excel, and ZIP archives containing those formats up to {MAX_TAD_UPLOAD_MB} MB.",
        )

        tad_index = None
        source_name = ""
        if tad_file:
            raw = tad_file.getvalue()
            key = _fingerprint(raw) + ":" + str(hash(tuple(sorted(target_house_numbers(addresses)))))
            cached = st.session_state.get("bulk_tad_cache")
            if cached and cached.get("key") == key:
                tad_index = cached["index"]
                source_name = cached["source"]
            else:
                try:
                    with st.spinner("Loading TAD data and building a lead-specific property index..."):
                        tad_df, source_name = load_tad_dataframe(tad_file, addresses, MAX_TAD_UPLOAD_MB)
                        tad_index = BulkTADIndex(tad_df)
                    st.session_state["bulk_tad_cache"] = {"key": key, "index": tad_index, "source": source_name}
                except Exception as exc:
                    st.error(f"Unable to load TAD bulk data: {exc}")
                    return
            st.success(f"TAD ready: {len(tad_index.working):,} candidate rows from {source_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lead rows", f"{len(df):,}")
        c2.metric("Street-number groups", f"{len(target_house_numbers(addresses)):,}")
        c3.metric("TAD candidates", f"{len(tad_index.working):,}" if tad_index else "0")
        c4.metric("TAD source", "Loaded" if tad_index else "Not loaded")

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
            st.session_state["tad_source_name"] = source_name
            progress.progress(1.0)
            status.success(f"Research complete for {len(enriched):,} properties.")
            st.rerun()

    with results_tab:
        results = st.session_state.get("results")
        if results is None:
            st.info("Run research first.")
        else:
            source = st.session_state.get("tad_source_name", "")
            if source:
                st.caption(f"TAD bulk source: {source}")
            core.render_results(results)

    with help_tab:
        st.subheader("Bulk TAD workflow")
        st.markdown(
            """
1. Upload the foreclosure/distress `.xlsx` file.
2. Upload TAD property data as `.txt`, `.csv`, `.xlsx`, or `.zip`.
3. ZIP files may contain a TAD TXT/CSV property export or an Excel workbook such as Residential Comp Attribute data.
4. Start research.
5. Review fuzzy/uncertain matches before relying on them.
6. Export the enriched Excel workbook.

The importer recognizes TAD account identifiers such as `Account_Num`, `Account Number`, `APN`, and `PIN`, plus common situs-address column variations.

The bulk TAD file is the primary property source. PubRecord remains a manual fallback. The app does not bulk-scrape PubRecord.
"""
        )
        core.render_help()

    with privacy_tab:
        core.render_privacy()


if __name__ == "__main__":
    run()
