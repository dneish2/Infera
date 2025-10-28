import csv
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from graphAgent import CompanyAnalyzer, MY_GOOGLE_API_KEY, MY_CSE_ID, MY_OPENAI_API_KEY

try:
    import pdfkit

    HAS_PDFKIT = True
except Exception:  # pragma: no cover - optional dependency
    pdfkit = None
    HAS_PDFKIT = False

try:
    from markdown_it import MarkdownIt

    md_renderer = MarkdownIt()
except Exception:  # pragma: no cover - optional dependency
    MarkdownIt = None
    md_renderer = None

load_dotenv()

analyzer = CompanyAnalyzer()

st.title("Infera Company Analyzer")

st.write(
    "Enter one or more company names separated by commas to generate a competitive report."
)


def save_email_lead(email: str, note: str | None = None) -> None:
    leads_path = Path("output") / "email_leads.csv"
    leads_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    rows = [timestamp, email, note or ""]
    file_exists = leads_path.exists()
    with leads_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "email", "note"])
        writer.writerow(rows)


with st.sidebar:
    st.header("Stay in the loop")
    st.caption("Share your email to receive upcoming feature updates and early access.")
    with st.form("lead_capture"):
        email_input = st.text_input("Email address")
        notes_input = st.text_area("What insights are you looking for?", height=80)
        subscribe = st.form_submit_button("Notify me")
        if subscribe:
            if not email_input or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email_input):
                st.warning("Please enter a valid email address.")
            else:
                save_email_lead(email_input.strip(), notes_input.strip())
                st.success("Thanks! We'll keep you posted with updates.")


company_input = st.text_input("Companies", "UBER, LYFT")

col1, col2 = st.columns(2)
run_clicked = col1.button("Analyze")
clear_clicked = col2.button("Clear")

if clear_clicked:
    st.session_state.clear()
    st.experimental_rerun()

if run_clicked:
    companies = [c.strip() for c in company_input.split(',') if c.strip()]
    if not companies:
        st.warning("Please enter at least one company name.")
    else:
        with st.spinner("Running analysis..."):
            api_keys = {
                "google_search": MY_GOOGLE_API_KEY,
                "google_cse_id": MY_CSE_ID,
                "openai": MY_OPENAI_API_KEY,
            }
            result = analyzer.analyze_companies(companies, api_keys)
        st.session_state["analysis_result"] = result
        st.session_state["companies"] = companies
        st.session_state.pop("pdf_data", None)
        st.success("Analysis complete!")


def render_report(result: dict, companies: list[str]) -> None:
    ranked_md = result.get("ranked_companies", "")
    detail_sections = result.get("company_details", [])
    combined_md = ranked_md + ("\n\n" if ranked_md and detail_sections else "") + "\n\n".join(
        detail_sections
    )
    st.session_state["combined_md"] = combined_md

    tab_report, tab_edit = st.tabs(["Report", "Edit & Export"])

    with tab_report:
        st.header("Rankings")
        st.markdown(ranked_md or "No rankings available.")

        for company, report in zip(companies, detail_sections):
            st.markdown(report, unsafe_allow_html=True)
            safe_name = re.sub(r"\W+", "_", company)
            chart_path = os.path.join("charts", f"{safe_name}_radar.png")
            if os.path.exists(chart_path):
                st.image(chart_path)
                with open(chart_path, "rb") as f:
                    st.download_button(
                        label=f"Download {company} chart",
                        data=f,
                        file_name=os.path.basename(chart_path),
                    )

        if combined_md.strip():
            st.download_button(
                label="Download Full Report",
                data=combined_md,
                file_name="company_report.md",
            )

    with tab_edit:
        edited_md = st.text_area(
            "Edit Markdown", st.session_state.get("combined_md", ""), height=400
        )
        st.session_state["combined_md"] = edited_md

        if md_renderer is not None:
            preview_html = md_renderer.render(edited_md)
            st.markdown("#### Live Preview")
            components.html(preview_html, height=300, scrolling=True)
        else:
            st.info(
                "Install `markdown-it-py` to enable the live preview. The Markdown below shows the raw content."
            )
            st.markdown(edited_md)

        col_left, col_right = st.columns(2)
        col_left.download_button(
            label="Download Edited Markdown",
            data=edited_md,
            file_name="company_report.md",
        )

        if HAS_PDFKIT:
            if col_right.button("Generate PDF"):
                html_input = (
                    md_renderer.render(edited_md) if md_renderer is not None else edited_md
                )
                try:
                    st.session_state["pdf_data"] = pdfkit.from_string(html_input, False)
                except Exception as e:  # pragma: no cover - runtime environment
                    st.error(f"PDF generation failed: {e}")
                    st.session_state.pop("pdf_data", None)
            if pdf_data := st.session_state.get("pdf_data"):
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name="company_report.pdf",
                )
        else:
            col_right.button("Generate PDF", disabled=True)
            st.warning(
                "PDF export requires `pdfkit` and the wkhtmltopdf binary. Install them to enable this feature."
            )


if "analysis_result" in st.session_state and "companies" in st.session_state:
    render_report(st.session_state["analysis_result"], st.session_state["companies"])


