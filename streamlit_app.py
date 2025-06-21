import os
import re
import streamlit as st
from dotenv import load_dotenv
from graphAgent import CompanyAnalyzer, MY_GOOGLE_API_KEY, MY_CSE_ID, MY_OPENAI_API_KEY

load_dotenv()

analyzer = CompanyAnalyzer()

st.title("Infera Company Analyzer")

st.write(
    "Enter one or more company names separated by commas to generate a competitive report."
)

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
        st.success("Analysis complete!")

        # Rankings summary
        st.header("Rankings")
        st.markdown(result.get("ranked_companies", ""))

        # Detailed company sections
        for company, report in zip(companies, result.get("company_details", [])):
            st.markdown(report, unsafe_allow_html=True)
            # Attempt to display radar chart image
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

        # Combined markdown for download
        combined_md = result.get("ranked_companies", "") + "\n\n" + "\n\n".join(
            result.get("company_details", [])
        )
        st.download_button(
            label="Download Full Report",
            data=combined_md,
            file_name="company_report.md",
        )

