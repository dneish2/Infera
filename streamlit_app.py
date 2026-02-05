import os, re, streamlit as st
from dotenv import load_dotenv
from graphAgent import CompanyAnalyzer, MY_GOOGLE_API_KEY, MY_CSE_ID, MY_OPENAI_API_KEY

load_dotenv()
analyzer = CompanyAnalyzer()

def clear_input():
    st.session_state["company_input"] = ""

st.title("🔥 Infera Company Analyzer")
st.write("Enter one or more company names separated by commas to generate a competitive report.")

company_input = st.text_input("Companies", "UBER, LYFT", key="company_input")

col_analyze, col_clear = st.columns(2, gap="medium")
run_clicked   = col_analyze.button("Analyze", use_container_width=True)
col_clear.button("Clear", use_container_width=True, on_click=clear_input)

if run_clicked:
    companies = [c.strip() for c in company_input.split(',') if c.strip()]
    if not companies:
        st.warning("Please enter at least one company name.")
    else:
        with st.spinner("Running analysis..."):
            # Pull directly from OS to ensure they are NOT None
            current_keys = {
                "google_search": os.getenv("MY_GOOGLE_API_KEY"),
                "google_cse_id": os.getenv("MY_CSE_ID"),
                "openai": os.getenv("MY_OPENAI_API_KEY"),
            }
            # Call the analyzer with these fresh keys
            result = analyzer.analyze_companies(companies, current_keys)
        st.success("Analysis complete!")

        st.header("Rankings")
        st.markdown(result.get("ranked_companies", ""))

        for company, report in zip(companies, result.get("company_details", [])):
            st.markdown(report, unsafe_allow_html=True)
            safe_name = re.sub(r"\W+", "_", company)
            chart_path = os.path.join("charts", f"{safe_name}_radar.png")
            chart_html_path = os.path.join("charts_html", f"{safe_name}_radar.html")
            if os.path.exists(chart_path):
                st.image(chart_path)
                with open(chart_path, "rb") as f:
                    st.download_button(
                        label=f"Download {company} chart",
                        data=f,
                        file_name=os.path.basename(chart_path),
                    )
            if os.path.exists(chart_html_path):
                with open(chart_html_path, "r") as f:
                    st.components.v1.html(f.read(), height=600)
                with open(chart_html_path, "rb") as f:
                    st.download_button(
                        label=f"Download {company} interactive chart",
                        data=f,
                        file_name=os.path.basename(chart_html_path),
                    )

        report_path = os.path.join(os.path.dirname(__file__), "company_analysis_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_contents = f.read()
            st.download_button(
                label="Download Full Report",
                data=report_contents,
                file_name="company_analysis_report.md",
            )