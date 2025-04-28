import re
import os
import logging
import yfinance as yf
import wikipedia
import requests
from bs4 import BeautifulSoup
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from googleapiclient.discovery import build
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import datetime

# For interactive charts
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
MY_GOOGLE_API_KEY = os.getenv('MY_GOOGLE_API_KEY')
MY_CSE_ID = os.getenv('MY_CSE_ID')
MY_OPENAI_API_KEY = os.getenv('MY_OPENAI_API_KEY')
openai_client = OpenAI(api_key=MY_OPENAI_API_KEY)

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Schemas ---
class CompanyState(TypedDict):
    user_input: Annotated[list[str], START]
    company_type: list[str]
    company_data: list[dict]
    detailed_reports: list[str]
    summaries: list[str]
    ranked_companies: str

class ConfigSchema(TypedDict):
    api_keys: dict

# --- Helper functions for executive bio extraction ---
PRIORITY_KEYWORDS = ["CEO", "president", "chief executive", "co-founder", "CTO", "chair", "CFO", "president and ceo", "chairman", "chairman and CEO"]

def extract_executive_candidates_from_infobox(soup):
    """
    Given a BeautifulSoup object for a Wikipedia page, try to extract the list
    of executive candidates from the infobox “key people” field.
    Returns a list of dictionaries with keys: 'name' and 'role'.
    """
    candidates = []
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if not infobox:
        return candidates

    rows = infobox.find_all("tr")
    for row in rows:
        header = row.find("th")
        if header and "key people" in header.get_text(strip=True).lower():
            cell = row.find("td")
            if cell:
                # Replace <br> tags with newline characters
                for br in cell.find_all("br"):
                    br.replace_with("\n")
                text = cell.get_text(separator="\n", strip=True)
                # Remove citation markers like [1]
                text = re.sub(r'\[\d+\]', '', text)
                # Split by newlines (or semicolons) to get individual lines
                lines = re.split(r'\n|;', text)
                pattern = r'^(.+?)(?:\s*\((.+?)\))?$'
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1).strip()
                        role = match.group(2).strip() if match.group(2) else ""
                        if name and len(name.split()) > 1:
                            candidates.append({"name": name, "role": role})
            break  # Process only the first "key people" row.
    return candidates

def filter_priority_candidates(candidates):
    """
    Filter candidates to those whose role or name contains one of the priority keywords.
    If none match, return the original list.
    """
    priority = []
    for cand in candidates:
        role_text = cand.get("role", "").lower()
        name_text = cand.get("name", "").lower()
        if any(kw.lower() in role_text for kw in PRIORITY_KEYWORDS) or any(kw.lower() in name_text for kw in PRIORITY_KEYWORDS):
            priority.append(cand)
    return priority if priority else candidates

def fallback_extraction(soup):
    """
    As a fallback, scan the first non-empty paragraph and extract capitalized sequences as candidate names.
    """
    fallback = []
    paragraphs = soup.find_all("p")
    summary_text = ""
    for p in paragraphs:
        summary_text = p.get_text(strip=True)
        if summary_text:
            break
    # A simple regex to capture names (two or more consecutive capitalized words)
    names = re.findall(r'\b([A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+)+)\b', summary_text)
    for name in names:
        fallback.append({"name": name, "role": ""})
    return fallback

# --- Main Analyzer Class ---
class CompanyAnalyzer:
    def __init__(self):
        self.graph = StateGraph(CompanyState, config_schema=ConfigSchema)
        self._build_graph()
        self.ticker_cache = {}
        # Directory to store radar chart PNGs
        self.chart_dir = os.path.join(os.path.dirname(__file__), "charts")
        os.makedirs(self.chart_dir, exist_ok=True)
        # Directory to store radar chart HTMLs
        self.chart_html_dir = os.path.join(os.path.dirname(__file__), "charts_html")
        os.makedirs(self.chart_html_dir, exist_ok=True)

    def _build_graph(self):
        self.graph.add_node("user_input_node", self.user_input_node)
        self.graph.set_entry_point("user_input_node")
        self.graph.add_node("financial_data_node", self.financial_data_node)
        self.graph.add_edge("user_input_node", "financial_data_node")
        self.graph.add_node("company_background_node", self.company_background_node)
        self.graph.add_edge("financial_data_node", "company_background_node")
        self.graph.add_node("leadership_products_node", self.leadership_products_node)
        self.graph.add_edge("company_background_node", "leadership_products_node")
        self.graph.add_node("detailed_report_generation_node", self.detailed_report_generation_node)
        self.graph.add_edge("leadership_products_node", "detailed_report_generation_node")
        self.graph.add_node("summarization_node", self.summarization_node)
        self.graph.add_edge("detailed_report_generation_node", "summarization_node")
        self.graph.add_node("ranking_node", self.ranking_node)
        self.graph.add_edge("summarization_node", "ranking_node")
        self.graph.add_node("output_node", self.output_node)
        self.graph.add_edge("ranking_node", "output_node")
        self.graph.add_edge("output_node", END)
        self.compiled_graph = self.graph.compile()

    # --- Node Functions ---
    def user_input_node(self, state, config):
        user_input = state['user_input']
        company_types = [self.detect_company_type(company) for company in user_input]
        return {"company_type": company_types}

    def detect_company_type(self, company_name):
        ticker = self.verify_and_get_ticker(company_name)
        return "public" if ticker else "private"

    def verify_and_get_ticker(self, company_name):
        if company_name in self.ticker_cache:
            return self.ticker_cache[company_name]
        ticker = self.get_ticker(company_name)
        if ticker:
            self.ticker_cache[company_name] = ticker
            return ticker
        search_result = self.google_search(f"{company_name} stock ticker", MY_GOOGLE_API_KEY, MY_CSE_ID, num=1)
        for result in search_result:
            if "finance.yahoo.com" in result.get('link', ''):
                match = re.search(r'/quote/([^/?]+)', result['link'])
                if match:
                    ticker = match.group(1).upper()
                    self.ticker_cache[company_name] = ticker
                    return ticker
        return None

    def get_ticker(self, company_name):
        ticker_obj = yf.Ticker(company_name.upper())
        try:
            info = ticker_obj.info
            if 'shortName' in info:
                return company_name.upper()
        except Exception as e:
            logger.error(f"Error fetching ticker for {company_name}: {e}")
        return None

    def financial_data_node(self, state, config):
        company_types = state['company_type']
        company_data_list = []
        for company, ctype in zip(state['user_input'], company_types):
            company_data = {"company_name": company, "company_type": ctype}
            if ctype == "public":
                ticker = self.verify_and_get_ticker(company)
                if ticker:
                    company_data['ticker'] = ticker
                    try:
                        data = yf.Ticker(ticker).info
                        proper_name = data.get("longName") or data.get("shortName") or company
                        company_data["company_name"] = proper_name
                        financial_data = {
                            "Full Time Employees": data.get("fullTimeEmployees", "N/A"),
                            "Market Cap": data.get("marketCap", "N/A"),
                            "Total Revenue": data.get("totalRevenue", "N/A"),
                            "Quarterly Revenue": data.get("revenueQuarterly", "N/A"),
                            "Profit Margin": data.get("profitMargins", "N/A"),
                            "Free Cash Flow": data.get("freeCashflow", "N/A"),
                            "Day Low": data.get("dayLow", "N/A"),
                            "Day High": data.get("dayHigh", "N/A"),
                            "Revenue Growth": data.get("revenueGrowth", "N/A"),
                            "Total Cash": data.get("totalCash", "N/A"),
                            "Total Debt": data.get("totalDebt", "N/A")
                        }
                        if financial_data["Quarterly Revenue"] in [None, "N/A"]:
                            ticker_obj = yf.Ticker(ticker)
                            qf = ticker_obj.quarterly_financials
                            if not qf.empty and "Total Revenue" in qf.index:
                                financial_data["Quarterly Revenue"] = qf.loc["Total Revenue"].iloc[0]
                        company_data["financial_data"] = financial_data
                    except Exception as e:
                        logger.error(f"Error retrieving financial data for {ticker}: {e}")
                        company_data["financial_data"] = {}
                else:
                    company_data["financial_data"] = {}
            else:
                company_data["financial_data"] = {}
            company_data_list.append(company_data)
        return {"company_data": company_data_list}

    def company_background_node(self, state, config):
        for company_data in state['company_data']:
            company_name = company_data['company_name']
            search_result = self.google_search(f"{company_name} company overview", config["api_keys"]["google_search"], config["api_keys"]["google_cse_id"], num=1)
            company_data['google_overview'] = search_result[0].get('snippet', '') if search_result else ""
        return {"company_data": state['company_data']}

    def leadership_products_node(self, state, config):
        for company_data in state['company_data']:
            company_name = company_data['company_name']
            ticker = company_data.get('ticker', '')
            company_data['wikipedia_summary'] = self.get_wikipedia_summary(company_name, ticker)
            company_data['executive_bios'] = self.get_executive_bios(company_name)
        return {"company_data": state['company_data']}

    def detailed_report_generation_node(self, state, config):
        detailed_reports = []
        for company_data in state['company_data']:
            # Generate radar charts for financial health (PNG + HTML)
            chart_png_path, chart_html_path = self.generate_radar_chart(company_data)
            # Build the detailed report text
            detailed_report = self.generate_detailed_report(company_data, chart_png_path, chart_html_path)
            detailed_reports.append(detailed_report)
        return {"detailed_reports": detailed_reports}

    def summarization_node(self, state):
        summarizer = SummarizerAgent()
        summaries = [summarizer.summarize(report) for report in state['detailed_reports']]
        return {"summaries": summaries}

    def ranking_node(self, state):
        ranker = RankerAgent()
        ranked_companies = ranker.rank(state['summaries'])
        return {"ranked_companies": ranked_companies}

    def output_node(self, state):
        self.generate_markdown_report(state['ranked_companies'], state['detailed_reports'])
        return {"ranked_companies": state['ranked_companies'], "company_details": state['detailed_reports']}

    def analyze_companies(self, user_input, api_keys):
        state = {"user_input": user_input}
        config = {"api_keys": api_keys}
        return self.compiled_graph.invoke(state, config)

    # --- New: Radar Chart Generation Method ---
    def generate_radar_chart(self, company_data):
        """
        Generate an interactive radar chart (HTML) and a static PNG for the company's financial health.
        Uses key metrics: Profit Margin, Revenue Growth, Free Cash Flow % of Revenue, and Liquidity Ratio.
        Returns (png_path, html_path).
        """
        financial = company_data.get("financial_data", {})

        # Attempt to extract a date from YFinance data (if available)
        data_date = financial.get("regularMarketTime")
        if data_date:
            # Convert epoch to date string if possible
            try:
                data_date = datetime.datetime.fromtimestamp(data_date).strftime('%b %d, %Y')
            except Exception:
                data_date = "latest available"
        else:
            data_date = "latest available"

        # Extract raw numeric values
        def safe_float(val, default=0):
            try:
                return float(val or 0)
            except:
                return default

        pm = safe_float(financial.get("Profit Margin", 0))
        rg = safe_float(financial.get("Revenue Growth", 0))
        tr = safe_float(financial.get("Total Revenue", 0))
        fcf = safe_float(financial.get("Free Cash Flow", 0))
        tc = safe_float(financial.get("Total Cash", 0))
        td = safe_float(financial.get("Total Debt", 0))

        fcf_ratio = fcf / tr if tr > 0 else 0
        liquidity_ratio = tc / td if td > 0 else 1

        # Normalize metrics (clip at 1)
        pm_norm = min(pm, 1)
        rg_norm = min(rg, 1)
        fcf_norm = min(fcf_ratio, 1)
        liq_norm = min(liquidity_ratio, 1)

        metrics = ['Profit Margin', 'Revenue Growth', 'FCF % of Revenue', 'Liquidity Ratio']
        values = [pm_norm, rg_norm, fcf_norm, liq_norm]
        # Extended for closing the radar
        metrics_ext = metrics + [metrics[0]]
        values_ext = values + [values[0]]

        # Hover texts with actual values
        hover_texts = [
            f"Profit Margin: {pm*100:.1f}%",
            f"Revenue Growth: {rg*100:.1f}%",
            f"FCF % of Revenue: {fcf_ratio*100:.1f}%",
            f"Liquidity Ratio: {liquidity_ratio:.2f}"
        ]
        hover_ext = hover_texts + [hover_texts[0]]

        # Create figure
        fig = go.Figure(
            data=go.Scatterpolar(
                r=values_ext,
                theta=metrics_ext,
                fill='toself',
                name=company_data.get('company_name', 'Unknown'),
                hovertext=hover_ext,
                hoverinfo="text"
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=["0%", "25%", "50%", "75%", "100%"]
                )
            ),
            showlegend=True,
            title=(
                f"{company_data.get('company_name', 'Unknown')} Financial Health Radar Chart<br>"
                f"<sup>Data sourced from YFinance (Data as of {data_date})</sup>"
            )
        )

        # Save PNG
        safe_name = re.sub(r'\W+', '_', company_data.get('company_name', 'unknown'))
        png_path = os.path.join(self.chart_dir, f"{safe_name}_radar.png")
        try:
            fig.write_image(png_path)
            logger.info(f"Radar chart PNG saved for {company_data.get('company_name')} at {png_path}")
        except Exception as e:
            logger.error(f"Error saving radar chart PNG for {company_data.get('company_name')}: {e}")
            png_path = None

        # Save HTML
        html_path = os.path.join(self.chart_html_dir, f"{safe_name}_radar.html")
        try:
            fig.write_html(html_path)
            logger.info(f"Radar chart HTML saved for {company_data.get('company_name')} at {html_path}")
        except Exception as e:
            logger.error(f"Error saving radar chart HTML for {company_data.get('company_name')}: {e}")
            html_path = None

        return png_path, html_path

    # --- Report Generation Methods ---
    def generate_detailed_report(self, company_data, chart_png_path, chart_html_path):
        company_name = company_data.get('company_name', 'Unknown Company')
        ticker = company_data.get('ticker', '')
        wiki_summary = company_data.get('wikipedia_summary', '')
        if not wiki_summary:
            wiki_summary = company_data.get('google_overview', 'No summary available.')
        financial = company_data.get('financial_data', {})
        executive_bios = company_data.get('executive_bios', [])

        report_lines = [
            f"## {company_name} ({ticker})",
            "---",
            "### Financial Health Radar Chart"
        ]

        # If PNG path is available, embed it in Markdown
        if chart_png_path:
            rel_png = os.path.relpath(chart_png_path, os.path.dirname(__file__))
            report_lines.append(f"![Financial Health Radar Chart]({rel_png})")
        else:
            report_lines.append("Radar chart not available.")

        report_lines.extend([
            "### Business Summary",
            wiki_summary,
            "### Financial Metrics",
            "| Metric | Value |",
            "|--------|-------|"
        ])
        if financial:
            for key, value in financial.items():
                formatted_value = self.format_value(value)
                report_lines.append(f"| **{key}** | {formatted_value} |")
        else:
            report_lines.append("| No financial data available. |")

        report_lines.append("### Executive Bios")
        if executive_bios:
            for exec_bio in executive_bios:
                name = exec_bio.get('name', '').strip()
                bio = exec_bio.get('bio', '').strip()
                if not name or name.lower() in {"and", "&", "(", ")", ""}:
                    continue
                if re.fullmatch(r'[\W_]+', name):
                    continue
                if len(bio.split()) < 5 or any(token in bio.lower() for token in ["the and sign", "graphical representation", "additional details on"]):
                    continue
                report_lines.extend([f"**{name}**", bio])
        else:
            report_lines.append("Executive bios not available.")

        report_lines.append("### Lawsuit Snapshot")
        lawsuit_info = self.get_lawsuit_snapshot(company_name, MY_GOOGLE_API_KEY, MY_CSE_ID)
        report_lines.append(lawsuit_info)

        return "\n\n".join(report_lines)

    def generate_markdown_report(self, ranked_companies, detailed_reports):
        report = "# Competitive Analysis Report\n\n"
        report += "## Rankings\n\n"
        report += ranked_companies + "\n\n"
        for rep in detailed_reports:
            report += rep + "\n\n"
        report_file_path = os.path.join(os.path.dirname(__file__), "company_analysis_report.md")
        with open(report_file_path, "w") as file:
            file.write(report)
        print(f"Markdown report generated as '{report_file_path}'")

    # --- Utility Functions ---
    def format_value(self, value):
        if value is None or value == "N/A":
            return 'N/A'
        elif isinstance(value, (int, float)):
            if abs(value) >= 1e12:
                return f"${value/1e12:,.2f} Trillion"
            elif abs(value) >= 1e9:
                return f"${value/1e9:,.2f} Billion"
            elif abs(value) >= 1e6:
                return f"${value/1e6:,.2f} Million"
            else:
                return f"${value:,.2f}"
        return str(value)

    def get_wikipedia_summary(self, company_name, ticker):
        primary_query = f"{company_name} ({ticker})" if ticker else f"{company_name} (company)"
        logger.info(f"Primary Wikipedia query: {primary_query}")
        try:
            page = wikipedia.page(primary_query)
            return page.summary
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
            logger.warning(f"Primary query failed for {company_name}: {e}")
        for suffix in [" Inc.", " Corporation", " Co."]:
            alt_query = f"{company_name}{suffix} (company)"
            logger.info(f"Trying alternative query: {alt_query}")
            try:
                page = wikipedia.page(alt_query)
                return page.summary
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                logger.warning(f"Alternative query '{alt_query}' failed: {e}")
        search_result = self.google_search(f"{company_name} company overview", MY_GOOGLE_API_KEY, MY_CSE_ID, num=1)
        if search_result:
            snippet = search_result[0].get('snippet', 'No summary found.')
            return snippet
        return "No summary found."

    def get_executive_bios(self, company_name):
        """
        Fetch executive bios for a company using the Wikipedia page.
        """
        logger.info(f"Fetching executive bios for {company_name}")
        try:
            page = wikipedia.page(f"{company_name} (company)")
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
            logger.warning(f"Primary Wikipedia page query failed for {company_name}: {e}")
            try:
                page = wikipedia.page(company_name)
            except Exception as e2:
                logger.error(f"Error fetching fallback Wikipedia page for {company_name}: {e2}")
                return [{"name": "Not Available", "title": "Not Available", "bio": "No executive information available."}]
        html_content = page.html()
        soup = BeautifulSoup(html_content, 'lxml')
        candidates = extract_executive_candidates_from_infobox(soup)
        if candidates:
            candidates = filter_priority_candidates(candidates)
        else:
            logger.info(f"No 'Key people' infobox found for {company_name}, using fallback extraction.")
            candidates = fallback_extraction(soup)
        if not candidates:
            logger.warning(f"No valid executives found for {company_name}.")
            return [{"name": "Not Available", "title": "Not Available", "bio": "No executive information available."}]
        executive_bios = []
        for cand in candidates:
            name = cand.get("name")
            role = cand.get("role")
            try:
                logger.info(f"Fetching bio for executive: {name}")
                try:
                    bio = wikipedia.summary(name, sentences=2)
                except wikipedia.exceptions.PageError:
                    bio = wikipedia.summary(f"{company_name} {name}", sentences=2)
            except wikipedia.exceptions.DisambiguationError:
                bio = f"Additional details on {name} can be found via other sources."
            except Exception as e:
                logger.error(f"Error retrieving bio for {name}: {e}")
                bio = "Biography not available."
            executive_bios.append({
                "title": role if role else "N/A",
                "name": name,
                "bio": bio
            })
        return executive_bios

    def get_lawsuit_snapshot(self, company_name, api_key, cse_id):
        search_term = f"{company_name} lawsuit"
        results = self.google_search(search_term, api_key, cse_id, num=1)
        if results:
            snippet = results[0].get('snippet', 'No lawsuit information found.')
            link = results[0].get('link', '')
            return f"{snippet}\n\nRead more: {link}" if link else snippet
        return "No lawsuit information found."

    def google_search(self, search_term, api_key, cse_id, **kwargs):
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
            return res.get('items', [])
        except Exception as e:
            logger.error(f"Google search error for query '{search_term}': {e}")
            return []

# --- OpenAI Agents ---
class SummarizerAgent:
    def summarize(self, text):
        prompt = f"Please summarize the following text in three paragraphs with up to 4 sentences each:\n\n{text}"
        return generate_text(prompt)

class RankerAgent:
    def rank(self, summaries):
        prompt = f"Rank the following company summaries based on overall performance, financial health, market position, and leadership:\n\n{summaries}"
        return generate_text(prompt)

def generate_text(prompt, model="gpt-4o-mini", max_tokens=2000, temperature=0.7):
    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes and ranks company data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI generate_text error: {e}")
        return "Error generating text."

# --- Graph Visualization (Optional) ---
def visualize_graph(compiled_graph, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    ascii_file = os.path.join(output_dir, "graph_ascii.txt")
    with open(ascii_file, "w") as f:
        ascii_art = compiled_graph.get_graph().draw_ascii()
        f.write(ascii_art)
    print(f"ASCII graph saved to {ascii_file}")
    mermaid_file = os.path.join(output_dir, "graph_mermaid.mmd")
    with open(mermaid_file, "w") as f:
        mermaid_syntax = compiled_graph.get_graph().draw_mermaid()
        f.write(mermaid_syntax)
    print(f"Mermaid graph saved to {mermaid_file}")

# --- Example Usage ---
if __name__ == "__main__":
    analyzer = CompanyAnalyzer()
    user_input = ["UBER", "LYFT"]
    api_keys = {
        "google_search": MY_GOOGLE_API_KEY,
        "google_cse_id": MY_CSE_ID,
        "openai": MY_OPENAI_API_KEY
    }
    result = analyzer.analyze_companies(user_input, api_keys)
    pprint(result)
    visualize_graph(analyzer.compiled_graph)