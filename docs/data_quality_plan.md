# Data Quality, Enrichment, and Automation Plan

This document outlines how Infera validates third-party data, enriches executive biographies, and prepares the pipeline for scale. It covers the steps currently implemented in the codebase and the next milestones required to support productized usage.

## 1. Financial Data Validation

### Current Safeguards
- **Multi-endpoint cross-checks:** `graphAgent.CompanyAnalyzer.build_financial_snapshot` now pulls metrics from `info`, `fast_info`, quarterly statements, income statements, and cash-flow/balance-sheet tables. Each metric is tracked with its source label so discrepancies can be inspected in the generated report.
- **Missing-data protection:** The helper `_is_missing_value` filters out `None`, `N/A`, and NaN values before surfacing results. When a metric cannot be recovered, the report explicitly displays `N/A` instead of stale data.
- **Human-readable audit trail:** Reports note which Yahoo Finance endpoints provided metrics. This enables analysts to spot when a fallback path (e.g., `yfinance:income_statement`) was required.

### Next Steps
- **SEC Filing Cross-Checks:** Integrate the SEC EDGAR API to validate revenue, cash, and debt figures against official filings. Surface discrepancies inside the report and flag companies that diverge from filings beyond a tolerance threshold.
- **Caching & Freshness Windows:** Store snapshots in a document database (e.g., MongoDB or PostgreSQL) with timestamps so repeated analyses reuse cached metrics within a freshness window, reducing API load and cost.
- **Data Quality Alerts:** Add background validation jobs that compare cached results against new pulls. If a metric shifts unexpectedly, notify users via email (leveraging the captured lead list) or in-app alerts.

## 2. Executive Biography Automation

### Current Safeguards
- **Infobox parsing with fallbacks:** The analyzer extracts "Key people" from Wikipedia infoboxes and falls back to paragraph scanning when the infobox is missing.
- **Search-powered enrichment:** When a biography is too short, the app automatically performs a Google Custom Search (`"<name> <company> leadership biography"`) and uses the resulting snippet as a concise bio. This ensures coverage when Wikipedia content is sparse.

### Next Steps
- **Structured Source Priority:** Add connectors for company newsroom pages and LinkedIn public bios. Use a priority order (official site → LinkedIn → press releases → Wikipedia) so the cleanest data wins.
- **Automated Refresh Queue:** Schedule periodic jobs (Celery/Redis) to refresh executive bios weekly. Cache HTML snapshots and diff changes to highlight leadership moves.
- **Entity Resolution:** Introduce lightweight fuzzy matching (e.g., spaCy or RapidFuzz) to confirm that retrieved bios match the intended executive before storing them.

## 3. Scalable Data Collection Workflow

### Current Safeguards
- **State-machine orchestration:** LangGraph keeps network calls isolated per node, making it straightforward to add retries, caching, or asynchronous execution.
- **Lead capture groundwork:** Email leads collected in `output/email_leads.csv` can be funneled into CRM tooling for customer development and to notify users about data-quality improvements.

### Next Steps
- **Task Queue & Worker Pool:** Move longer-running fetches (e.g., multi-company analyses) into a worker queue. Provide user-facing progress updates while the backend scales horizontally.
- **Observability:** Instrument nodes with tracing (OpenTelemetry) and central logging so failures in upstream APIs are visible and alertable.
- **Data Lake Export:** Persist normalized snapshots into cloud object storage (S3/GCS) for downstream analytics and to power historical trend charts.

By staging improvements through these phases, Infera can deliver trustworthy insights while keeping operational costs lean and enabling enterprise-scale workflows.
