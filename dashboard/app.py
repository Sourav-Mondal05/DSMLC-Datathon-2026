"""WellGuard — simple Streamlit dashboard for the datathon.

What this file does
-------------------
1. Shows the outcomes of Member 1 (data engineering) and Member 2 (machine
   learning + inspection prioritization) in ONE easy-to-run website.
2. Loads PRIVATE local CSV files containing recent production, well risk
   scores, maintenance priorities, and equipment failure history.
3. Displays five tabs: Overview, Failure Risk, Maintenance, Well Explorer,
   and Model Results. Provides filters, charts, and downloadable tables.
4. Shows a helpful setup message if an input file is missing; it does not
   invent production values or pretend demo scores came from a trained model.
5. Never retrains the model. Run training/prediction scripts separately,
   then refresh this dashboard to see the latest generated outputs.

Expected private inputs, relative to the project root
-----------------------------------------------------
- outputs/well_risk_predictions.csv   (from src/predict.py)
- outputs/maintenance_priorities.csv (from src/optimization.py)
- outputs/dashboard_daily.csv         (from prepare_dashboard_data.py)
- outputs/dashboard_recent_well_daily.csv  (from prepare_dashboard_data.py)
- outputs/dashboard_failures.csv      (optional, from prepare_dashboard_data.py)
- results/model_metrics.json         (aggregate, from src/train_model.py)

How to start
------------
From the repository root, run: streamlit run dashboard/app.py

Scientific and data-use cautions
--------------------------------
- A risk_score (e.g., 0.80) is an UNCALIBRATED relative model score, not an
  80% probability that the well will fail.
- 'Hypothetical exposed oil' is a scenario, not proven recoverable oil or
  measured financial savings. Maintenance priorities need operator review.
- Historical daily production data is not real-time sensor monitoring.
- The outputs directory contains restricted well-level information and must
  NOT be committed to a public GitHub repository.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RESULTS = ROOT / "results"

st.set_page_config(page_title="WellGuard | Oil Well Intelligence", page_icon="🛢️", layout="wide")

# Small amount of styling; all main elements are native Streamlit controls.
st.markdown("""
<style>
.block-container { max-width: 1350px; padding-top: 1.7rem; }
h1, h2, h3 { color: #15333A; }
div[data-testid="stMetric"] { border: 1px solid #E3ECEB; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_csv_cached(path_string, modified):
    """Cache a CSV until its file modification time changes."""
    return pd.read_csv(path_string, low_memory=False)


@st.cache_data(show_spinner=False)
def read_json_cached(path_string, modified):
    with open(path_string, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(name, date_columns=()):
    path = OUT / name
    if not path.is_file():
        return pd.DataFrame()
    df = read_csv_cached(str(path), path.stat().st_mtime_ns).copy()
    for c in date_columns:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "uwi" in df:
        df["uwi"] = df["uwi"].astype("string").str.strip()
    return df


def load_metrics():
    path = RESULTS / "model_metrics.json"
    return read_json_cached(str(path), path.stat().st_mtime_ns) if path.is_file() else {}


def fmt(value, digits=1):
    return f"{value:,.{digits}f}" if pd.notna(value) else "—"


def download(df, filename):
    if not df.empty:
        st.download_button("Download displayed table (CSV)", df.to_csv(index=False).encode("utf-8"),
                           filename, "text/csv", key="dl_" + filename)


def missing_file_help(files):
    st.info("To display this section, generate the following private local file(s): "
            + ", ".join(f"`outputs/{name}`" for name in files))


# Load only compact, precomputed CSVs. The original 5-million-row Parquet file
# is intentionally not opened on every Streamlit page refresh.
daily = load_csv("dashboard_daily.csv", ["production_date"])
recent = load_csv("dashboard_recent_well_daily.csv", ["production_date"])
failures = load_csv("dashboard_failures.csv", ["failure_start_date"])
risks = load_csv("well_risk_predictions.csv", ["as_of_date"])
priorities = load_csv("maintenance_priorities.csv", ["as_of_date"])
metrics = load_metrics()

if not risks.empty and {"uwi", "risk_score"}.issubset(risks.columns):
    risks["risk_score"] = pd.to_numeric(risks["risk_score"], errors="coerce")
    risks = risks.dropna(subset=["uwi", "risk_score"])
    risks = risks[risks["risk_score"].between(0, 1)]
    risks = risks.sort_values("as_of_date" if "as_of_date" in risks else "risk_score")
    risks = risks.drop_duplicates("uwi", keep="last")
    risks["risk_percentile"] = risks["risk_score"].rank(pct=True) * 100
else:
    risks = pd.DataFrame()

if not priorities.empty and "priority_score" in priorities:
    priorities["priority_score"] = pd.to_numeric(priorities["priority_score"], errors="coerce")

st.title("🛢️ WellGuard — Oil Well Intelligence")
st.caption("Historical production monitoring · 7-day failure-risk ranking · inspection priorities")
st.warning("Decision-support prototype. Risk scores are **relative, uncalibrated ML scores**—not failure probabilities. "
           "Production-loss figures are estimates or hypothetical scenarios, not proven savings.", icon="ℹ️")

with st.sidebar:
    st.header("Input files")
    status = [
        ("ML predictions", not risks.empty),
        ("Maintenance priorities", not priorities.empty),
        ("Production summaries", not daily.empty and not recent.empty),
        ("Failure history", not failures.empty),
        ("Model metrics", bool(metrics)),
    ]
    for label, ready in status:
        st.write(("✅ " if ready else "◻️ ") + label)
    st.caption("Files are read from the repository's private `outputs/` folder, and from `results/` for metrics.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

overview_tab, risk_tab, maintenance_tab, well_tab, quality_tab = st.tabs([
    "📊 Overview", "🛡️ Failure Risk", "🛠️ Maintenance", "🔎 Well Explorer", "📈 Model Results"
])

with overview_tab:
    st.subheader("Field overview")
    left, mid, right = st.columns(3)
    left.metric("Wells scored by ML", f"{risks['uwi'].nunique():,}" if not risks.empty else "—")
    high_count = int((priorities["priority"] == "High").sum()) if "priority" in priorities else None
    mid.metric("High inspection priority", f"{high_count:,}" if high_count is not None else "—")
    if not daily.empty and "total_oil_m3" in daily:
        end = daily["production_date"].max()
        last_30 = daily[daily["production_date"] >= end - pd.Timedelta(days=29)]
        right.metric("Oil produced · last 30 calendar days (m³)", fmt(last_30["total_oil_m3"].sum(), 0))
    else:
        right.metric("Recent oil produced", "—")

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("**Daily field oil production**")
        if not daily.empty and {"production_date", "total_oil_m3"}.issubset(daily.columns):
            last_90 = daily.sort_values("production_date").tail(90)
            fig = px.line(last_90, x="production_date", y="total_oil_m3",
                          labels={"production_date": "Date", "total_oil_m3": "Oil (m³)"})
            fig.update_traces(line_color="#137A77")
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_file_help(["dashboard_daily.csv"])
    with right:
        st.markdown("**Inspection priority counts**")
        if "priority" in priorities:
            counts = priorities["priority"].value_counts().rename_axis("Priority").reset_index(name="Wells")
            fig = px.bar(counts, x="Priority", y="Wells", color="Priority",
                         color_discrete_map={"High": "#C95E5E", "Medium": "#E6A64A", "Low": "#137A77"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_file_help(["maintenance_priorities.csv"])

    st.markdown("**Top wells needing review**")
    if not priorities.empty and "priority_score" in priorities:
        fields = [c for c in ["uwi", "priority", "risk_score", "priority_score",
                              "daily_oil_at_risk_m3", "recommended_action"] if c in priorities]
        top = priorities.sort_values("priority_score", ascending=False).head(15)
        st.dataframe(top[fields], use_container_width=True, hide_index=True)
    else:
        missing_file_help(["maintenance_priorities.csv"])

with risk_tab:
    st.subheader("Predictive maintenance — next 7 days")
    st.caption("Ranks well-days by predicted association with a future recorded failure. "
               "High risk rank ≠ known probability of failure.")
    if risks.empty:
        missing_file_help(["well_risk_predictions.csv"])
    else:
        ranked = risks.sort_values("risk_score", ascending=False).copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Scored wells", f"{len(ranked):,}")
        c2.metric("Highest raw model score", fmt(ranked["risk_score"].max(), 3))
        c3.metric("Latest prediction date", str(ranked["as_of_date"].max().date()) if "as_of_date" in ranked else "—")
        st.markdown("**Top 20 wells by relative risk rank**")
        top = ranked.head(20).sort_values("risk_percentile")
        fig = px.bar(top, x="risk_percentile", y="uwi", orientation="h",
                     labels={"risk_percentile": "Relative risk percentile (0–100)", "uwi": "Well"})
        fig.update_traces(marker_color="#C95E5E")
        st.plotly_chart(fig, use_container_width=True)
        fields = [c for c in ["uwi", "as_of_date", "risk_score", "risk_percentile",
                              "oil_baseline_7d", "historical_shortfall_indicator_m3"] if c in ranked]
        st.dataframe(ranked[fields], use_container_width=True, hide_index=True)
        download(ranked[fields], "risk_ranking_private.csv")

with maintenance_tab:
    st.subheader("Maintenance inspection priorities")
    st.caption("Priority index = relative risk rank × recent daily oil baseline. "
               "This is a screening heuristic, not a verified cost optimization.")
    if priorities.empty:
        missing_file_help(["maintenance_priorities.csv"])
    else:
        display = priorities.sort_values("priority_score", ascending=False).copy() if "priority_score" in priorities else priorities
        c1, c2, c3 = st.columns(3)
        c1.metric("Wells prioritized", f"{len(display):,}")
        c2.metric("High priority", str(int((display["priority"] == "High").sum())) if "priority" in display else "—")
        if "hypothetical_exposed_oil_m3" in display:
            c3.metric("Hypothetical oil exposure (m³)", fmt(pd.to_numeric(
                display["hypothetical_exposed_oil_m3"], errors="coerce").sum(), 0))
        else:
            c3.metric("Hypothetical oil exposure", "—")
        st.info("Exposure assumes downtime for the specified scenario. It is **not** actual loss, "
                "avoidable loss, or proven savings.")
        fields = [c for c in ["uwi", "priority", "priority_score", "risk_score",
                              "daily_oil_at_risk_m3", "hypothetical_exposed_oil_m3",
                              "historical_median_job_estimate", "recommended_action"] if c in display]
        st.dataframe(display[fields], use_container_width=True, hide_index=True)
        download(display[fields], "maintenance_priorities_private.csv")

with well_tab:
    st.subheader("Well explorer — historical operating conditions")
    if recent.empty or "uwi" not in recent:
        missing_file_help(["dashboard_recent_well_daily.csv"])
    else:
        wells = sorted(recent["uwi"].dropna().astype(str).unique())
        default_well = str(risks.sort_values("risk_score", ascending=False).iloc[0]["uwi"]) if not risks.empty else wells[0]
        index = wells.index(default_well) if default_well in wells else 0
        well = st.selectbox("Select a well", wells, index=index)
        series = recent[recent["uwi"].astype(str) == well].sort_values("production_date")
        risk_row = risks[risks["uwi"].astype(str) == well] if not risks.empty else pd.DataFrame()
        priority_row = priorities[priorities["uwi"].astype(str) == well] if "uwi" in priorities else pd.DataFrame()
        c1, c2, c3 = st.columns(3)
        c1.metric("Oil produced · recent window (m³)", fmt(series["oil_produced_m3"].sum(), 0))
        c2.metric("Relative model risk score", fmt(risk_row.iloc[0]["risk_score"], 3) if not risk_row.empty else "—")
        c3.metric("Review priority", str(priority_row.iloc[0]["priority"]) if not priority_row.empty and "priority" in priority_row else "—")
        for column, title in [
            ("oil_produced_m3", "Daily oil produced (m³)"),
            ("pcp_pump_rpm", "Pump rotation speed (RPM)"),
            ("pump_efficiency_pct", "Pump efficiency (stored as decimal proportion)"),
            ("hours_down", "Downtime (hours)"),
        ]:
            if column in series and series[column].notna().any():
                fig = px.line(series, x="production_date", y=column, title=title,
                              labels={"production_date": "Date", column: title})
                fig.update_traces(line_color="#137A77")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Recorded failure history**")
        if not failures.empty and "uwi" in failures:
            history = failures[failures["uwi"].astype(str) == well]
            if history.empty:
                st.info("No recorded failure event for this well in the provided failure log.")
            else:
                st.dataframe(history.sort_values("failure_start_date", ascending=False),
                             use_container_width=True, hide_index=True)
        else:
            missing_file_help(["dashboard_failures.csv"])

with quality_tab:
    st.subheader("How reliable is the model?")
    if not metrics:
        st.info("Model performance file missing: `results/model_metrics.json`. Run `python src/train_model.py`.")
    else:
        selected = metrics.get("selected_model", "Not recorded")
        model = metrics.get("metrics", {}).get(selected, {})
        test = model.get("test", {})
        baseline = metrics.get("metrics", {}).get("oil_decline_baseline", {}).get("test", {})
        st.write("**Selected model:**", selected.replace("_", " ").title())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test ROC-AUC", fmt(test.get("roc_auc"), 3))
        c2.metric("Test PR-AUC", fmt(test.get("average_precision"), 3))
        c3.metric("Baseline PR-AUC", fmt(baseline.get("average_precision"), 3))
        c4.metric("Test positive-label rate", f"{100 * test['prevalence']:.2f}%" if test.get("prevalence") is not None else "—")
        st.caption("Testing uses later dates. Positive labels are well-days, not distinct equipment failures. "
                   "These are historical evaluation results, not live operating performance.")
        st.markdown("**Model limitations**")
        for note in metrics.get("notes", []):
            st.write("• " + str(note))
        st.caption("Verify the recorded metrics against your team’s latest rerun of the full data pipeline before presenting.")

st.divider()
st.caption("Built for the DSMLC Datathon 2026 · Historical decision-support demonstration · "
           "Keep original and derived well-level data private.")
