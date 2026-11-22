"""
INVISIBLE OIL LOSS DETECTOR — REAL-DATA STREAMLIT WEBSITE

What this file does
-------------------
1. Shows actual historical well data prepared by prepare_real_data.py.
2. Displays five sections: Overview, Failure Risk, Maintenance Priorities,
   Individual Wells, and ML Validation. All plotted data comes from files.
3. Connects Member 2's real outputs: well_risk_predictions.csv,
   maintenance_priorities.csv, and results/model_metrics.json.
4. Uses actual well IDs and failure events. It never generates synthetic wells,
   simulated failures, fictional volumes, or randomly assigned risk values.
5. If a required file is missing, displays a setup message instead of quietly
   replacing the real dataset with demo data.
6. Separates observed production, recorded failure events, model scores, and
   scenario calculations. An uncalibrated risk score is not a probability.
7. Reads restricted well-level data ONLY from local outputs/ and data/;
   do not push any derived well-level tables or raw competition Parquet to
   public GitHub, or publicly expose them without provider permission.

Run from repository root:
    python -m streamlit run dashboard/app.py

Prerequisites:
    python dashboard/prepare_real_data.py
    python src/predict.py
    python src/optimization.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULTS = ROOT / "results"

st.set_page_config(page_title="Invisible Oil Loss Detector", page_icon="🛢️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1430px; padding-top: 1.5rem;}
[data-testid="stMetric"] {background: #f3faf9; border: 1px solid #cde4e1;
    border-radius: 13px; padding: 14px;}
h1,h2,h3 {color: #0d2d3d;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _csv(path: str, modified: int) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
def _json(path: str, modified: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(name: str, *, location: Path = OUT, dates=()) -> pd.DataFrame:
    p = location / name
    if not p.is_file():
        return pd.DataFrame()
    df = _csv(str(p), p.stat().st_mtime_ns).copy()
    for col in dates:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "uwi" in df:
        df["uwi"] = df["uwi"].astype("string").str.strip()
    return df


def load_json(name: str, *, location: Path) -> dict:
    p = location / name
    return _json(str(p), p.stat().st_mtime_ns) if p.is_file() else {}


def num(value, decimals=0):
    return f"{value:,.{decimals}f}" if pd.notna(value) else "—"


def show_table(df, cols=None, title=None, max_rows=50):
    if title:
        st.markdown(f"**{title}**")
    if df.empty:
        st.info("No matching records found.")
        return
    if cols:
        df = df[[x for x in cols if x in df.columns]]
    st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)


def missing(name: str, instructions: str = ""):
    st.info(f"Missing private file: `outputs/{name}`. {instructions}")


daily = load_csv("real_daily.csv", dates=["production_date"])
area_daily = load_csv("real_area_daily.csv", dates=["production_date"])
wells = load_csv("real_wells.csv", dates=["first_date", "last_date"])
history = load_csv("real_well_history.csv", dates=["production_date"])
failures = load_csv("real_failures.csv", dates=["failure_start_date"])
jobs = load_csv("real_jobs.csv", dates=["last_job_date"])
risks = load_csv("well_risk_predictions.csv", dates=["as_of_date"])
priorities = load_csv("maintenance_priorities.csv", dates=["as_of_date"])
metrics = load_json("model_metrics.json", location=RESULTS)
summary = load_json("real_data_summary.json", location=OUT)

st.title("🛢️ Invisible Oil Loss Detector")
st.caption("Real historical oil-well production • equipment failure risk • maintenance decision support")
st.info("All figures displayed are computed from provided historical datasets or the trained model. "
        "There is **no synthetic-data fallback**. The source records are historical, not live telemetry.")

if daily.empty or wells.empty:
    st.error("Real production files are not available. This site does not show fabricated data.")
    st.code("python src/data_cleaning.py\npython dashboard/prepare_real_data.py\n"
            "python -m streamlit run dashboard/app.py", language="powershell")
    st.warning("Keep private well-level CSV and Parquet files outside public GitHub. "
               "A public deployment showing restricted well-level records needs permission from the data provider.")
    st.stop()

# An explicit source-date filter makes it clear what time period is displayed.
all_areas = sorted(wells.get("field_operating_area", pd.Series(dtype=str)).dropna().unique().tolist())
with st.sidebar:
    st.header("Filters")
    area = st.selectbox("Operating area", ["All areas"] + all_areas)
    first_day = daily["production_date"].min().date()
    last_day = daily["production_date"].max().date()
    dates = st.date_input("Historical date range", value=(first_day, last_day),
                          min_value=first_day, max_value=last_day)
    st.divider()
    st.caption(f"Data source: cleaned production records ({first_day} to {last_day})")
    st.caption(f"Total wells in prepared data: {len(wells):,}")
    if st.button("Reload files"):
        st.cache_data.clear()
        st.rerun()

if not isinstance(dates, (list, tuple)) or len(dates) != 2:
    st.info("Select both the start and end date to continue.")
    st.stop()
start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
filtered_wells = wells if area == "All areas" else wells[wells["field_operating_area"] == area]
if area == "All areas":
    selected_daily = daily[daily["production_date"].between(start, end)].copy()
else:
    selected_daily = area_daily[
        area_daily["production_date"].between(start, end)
        & area_daily["field_operating_area"].eq(area)
    ].copy()
selected_daily = selected_daily.sort_values("production_date")
selected_failures = failures.copy()
if not failures.empty:
    selected_failures = failures[failures["failure_start_date"].between(start, end)].copy()
    if area != "All areas" and "field_operating_area" in selected_failures:
        selected_failures = selected_failures[selected_failures["field_operating_area"].eq(area)]

# Join only to select the current area; no row multiplication or hidden label use.
well_ids = set(filtered_wells["uwi"].dropna().astype(str))
if "uwi" in risks:
    risks = risks[risks["uwi"].isin(well_ids)].sort_values("as_of_date")
    risks = risks.drop_duplicates("uwi", keep="last")
if "uwi" in priorities:
    priorities = priorities[priorities["uwi"].isin(well_ids)]

t1, t2, t3, t4, t5 = st.tabs([
    "📊 Overview", "🛡️ Failure Risk", "🛠️ Maintenance", "🔎 Well Explorer", "📈 Model Results"
])

with t1:
    st.subheader("Production and reliability overview")
    a, b, c, d = st.columns(4)
    a.metric("Well IDs in selected area", f"{len(filtered_wells):,}")
    b.metric("Oil produced · m³", num(selected_daily["oil_m3"].sum()))
    c.metric("Downtime · hours", num(selected_daily["downtime_hours"].sum()))
    d.metric("Recorded failures", f"{len(selected_failures):,}" if not failures.empty else "Unavailable")

    if selected_daily.empty:
        st.warning("No production observations match these filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Daily oil production — actual reported totals**")
            fig = px.line(selected_daily, x="production_date", y="oil_m3",
                          labels={"production_date": "Date", "oil_m3": "Oil production (m³)"})
            fig.update_traces(line_color="#148d8b")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Daily downtime — reported hours**")
            fig = px.line(selected_daily, x="production_date", y="downtime_hours",
                          labels={"production_date": "Date", "downtime_hours": "Downtime (hours)"})
            fig.update_traces(line_color="#da8253")
            st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.markdown("**Recorded failure types**")
        if not selected_failures.empty and "failure_type" in selected_failures:
            counts = selected_failures["failure_type"].fillna("Unknown").value_counts().reset_index()
            counts.columns = ["Failure component", "Events"]
            fig = px.bar(counts.head(12), x="Failure component", y="Events")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No failure data available for this selection.")
    with col2:
        show_table(filtered_wells.sort_values("oil_m3", ascending=False),
                   ["uwi", "field_operating_area", "oil_m3", "downtime_hours", "observations"],
                   "Highest historical producing wells", 15)
    st.caption("All production and downtime totals are historical observations; wells with no recent production may remain in the total well count.")

with t2:
    st.subheader("Predictive maintenance — next 7 days")
    st.caption("The scores are **uncalibrated relative model scores**, not literal failure probabilities.")
    if risks.empty or "risk_score" not in risks:
        missing("well_risk_predictions.csv", "Run `python src/predict.py` after training the ML model.")
    else:
        risks["risk_score"] = pd.to_numeric(risks["risk_score"], errors="coerce")
        risks = risks.dropna(subset=["risk_score"])
        risks["risk_rank_percentile"] = 100 * risks["risk_score"].rank(pct=True)
        risks = risks.sort_values("risk_score", ascending=False)
        st.metric("Wells scored", f"{len(risks):,}")
        if "as_of_date" in risks:
            st.caption(f"Scores are from historical snapshots through {risks['as_of_date'].max().date()}.")
        n = st.slider("Top wells to display", min_value=5, max_value=min(50, len(risks)),
                      value=min(20, len(risks))) if len(risks) >= 5 else len(risks)
        top = risks.head(n).sort_values("risk_score")
        fig = px.bar(top, x="risk_score", y="uwi", orientation="h",
                     labels={"risk_score": "Relative model score (uncalibrated)", "uwi": "Well ID"})
        fig.update_traces(marker_color="#148d8b")
        fig.update_layout(height=max(350, n * 23))
        st.plotly_chart(fig, use_container_width=True)
        show_table(risks, ["uwi", "as_of_date", "risk_score", "risk_rank_percentile", "oil_baseline_7d"],
                   "Ranked well-level predictions", 100)

with t3:
    st.subheader("Risk-based maintenance priorities")
    st.caption("Priority is based on risk rank × recent production baseline. "
               "It is a screening heuristic, not a validated economic optimizer.")
    if priorities.empty or "priority" not in priorities:
        missing("maintenance_priorities.csv", "Run `python src/optimization.py` after predictions.")
    else:
        cnt = priorities["priority"].fillna("Unknown").value_counts().reset_index()
        cnt.columns = ["Priority", "Wells"]
        c1, c2, c3 = st.columns(3)
        c1.metric("High priority", int((priorities["priority"] == "High").sum()))
        c2.metric("Medium priority", int((priorities["priority"] == "Medium").sum()))
        c3.metric("Wells ranked", len(priorities))
        fig = px.bar(cnt, x="Priority", y="Wells", color="Priority",
                     color_discrete_map={"High": "#c85b5a", "Medium": "#de9b40", "Low": "#148d8b"})
        st.plotly_chart(fig, use_container_width=True)
        levels = st.multiselect("Priorities to display", ["High", "Medium", "Low"], default=["High", "Medium"])
        choice = priorities[priorities["priority"].isin(levels)].copy()
        if "priority_score" in choice:
            choice = choice.sort_values("priority_score", ascending=False)
        show_table(choice, ["uwi", "priority", "priority_score", "risk_score", "daily_oil_at_risk_m3",
                            "hypothetical_exposed_oil_m3", "historical_median_job_estimate", "recommended_action"],
                   "Maintenance review list", 100)
        st.caption("Hypothetical exposed oil uses an assumed downtime duration; it does not equal proven recoverable production or savings.")

with t4:
    st.subheader("Individual well explorer")
    options = sorted(filtered_wells["uwi"].dropna().astype(str).tolist())
    if not options:
        st.warning("No well IDs in this selection.")
    else:
        picked = st.selectbox("Choose a real well ID", options)
        detail = filtered_wells[filtered_wells["uwi"].eq(picked)]
        a, b, c = st.columns(3)
        a.metric("Historical produced oil · m³", num(detail["oil_m3"].iloc[0]))
        b.metric("Reported downtime · h", num(detail["downtime_hours"].iloc[0]))
        score = risks[risks["uwi"].eq(picked)] if "uwi" in risks else pd.DataFrame()
        c.metric("ML relative risk score", num(score["risk_score"].iloc[0], 3) if not score.empty else "Not scored")
        st.markdown("**Real sensor and production history — last 120 calendar days**")
        hist = history[history["uwi"].eq(picked)].sort_values("production_date") if "uwi" in history else pd.DataFrame()
        if hist.empty:
            st.info("No recent well-day records available for this well. The well may be inactive or have missing observations.")
        else:
            available = [col for col in ["oil_produced_m3", "water_produced_m3", "hours_down",
                                          "pcp_pump_rpm", "pump_efficiency_pct", "tubing_pressure_kpa",
                                          "casing_pressure_kpa"] if col in hist and hist[col].notna().any()]
            if available:
                metric = st.selectbox("Measurement", available)
                fig = px.line(hist, x="production_date", y=metric, markers=False,
                              labels={"production_date": "Date", metric: metric.replace("_", " ")})
                fig.update_traces(line_color="#148d8b")
                st.plotly_chart(fig, use_container_width=True)
            show_table(hist.sort_values("production_date", ascending=False),
                       ["production_date", "oil_produced_m3", "hours_down", "pcp_pump_rpm", "pump_efficiency_pct"],
                       "Recent daily observations", 30)
        if "uwi" in failures:
            f = failures[failures["uwi"].eq(picked)].sort_values("failure_start_date", ascending=False)
            show_table(f, ["failure_start_date", "failure_type", "cause"], "Recorded failure history", 30)
        if "uwi" in jobs:
            show_table(jobs[jobs["uwi"].eq(picked)],
                       ["jobs_logged", "last_job_date", "median_estimated_job_cost"],
                       "Historical maintenance overview", 10)

with t5:
    st.subheader("Machine learning — evaluation")
    if not metrics:
        st.info("`results/model_metrics.json` is missing. Train the model to view verified evaluation metrics.")
    else:
        selected = metrics.get("selected_model", "Unknown")
        test = metrics.get("metrics", {}).get(selected, {}).get("test", {})
        baseline = metrics.get("metrics", {}).get("oil_decline_baseline", {}).get("test", {})
        st.write(f"**Model selected on validation data:** `{selected}`")
        a, b, c = st.columns(3)
        a.metric("Test ROC-AUC", num(test.get("roc_auc", float("nan")), 3))
        b.metric("Test PR-AUC", num(test.get("average_precision", float("nan")), 3))
        c.metric("Baseline PR-AUC", num(baseline.get("average_precision", float("nan")), 3))
        st.caption("Chronological test observations are well-days, not distinct failure events. "
                   "ROC-AUC is not classification accuracy, and PR-AUC should be compared with event prevalence.")
        compared = []
        for name, result in metrics.get("metrics", {}).items():
            score = result.get("test", {}).get("average_precision")
            if score is not None:
                compared.append({"Model": name, "Test PR-AUC": score})
        if compared:
            fig = px.bar(pd.DataFrame(compared), x="Model", y="Test PR-AUC")
            st.plotly_chart(fig, use_container_width=True)
        imp = RESULTS / "feature_importance.png"
        if imp.exists():
            st.image(str(imp), caption="Permutation importance on validation records")
        st.markdown("**Limitations**")
        st.write("Recorded failures are sparse in earlier years; risk scores are uncalibrated; "
                 "historical associations do not prove that a maintenance action prevents failure. "
                 "Claims of saved oil or money require real intervention studies.")

st.divider()
st.caption("Datathon 2026 · Confidential Strathcona competition data must remain in approved environments. "
           "Do not publish raw or well-level derived data without permission.")
