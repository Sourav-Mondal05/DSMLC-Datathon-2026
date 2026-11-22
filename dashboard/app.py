"""Invisible Oil Loss Detector — streamlined, interactive Streamlit UI.

What this file does
-------------------
1. Answers the project's operational question in five focused tabs:
   Overview, Hidden Oil Loss, Maintenance, Well Explorer, Model Evidence.
2. Reads real production-history summaries prepared by build_exports.py and
   Member 2's existing prediction and maintenance-priority CSVs.
3. Shows only necessary visuals; changes well, time range and priority via
   ordinary Streamlit widgets; never constructs synthetic examples.
4. Clearly separates OBSERVED oil output, HEURISTIC hidden-loss screening,
   UNCALIBRATED ML scores and HYPOTHETICAL downtime exposure.
5. Protects confidential competition records: PUBLIC mode is the default and
   NEVER reads private well-level files. To use local private data, explicitly
   set WELLGUARD_PRIVATE_LOCAL=1 before launching on your own computer.
6. Public mode displays existing public model-validation metrics and any
   provider-approved aggregate summary added to dashboard/approved_public_summary.json.
7. Nothing here retrains the model or changes operating equipment. Run Member
   1 and Member 2 scripts separately whenever the data or model changes.

Local Windows example (run from repo root):
    $env:WELLGUARD_PRIVATE_LOCAL = "1"
    python -m streamlit run dashboard/app.py

Cloud deployment: DO NOT set WELLGUARD_PRIVATE_LOCAL in public Streamlit Cloud.
Never publicly expose source Parquet or well-level derived CSV files without
explicit permission from the dataset provider.
"""
from __future__ import annotations

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from data_io import load_private, load_public, require_columns

NAVY = "#102A3C"
TEAL = "#009D93"
ORANGE = "#EC9C54"
RED = "#D36064"
MUTED = "#718296"

st.set_page_config(
    page_title="Invisible Oil Loss Detector", page_icon="🛢️", layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
:root {--primary: #009D93;}
.block-container {max-width: 1310px; padding-top: 1.3rem; padding-bottom: 3rem;}
[data-testid="stAppViewContainer"] {background: #F6F8FB;}
[data-testid="stHeader"] {background: #F6F8FB;}
[data-testid="stMetric"] {background: white; border: 1px solid #E2EAF0;
    border-radius: 14px; padding: 16px 20px; box-shadow:0 2px 13px #102A3C08;}
[data-testid="stMetricLabel"] {font-weight: 550; color: #687D91;}
[data-testid="stMetricValue"] {color: #102A3C;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    border: 1px solid #E2EAF0; border-radius: 14px; background: #fff; padding: 5px;}
div[data-baseweb="tab-list"] {gap: 8px;}
button[data-baseweb="tab"] {font-weight: 650;}
h1, h2, h3 {letter-spacing: -.025em; color: #102A3C;}
.hero {background: linear-gradient(120deg,#102A3C 0%,#123D52 67%,#056F71 100%);
    padding: 28px 34px; border-radius: 18px; color: #fff; margin-bottom: 20px;}
.hero .eyebrow {font-size:.75rem; letter-spacing:.14em; font-weight:750; color:#7FE6DD;}
.hero h1 {font-size:2.05rem; margin:10px 0 7px; color:#fff !important;}
.hero .sub {font-size:.96rem; color:#D9EAF0; max-width:760px;}
.section-lead {margin: 6px 0 20px; color:#647A8E; font-size:.94rem;}
.note {padding:10px 15px; background:#EAF6F4; color:#245D5D; border-radius:10px;
    border-left:3px solid #00A59B; font-size:.87rem; margin:10px 0 16px;}
</style>""", unsafe_allow_html=True)


def metric(v, decimals=0) -> str:
    """Human-scale metric formatting, preserving unavailable values."""
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.{decimals}f}"


def plot(fig, height=340):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=15, r=22, t=28, b=24),
        font=dict(family="Arial, sans-serif", color=NAVY, size=12),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#ECF0F3", zeroline=False),
        hoverlabel=dict(bgcolor="white", font_color=NAVY),
        legend=dict(orientation="h", y=1.10, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def line(df, x, y, color=TEAL, label=None, height=335):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], name=label or y,
                             mode="lines", line=dict(width=2.6, color=color)))
    plot(fig, height)


def alert(message):
    st.info(message)


PRIVATE = os.environ.get("WELLGUARD_PRIVATE_LOCAL", "").strip() == "1"

@st.cache_data(show_spinner="Loading project data…")
def get_data(private: bool):
    return load_private() if private else load_public()

try:
    d = get_data(PRIVATE)
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"The dashboard could not read its real data files: {exc}")
    st.stop()

st.markdown("""
<div class="hero">
  <div class="eyebrow">DSMLC 2026 · INDUSTRIAL INTELLIGENCE</div>
  <h1>Invisible Oil Loss Detector</h1>
  <div class="sub">Find unexpected production declines. Rank equipment-failure risk.
  Prioritize the wells that deserve an operator's attention.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About this dashboard")
    st.caption("Historical heavy-oil well data · 2021–2026")
    st.caption("Prediction target: recorded failure in the next 7 days")
    st.divider()
    if PRIVATE:
        st.success("LOCAL PRIVATE MODE · real well-level records")
        st.caption("Do not enable this mode on a publicly accessible server.")
    else:
        st.info("PUBLIC MODE · no individual well data loaded")
    if st.button("↻ Reload data", use_container_width=True):
        get_data.clear()
        st.rerun()
    st.divider()
    st.caption("Model scores are not calibrated probabilities. Shortfalls are screening indicators, not proven savings.")

if PRIVATE:
    daily = d["daily"]
    priority = d["priorities"]
    recent = d["recent"]
    candidates = d["candidates"]
    failures = d["failures"]
    risks = d["risks"]
    summary = d["summary"]
    metrics = d["metrics"]
    if daily.empty or not summary:
        st.error("No real dashboard exports were found. This app never substitutes demo data.")
        st.code("python src/data_cleaning.py\npython src/feature_engineering.py\n"
                "python dashboard/build_exports.py\npython src/train_model.py\n"
                "python src/predict.py\npython src/optimization.py", language="powershell")
        st.stop()

    st.markdown('<div class="note">Actual historical records · Private local analysis · No simulated data or live sensor feed</div>', unsafe_allow_html=True)
    t_overview, t_loss, t_action, t_well, t_model = st.tabs([
        "Overview", "Hidden Oil Loss", "Maintenance", "Well Explorer", "Model Evidence"
    ])

    with t_overview:
        st.subheader("The operating picture")
        st.markdown('<div class="section-lead">A few metrics that establish the scale of the problem—not every available sensor.</div>', unsafe_allow_html=True)
        a, b, c, e = st.columns(4)
        a.metric("Wells in source history", f'{int(summary.get("all_wells", 0)):,}')
        b.metric("Wells with ML scores", f"{len(risks):,}")
        c.metric("High inspection priority", f"{int((priority['priority'] == 'High').sum()):,}" if not priority.empty else "—")
        e.metric("Recorded failure events", f'{summary["recorded_failures"]:,}' if summary.get("recorded_failures") is not None else "—")
        st.caption("Well count covers all source statuses; ML scores cover recent producing wells only.")
        st.markdown("#### Historical production trend")
        last = daily["production_date"].max()
        first = daily["production_date"].min()
        selected = st.select_slider("Time window", options=["90 days", "180 days", "1 year", "All history"], value="180 days")
        days = {"90 days": 90, "180 days": 180, "1 year": 365, "All history": None}[selected]
        shown = daily if days is None else daily[daily["production_date"] >= last - pd.Timedelta(days=days - 1)]
        line(shown, "production_date", "oil_m3", label="Daily oil produced (m³)")
        st.caption(f"Observed oil-volume total by day, {shown['production_date'].min():%b %Y}–{last:%b %Y}. Shifts in well activity affect the aggregate trend.")
        if not priority.empty:
            st.markdown("#### Inspection workload at a glance")
            counts = priority["priority"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
            fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                                   marker_color=[RED, ORANGE, TEAL], text=counts.values, textposition="outside"))
            fig.update_layout(xaxis_title="Wells", yaxis_autorange="reversed")
            plot(fig, 215)

    with t_loss:
        st.subheader("Find production declines without reported downtime")
        st.markdown('<div class="section-lead">Candidate rule from feature engineering: ≥30% oil decline versus the prior 7-day baseline, with zero recorded downtime and positive operating hours.</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Wells flagged historically", f'{summary.get("candidate_wells", 0):,}')
        b.metric("Flagged well-days", f'{summary.get("candidate_well_days", 0):,}')
        c.metric("Indicative shortfall · m³", metric(summary.get("candidate_shortfall_m3")))
        st.caption("Indicative shortfall is baseline minus observed output on flagged days—not confirmed recoverable oil.")
        st.markdown("#### When did candidate declines occur?")
        timeframe = st.selectbox("Time range", ["Last 12 months", "All history"], key="loss_period")
        last = daily["production_date"].max()
        flagged = daily if timeframe == "All history" else daily[daily["production_date"] >= last - pd.Timedelta(days=365)]
        line(flagged, "production_date", "candidate_days", color=ORANGE, label="Flagged well-days per date")
        if not candidates.empty:
            st.markdown("#### Which wells merit a closer look?")
            min_days = st.slider("Minimum flagged days", min_value=1, max_value=20, value=3)
            selected = candidates[candidates["candidate_days"] >= min_days].sort_values("estimated_shortfall_m3", ascending=False)
            if selected.empty:
                alert("No wells match this threshold. Reduce the minimum flagged days.")
            else:
                top = selected.head(12).iloc[::-1]
                fig = go.Figure(go.Bar(x=top["estimated_shortfall_m3"], y=top["uwi"],
                                       orientation="h", marker_color=TEAL))
                fig.update_layout(xaxis_title="Estimated shortfall indicator (m³)", yaxis_title="Well ID")
                plot(fig, 385)
                with st.expander("See candidate well summary"):
                    st.dataframe(selected.head(100), use_container_width=True, hide_index=True)

    with t_action:
        st.subheader("Decide what to investigate first")
        st.markdown('<div class="section-lead">The ranking combines ML risk percentile with recent daily oil baseline. It helps schedule human review; it is not an automated repair order.</div>', unsafe_allow_html=True)
        if priority.empty:
            alert("No maintenance rankings found. Run `python src/predict.py` and then `python src/optimization.py`.")
        else:
            options = ["High", "Medium", "Low"]
            col1, col2, col3 = st.columns([1.2, 1.1, 1])
            with col1:
                chosen = st.multiselect("Priority", options, default=["High", "Medium"])
            with col2:
                areas = sorted(priority["field_operating_area"].dropna().astype(str).unique()) if "field_operating_area" in priority else []
                area = st.selectbox("Field", ["All fields"] + areas)
            with col3:
                count = st.slider("Show top wells", 5, 40, 15)
            subset = priority[priority["priority"].isin(chosen)].copy()
            if area != "All fields":
                subset = subset[subset["field_operating_area"].eq(area)]
            subset = subset.sort_values("priority_score", ascending=False)
            a, b, c = st.columns(3)
            a.metric("Wells in selection", len(subset))
            b.metric("High priority", int((subset["priority"] == "High").sum()))
            c.metric("Highest model risk percentile", f"{subset['risk_percentile'].max():.1f}" if len(subset) and "risk_percentile" in subset else "—")
            if subset.empty:
                alert("No wells match these filters.")
            else:
                st.markdown("#### Top inspection candidates")
                top = subset.head(count).iloc[::-1]
                fig = go.Figure(go.Bar(x=top["priority_score"], y=top["uwi"], orientation="h",
                                       marker_color=[{"High": RED, "Medium": ORANGE, "Low": TEAL}.get(k, TEAL) for k in top["priority"]]))
                fig.update_layout(xaxis_title="Priority score (risk rank × oil baseline)", yaxis_title="Well ID")
                plot(fig, max(300, min(630, 42 * len(top))))
                cols = ["uwi", "priority", "risk_percentile", "daily_oil_at_risk_m3",
                        "candidate_days", "estimated_shortfall_m3", "recommended_action"]
                st.dataframe(subset[[v for v in cols if v in subset]].head(100), use_container_width=True, hide_index=True)
                with st.expander("Explore a hypothetical downtime scenario (not savings)"):
                    duration = st.slider("Assumed downtime duration (days)", 1, 14, 3)
                    chosen_well = st.selectbox("Well", subset["uwi"].head(40).tolist())
                    row = subset[subset["uwi"].eq(chosen_well)].iloc[0]
                    daily_oil = float(row.get("daily_oil_at_risk_m3", 0) or 0)
                    st.metric("Production exposed in scenario · m³", metric(daily_oil * duration, 1))
                    st.caption("Scenario = recent oil baseline × assumed downtime. It is not a prediction of actual loss or intervention benefit.")
            st.caption("The maintenance categories and priority scores come from Member 2's optimization.py output.")

    with t_well:
        st.subheader("Investigate one well")
        st.markdown('<div class="section-lead">Compare reported production with its previous seven-day baseline, then check pump readings and logged failures.</div>', unsafe_allow_html=True)
        if recent.empty or "uwi" not in recent:
            alert("No recent well history available. Run `python dashboard/build_exports.py`.")
        else:
            preferred = priority["uwi"].tolist() if not priority.empty else []
            available = set(recent["uwi"].dropna().astype(str))
            options = [x for x in preferred if x in available]
            options += sorted(available - set(options))
            picked = st.selectbox("Choose a well (highest inspection priority shown first)", options)
            hist = recent[recent["uwi"].eq(picked)].sort_values("production_date")
            pr = priority[priority["uwi"].eq(picked)] if not priority.empty else pd.DataFrame()
            r = risks[risks["uwi"].eq(picked)] if not risks.empty else pd.DataFrame()
            f = failures[failures["uwi"].eq(picked)] if not failures.empty and "uwi" in failures else pd.DataFrame()
            a, b, c, e = st.columns(4)
            a.metric("Inspection priority", str(pr.iloc[0]["priority"]) if not pr.empty else "Unranked")
            b.metric("Relative risk percentile", f"{float(r.iloc[0]['risk_percentile']):.1f}" if not r.empty else "Not scored")
            c.metric("Flagged recent days", int(hist["possible_invisible_loss"].astype("string").str.lower().eq("true").sum()) if "possible_invisible_loss" in hist else "—")
            e.metric("Recorded failures", len(f))
            st.markdown("#### Oil output vs. historical baseline")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["production_date"], y=hist["oil_produced_m3"],
                                     name="Actual oil (m³/day)", mode="lines", line=dict(color=TEAL, width=2.4)))
            if "oil_baseline_7d" in hist:
                fig.add_trace(go.Scatter(x=hist["production_date"], y=hist["oil_baseline_7d"],
                                         name="Prior 7-day baseline", mode="lines", line=dict(color=NAVY, width=2, dash="dot")))
            if "possible_invisible_loss" in hist:
                flagged = hist[hist["possible_invisible_loss"].astype("string").str.lower().eq("true")]
                fig.add_trace(go.Scatter(x=flagged["production_date"], y=flagged["oil_produced_m3"],
                                         name="Candidate decline", mode="markers", marker=dict(size=10, color=RED, symbol="diamond")))
            fig.update_layout(yaxis_title="Produced oil (m³/day)", xaxis_title="Date")
            plot(fig, 370)
            columns = [v for v in ["pcp_pump_rpm", "pump_efficiency_pct", "hours_down", "tubing_pressure_kpa"] if v in hist and hist[v].notna().any()]
            if columns:
                metric_choice = st.selectbox("Check an operating signal", columns,
                                             format_func=lambda x: {"pcp_pump_rpm": "Pump speed (RPM)",
                                                                    "pump_efficiency_pct": "Pump efficiency (decimal)",
                                                                    "hours_down": "Downtime (hours)",
                                                                    "tubing_pressure_kpa": "Tubing pressure (kPa)"}.get(x, x))
                line(hist, "production_date", metric_choice, color=ORANGE, label=metric_choice, height=270)
            if not f.empty:
                with st.expander(f"View {len(f)} recorded failure events for this well"):
                    cols = [x for x in ["failure_start_date", "failure_type", "cause"] if x in f]
                    st.dataframe(f[cols].sort_values("failure_start_date", ascending=False), use_container_width=True, hide_index=True)
            st.caption("Shown observations are historical and do not establish that a specific failure will occur.")

    with t_model:
        st.subheader("How much evidence supports the model?")
        st.markdown('<div class="section-lead">The classifier is selected on 2026 validation data and evaluated against later 2026 records; the test figures are retrospective.</div>', unsafe_allow_html=True)
        model = metrics.get("selected_model")
        by_model = metrics.get("metrics", {})
        chosen_metrics = by_model.get(model, {}).get("test", {}) if model else {}
        baseline_metrics = by_model.get("oil_decline_baseline", {}).get("test", {})
        if not chosen_metrics:
            alert("The public model_metrics.json file is unavailable. Run `python src/train_model.py`.")
        else:
            a, b, c = st.columns(3)
            a.metric("ML test PR-AUC", metric(chosen_metrics.get("average_precision"), 3))
            b.metric("Oil-decline baseline PR-AUC", metric(baseline_metrics.get("average_precision"), 3))
            c.metric("ML test ROC-AUC", metric(chosen_metrics.get("roc_auc"), 3))
            st.caption(f"Selected model: {model}. Held-out test positives are labeled well-days, not distinct failures.")
            show = []
            for name, payload in by_model.items():
                score = payload.get("test", {}).get("average_precision")
                if score is not None:
                    show.append({"model": name.replace("_", " ").title(), "PR-AUC": score})
            if show:
                chart = pd.DataFrame(show)
                fig = go.Figure(go.Bar(x=chart["model"], y=chart["PR-AUC"],
                                       marker_color=[TEAL if n == model.replace("_", " ").title() else "#92AEB9" for n in chart["model"]]))
                fig.update_layout(yaxis_title="Test PR-AUC (higher is better)")
                plot(fig, 300)
            imp = pd.DataFrame(metrics.get("feature_importance", [])).head(8)
            if not imp.empty and {"feature", "ap_decrease"}.issubset(imp.columns):
                st.markdown("#### What signals mattered most?")
                imp = imp.sort_values("ap_decrease")
                fig = go.Figure(go.Bar(x=imp["ap_decrease"], y=imp["feature"], orientation="h", marker_color=TEAL))
                fig.update_layout(xaxis_title="Reduction in PR-AUC when shuffled", yaxis_title="")
                plot(fig, 325)
        st.warning("Interpretation limits: incomplete early failure logging; uncalibrated scores; overlapping 7-day failure labels; no measured intervention savings or proof of causality.")

else:
    summary = d["summary"]
    metrics = d["metrics"]
    st.markdown('<div class="note">Public-safe presentation · Individual wells, identifiers, and restricted source data are not loaded.</div>', unsafe_allow_html=True)
    if summary and summary.get("monthly"):
        st.subheader("Historical portfolio overview")
        st.caption("Showing only approved aggregate observations. These are not live sensor values.")
        stats = summary.get("summary", {})
        a, b, c = st.columns(3)
        a.metric("Wells in historical data", f'{stats.get("all_wells", 0):,}')
        b.metric("Flagged candidate well-days", f'{stats.get("candidate_well_days", 0):,}')
        c.metric("Recorded failures", f'{stats.get("recorded_failures", 0):,}')
        mon = pd.DataFrame(summary["monthly"])
        mon["month"] = pd.to_datetime(mon["month"], errors="coerce")
        tabs = st.tabs(["Production", "Candidate declines", "Model validation"])
        with tabs[0]:
            line(mon, "month", "oil_m3", label="Reported monthly oil production (m³)")
        with tabs[1]:
            line(mon, "month", "candidate_days", color=ORANGE, label="Candidate well-days per month")
            st.caption("Screening candidates, not proven recoverable production losses.")
        with tabs[2]:
            st.caption("The comparison below uses existing public, retrospective evaluation results.")
    else:
        st.subheader("Project evidence")
        st.info("Public production visuals require the dataset owner's approval of a non-identifying aggregate export. The model evidence below is already published in the repository.")
    chosen = metrics.get("selected_model", "")
    val = metrics.get("metrics", {}).get(chosen, {}).get("test", {})
    base = metrics.get("metrics", {}).get("oil_decline_baseline", {}).get("test", {})
    if val:
        st.markdown("#### Retrospective model validation")
        a, b, c = st.columns(3)
        a.metric("ML PR-AUC", metric(val.get("average_precision"), 3))
        b.metric("Baseline PR-AUC", metric(base.get("average_precision"), 3))
        c.metric("ML ROC-AUC", metric(val.get("roc_auc"), 3))
        split = st.radio("Evaluation period", ["Test", "Validation"], horizontal=True)
        rows = []
        for name, outcome in metrics.get("metrics", {}).items():
            score = outcome.get(split.lower(), {}).get("average_precision")
            if score is not None:
                rows.append({"Model": name.replace("_", " ").title(), "PR-AUC": float(score)})
        if rows:
            records = pd.DataFrame(rows)
            fig = go.Figure(go.Bar(x=records["Model"], y=records["PR-AUC"],
                                   marker_color=[TEAL if r == chosen.replace("_", " ").title() else "#A7B8C4" for r in records["Model"]]))
            fig.update_layout(yaxis_title="Average precision (PR-AUC)")
            plot(fig, 285)
        st.caption("PR-AUC is a ranking metric for rare failures; ROC-AUC is not prediction accuracy. Scores are not calibrated probabilities.")
        imp = pd.DataFrame(metrics.get("feature_importance", []))
        if not imp.empty and {"feature", "ap_decrease"}.issubset(imp.columns):
            imp = imp.nlargest(7, "ap_decrease").sort_values("ap_decrease")
            st.markdown("#### Most useful recorded signals")
            fig = go.Figure(go.Bar(y=imp["feature"], x=imp["ap_decrease"], orientation="h", marker_color=TEAL))
            fig.update_layout(xaxis_title="Decrease in PR-AUC when feature shuffled")
            plot(fig, 290)
    else:
        st.warning("The repository's model_metrics.json is missing. No metric values were fabricated.")
    st.markdown("#### What an operator would see in authorized local mode")
    st.write("A portfolio overview, hidden-loss candidate trends, inspection rankings, and individual well investigation charts. Restricted well-level records are deliberately unavailable to public viewers.")

st.divider()
st.caption("DSMLC Datathon 2026 · Historical and retrospective analysis · AI-assisted code; results require team verification. No verified production savings are claimed.")
