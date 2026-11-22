import streamlit as st
import plotly.express as px
from utils.ui import style_chart, palette
from utils.dashboard_helpers import context, init_page, downloadable

init_page('Predictive Maintenance', 'Ranked well-level failure probabilities from the connected model, where available')
p, f, risks, demo = context()
if risks.empty:
    st.warning('No model predictions loaded. Add outputs/failure_risk_predictions.csv or .parquet with columns uwi and failure_risk_score (0–1).')
    st.stop()
rank = risks.sort_values('failure_risk_score', ascending=False).copy()
rank['risk_percent'] = (100 * rank['failure_risk_score']).round(1)
rank['risk_band'] = pd_cut = rank['failure_risk_score'].apply(lambda x: 'High' if x >= .7 else ('Medium' if x >= .4 else 'Low'))
if demo: st.warning('These scores are randomly constructed demo values, NOT trained-model predictions.')
col1, col2, col3 = st.columns(3)
col1.metric('Wells with scores', len(rank))
col2.metric('High-risk wells · illustrative threshold ≥70%', (rank['risk_band'] == 'High').sum())
col3.metric('Highest score', f"{rank['risk_percent'].max():.1f}%")
st.caption('Risk bands (≥70%, 40–69%, <40%) are dashboard display thresholds, not validated operational thresholds.')
fig = px.bar(rank.head(20).sort_values('risk_percent'), x='risk_percent', y='uwi', orientation='h', color='risk_band',
             color_discrete_map={'High':palette()['red'],'Medium':palette()['orange'],'Low':palette()['accent']},
             labels={'risk_percent':'Failure risk (%)','uwi':'Well'})
style_chart(fig, height=max(350, min(800, len(rank)*30)))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(rank[['uwi','risk_percent','risk_band','risk_source']], use_container_width=True, hide_index=True)
downloadable(rank, 'ranked_failure_risks.csv')
