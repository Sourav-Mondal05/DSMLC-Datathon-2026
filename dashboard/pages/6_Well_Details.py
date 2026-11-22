import streamlit as st
import plotly.express as px
from utils.ui import style_chart, palette
from utils.dashboard_helpers import context, init_page, metric, downloadable

init_page('Well Details', 'Inspect historical production and recorded equipment failures for an individual well')
p, f, risks, demo = context()
well = st.selectbox('Choose a well', sorted(p['uwi'].dropna().unique()))
view = p[p['uwi'] == well].sort_values('production_date')
a,b,c = st.columns(3)
with a: metric('Total oil · m³', view['oil_produced_m3'].sum())
with b: metric('Downtime · hours', view['hours_down'].sum() if 'hours_down' in view else float('nan'))
with c:
    score = risks.loc[risks['uwi']==well, 'failure_risk_score']
    st.metric('Failure score' + (' (DEMO)' if demo else ''), f'{score.iloc[0]*100:.1f}%' if len(score) else 'Not available')
chosen = [c for c in ['oil_produced_m3','water_produced_m3','pump_efficiency_pct','pcp_pump_rpm','hours_down','casing_pressure_kpa','tubing_pressure_kpa'] if c in view]
measure = st.selectbox('Plot measurement', chosen)
fig = px.line(view, x='production_date', y=measure, title=f'{well} · {measure}')
fig.update_traces(line_color=palette()['accent'], line_width=2.6)
style_chart(fig, height=390)
st.plotly_chart(fig, use_container_width=True)
st.subheader('Recorded failures in the selected date range')
matched = f[f['uwi'].astype(str) == str(well)] if 'uwi' in f else f
if matched.empty: st.info('No failure events for this well within the selected date range.')
else: st.dataframe(matched, use_container_width=True, hide_index=True)
st.subheader('Daily measurements')
st.dataframe(view.sort_values('production_date', ascending=False), use_container_width=True, hide_index=True)
downloadable(view, f'{well}_history.csv')
