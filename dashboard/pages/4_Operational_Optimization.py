import streamlit as st
import plotly.express as px
from utils.ui import style_chart, palette
from utils.dashboard_helpers import context, init_page, per_well, downloadable

init_page('Operational Optimization', 'Inspect pump efficiency, rotational speed, pressure and operating patterns')
p, f, risks, demo = context()
metric_choice = st.selectbox('Measurement', ['pump_efficiency_pct','pcp_pump_rpm','casing_pressure_kpa','tubing_pressure_kpa'])
if metric_choice not in p or p[metric_choice].notna().sum() == 0:
    st.warning(f'{metric_choice} is not available in the current production file.')
else:
    series = p.groupby('production_date', as_index=False)[metric_choice].mean()
    fig = px.line(series, x='production_date', y=metric_choice, title='Field-wide daily mean (observed records)')
    style_chart(fig, height=370)
    st.plotly_chart(fig, use_container_width=True)
left, right = st.columns(2)
with left:
    if {'pcp_pump_rpm','pump_efficiency_pct'}.issubset(p.columns):
        fig = px.scatter(p.dropna(subset=['pcp_pump_rpm','pump_efficiency_pct']).sample(n=min(len(p.dropna(subset=['pcp_pump_rpm','pump_efficiency_pct'])), 3500), random_state=7),
                         x='pcp_pump_rpm', y='pump_efficiency_pct', color='oil_produced_m3',
                         title='Pump speed vs efficiency', labels={'oil_produced_m3':'Oil · m³'})
        style_chart(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info('Pump RPM or efficiency unavailable.')
with right:
    if {'casing_pressure_kpa','tubing_pressure_kpa'}.issubset(p.columns):
        fig = px.scatter(p.dropna(subset=['casing_pressure_kpa','tubing_pressure_kpa']).sample(n=min(len(p.dropna(subset=['casing_pressure_kpa','tubing_pressure_kpa'])), 3500), random_state=7),
                         x='casing_pressure_kpa', y='tubing_pressure_kpa', color='oil_produced_m3',
                         title='Casing vs tubing pressure')
        style_chart(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info('Casing or tubing pressure unavailable.')
st.subheader('Mean operating characteristics by well')
measures = [c for c in ['pump_efficiency_pct','pcp_pump_rpm','casing_pressure_kpa','tubing_pressure_kpa','hours_down'] if c in p]
summary = p.groupby('uwi', as_index=False)[measures].mean() if measures else p[['uwi']].drop_duplicates()
st.dataframe(summary, hide_index=True, use_container_width=True)
st.caption('These are descriptive operational patterns; causal inefficiency or optimal setpoints have not been established.')
downloadable(summary, 'operating_summary.csv')
