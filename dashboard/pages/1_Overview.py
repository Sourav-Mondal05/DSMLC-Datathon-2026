import streamlit as st
import plotly.express as px
from utils.ui import style_chart, palette
from utils.dashboard_helpers import context, init_page, metric, daily_sum, per_well, line, downloadable

init_page('Field Performance Overview', 'Production, downtime and recorded failure activity across selected wells')
p, f, risks, demo = context()
a,b,c,d = st.columns(4)
with a: st.metric('Active wells in selection', p['uwi'].nunique())
with b: metric('Oil produced · m³', p['oil_produced_m3'].sum())
with c: metric('Downtime · hours', p['hours_down'].sum() if 'hours_down' in p else float('nan'))
with d: st.metric('Recorded failures', len(f))
left, right = st.columns([1.65, 1])
with left:
    daily = daily_sum(p, ['oil_produced_m3', 'hours_down'])
    line(daily, 'production_date', 'oil_produced_m3', 'Daily field oil production · m³')
with right:
    st.subheader('Failure breakdown')
    if len(f) and 'failure_type' in f:
        counts = f['failure_type'].fillna('Unknown').value_counts().rename_axis('Failure type').reset_index(name='Events')
        chart = px.bar(counts, x='Failure type', y='Events', color_discrete_sequence=[palette()['blue']])
        style_chart(chart, height=340)
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info('No recorded failures found for the selected filters.')
st.subheader('Well performance')
st.dataframe(per_well(p), use_container_width=True, hide_index=True)
downloadable(per_well(p), 'well_overview.csv')
