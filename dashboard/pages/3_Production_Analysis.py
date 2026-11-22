import streamlit as st
from utils.dashboard_helpers import context, init_page, metric, daily_sum, line, downloadable

init_page('Production Analysis', 'Oil, produced water, water cut and downtime trends')
p, f, risks, demo = context()
a,b,c = st.columns(3)
with a: metric('Total oil · m³', p['oil_produced_m3'].sum())
with b: metric('Total water · m³', p['water_produced_m3'].sum() if 'water_produced_m3' in p else float('nan'))
with c: metric('Downtime · hours', p['hours_down'].sum() if 'hours_down' in p else float('nan'))
daily = daily_sum(p, ['oil_produced_m3','water_produced_m3','hours_down'])
if 'water_produced_m3' in daily:
    denom = daily['oil_produced_m3'] + daily['water_produced_m3']
    daily['water_cut_pct'] = (100 * daily['water_produced_m3'] / denom.where(denom > 0))
elif 'water_cut_pct' in p:
    daily['water_cut_pct'] = p.groupby('production_date')['water_cut_pct'].mean().reindex(daily['production_date']).to_numpy()
for first, second, name1, name2 in [
    ('oil_produced_m3','water_produced_m3','Daily oil volume · m³','Daily produced water · m³'),
    ('water_cut_pct','hours_down','Field water cut · %','Daily downtime · hours')]:
    a,b = st.columns(2)
    with a: line(daily,'production_date',first,name1)
    with b: line(daily,'production_date',second,name2)
st.caption('Field water cut is calculated from aggregated water and oil volumes when both volumes are available.')
downloadable(daily, 'production_timeseries.csv')
