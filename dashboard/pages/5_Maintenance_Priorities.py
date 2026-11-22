import streamlit as st
from utils.dashboard_helpers import context, init_page, per_well, downloadable

init_page('Maintenance Priorities', 'Screen wells for operator review using transparent provisional ranking rules')
p, f, risks, demo = context()
well = per_well(p)
if 'oil_loss_m3' in well:
    well = well.rename(columns={'oil_loss_m3':'indicative_shortfall_m3'})
else:
    well['indicative_shortfall_m3'] = float('nan')
if not risks.empty:
    well = well.merge(risks[['uwi','failure_risk_score']], on='uwi', how='left')
else:
    well['failure_risk_score'] = float('nan')
# Do not impute missing model risk with invented probabilities.
well['review_priority'] = 'Unranked'
well.loc[well['failure_risk_score'].between(.4,.7, inclusive='left'), 'review_priority'] = 'Medium (screening)'
well.loc[well['failure_risk_score'] >= .7, 'review_priority'] = 'High (screening)'
well.loc[well['failure_risk_score'] < .4, 'review_priority'] = 'Low (screening)'
well = well.sort_values(['failure_risk_score','indicative_shortfall_m3'], ascending=False, na_position='last')
if demo: st.warning('Simulated demo risks and shortfalls: not actionable maintenance recommendations.')
st.caption('Priority categories are illustrative screening rules only. Confirm with maintenance engineers before scheduling work.')
st.dataframe(well, use_container_width=True, hide_index=True)
if 'indicative_shortfall_m3' in well and well['indicative_shortfall_m3'].notna().any():
    volume = well['indicative_shortfall_m3'].clip(lower=0).sum()
    st.metric('Sum of preliminary production shortfalls · m³', f'{volume:,.1f}')
    st.caption('Not confirmed recoverable oil. Shortfalls may include planned downtime, noise and baseline error.')
    price = st.number_input('Optional assumed oil value (currency units / m³)', min_value=0.0, value=0.0, step=10.0)
    if price > 0:
        st.metric('Illustrative gross shortfall value (no costs)', f'{volume * price:,.0f}')
else:
    st.info('No estimated shortfall column is available. Connect outputs/production_features.parquet or an oil_loss_m3 column to report business impact.')
st.markdown('**Suggested operator workflow:** verify data → review downtime and recent maintenance → inspect high-priority wells → record findings.')
downloadable(well, 'maintenance_screening_priorities.csv')
