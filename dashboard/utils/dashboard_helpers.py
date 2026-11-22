"""Shared data adapters, filters and visuals for the Invisible Oil Loss dashboard."""
from pathlib import Path
from utils.ui import app_header, palette, style_chart
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT / 'outputs', ROOT.parent / 'outputs', ROOT / 'data', ROOT.parent / 'data']
COLORS = {'oil': '#16a6a3', 'water': '#5b8def', 'downtime': '#f4a261'}


def find_file(names):
    for base in DATA_DIRS:
        for name in names:
            p = base / name
            if p.is_file():
                return p
    return None


def numeric(df, cols):
    for col in cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def standardize_production(df):
    df = df.copy()
    aliases = {'date': 'production_date', 'well_id': 'uwi', 'oil_volume_m3': 'oil_produced_m3',
               'water_volume_m3': 'water_produced_m3', 'downtime_hours': 'hours_down'}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df and v not in df})
    required = ['uwi', 'production_date', 'oil_produced_m3']
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError('Production file missing: ' + ', '.join(missing))
    df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce').dt.normalize()
    df['uwi'] = df['uwi'].astype('string').str.strip()
    cols = ['oil_produced_m3', 'water_produced_m3', 'hours_down', 'casing_pressure_kpa',
            'tubing_pressure_kpa', 'pcp_pump_rpm', 'pump_efficiency_pct', 'water_cut_pct',
            'oil_loss_m3', 'estimated_oil_loss_m3', 'expected_oil_m3', 'baseline_oil_m3']
    df = numeric(df, cols)
    df = df.dropna(subset=['uwi', 'production_date']).drop_duplicates()
    if 'water_cut_pct' not in df and 'water_produced_m3' in df:
        denominator = df['water_produced_m3'] + df['oil_produced_m3']
        df['water_cut_pct'] = 100 * df['water_produced_m3'].div(denominator.where(denominator > 0))
    if 'water_cut_pct' in df and df['water_cut_pct'].dropna().between(0, 1).all() and df['water_cut_pct'].notna().any():
        # Percentages expressed as fractions.
        df['water_cut_pct'] *= 100
    if 'estimated_oil_loss_m3' in df and 'oil_loss_m3' not in df:
        df['oil_loss_m3'] = df['estimated_oil_loss_m3']
    return df.sort_values(['uwi', 'production_date'])


@st.cache_data(show_spinner=False)
def demo_data():
    rng = np.random.default_rng(2026)
    dates = pd.date_range('2026-01-01', periods=120, freq='D')
    frames = []
    for i in range(18):
        well = f'WELL-{i+1:03d}'
        t = np.arange(len(dates))
        base = 28 + 4 * (i % 6) + 0.15 * i
        decline = np.maximum(t - (60 + i % 24), 0) * (0.08 + (i % 5) * 0.035)
        oil = np.maximum(base + 2 * np.sin(t / 9 + i) - decline + rng.normal(0, 1.4, len(t)), 0)
        water = np.maximum(8 + i * 0.5 + t * 0.04 + rng.normal(0, 1.2, len(t)), 0)
        down = rng.choice([0, 0, 0, 0, 0, 2, 6, 12], size=len(t))
        oil[down >= 6] *= 0.65
        frames.append(pd.DataFrame({
            'uwi': well, 'production_date': dates, 'oil_produced_m3': oil,
            'water_produced_m3': water, 'hours_down': down,
            'pcp_pump_rpm': 170 + i * 5 + rng.normal(0, 12, len(t)) - decline * 0.3,
            'pump_efficiency_pct': np.clip(85 - decline * 0.7 + rng.normal(0, 4, len(t)), 0, 100),
            'casing_pressure_kpa': 950 + 35 * i + rng.normal(0, 25, len(t)),
            'tubing_pressure_kpa': 740 + 19 * i + rng.normal(0, 20, len(t)),
            'expected_oil_m3': np.full(len(t), base),
        }))
    production = pd.concat(frames, ignore_index=True)
    production['water_cut_pct'] = 100 * production['water_produced_m3'] / (production['oil_produced_m3'] + production['water_produced_m3'])
    production['oil_loss_m3'] = (production['expected_oil_m3'] - production['oil_produced_m3']).clip(lower=0)
    failures = pd.DataFrame({
        'uwi': [f'WELL-{n:03d}' for n in [2, 5, 8, 11, 14, 17]],
        'failure_type': ['Rods', 'Rods', 'Pump', 'Rods', 'Tubing', 'Rods'],
        'failure_start_date': pd.to_datetime(['2026-03-11', '2026-03-20', '2026-04-02', '2026-03-26', '2026-04-08', '2026-04-19'])
    })
    return production, failures


@st.cache_data(show_spinner=False)
def read_parquet(path, stamp):
    return pd.read_parquet(path)


def load_data():
    prodpath = find_file(['cleaned_production.parquet', 'daily_production.parquet'])
    if prodpath is None:
        p, f = demo_data()
        return p, f, True, 'No production Parquet found; showing synthetic demonstration data.'
    try:
        prod = standardize_production(read_parquet(str(prodpath), prodpath.stat().st_mtime_ns))
        failpath = find_file(['production_failures.parquet'])
        if failpath:
            failures = read_parquet(str(failpath), failpath.stat().st_mtime_ns).copy()
            if 'failure_start_date' in failures:
                failures['failure_start_date'] = pd.to_datetime(failures['failure_start_date'], errors='coerce')
        else:
            failures = pd.DataFrame(columns=['uwi', 'failure_type', 'failure_start_date'])
        return prod, failures, False, f'Production source: {prodpath}'
    except Exception as exc:
        st.error(f'Unable to load production dataset: {exc}')
        st.stop()


def load_risks(demo, prod):
    if demo:
        latest = prod.sort_values('production_date').groupby('uwi').tail(1).copy()
        scores = np.linspace(.08, .93, len(latest))
        latest = latest.sort_values('uwi')
        latest['failure_risk_score'] = scores
        latest['risk_source'] = 'SYNTHETIC DEMO (not model output)'
        return latest[['uwi', 'failure_risk_score', 'risk_source']]
    path = find_file(['failure_risk_predictions.parquet', 'failure_risk_predictions.csv',
                      'model_predictions.parquet', 'model_predictions.csv',
                      'predictions.parquet', 'predictions.csv'])
    if path is None:
        return pd.DataFrame(columns=['uwi', 'failure_risk_score', 'risk_source'])
    try:
        df = pd.read_csv(path) if path.suffix == '.csv' else read_parquet(str(path), path.stat().st_mtime_ns)
        df = df.rename(columns={k: 'failure_risk_score' for k in ['risk_score', 'failure_probability', 'predicted_failure_probability'] if k in df and 'failure_risk_score' not in df})
        if not {'uwi', 'failure_risk_score'}.issubset(df.columns):
            st.warning(f'Prediction file {path.name} needs uwi and failure_risk_score columns.')
            return pd.DataFrame(columns=['uwi', 'failure_risk_score', 'risk_source'])
        df['uwi'] = df['uwi'].astype('string').str.strip()
        df['failure_risk_score'] = pd.to_numeric(df['failure_risk_score'], errors='coerce')
        # Only documented 0–1 probabilities accepted; percentages cannot silently be mixed in.
        df = df[df['failure_risk_score'].between(0, 1)]
        for dtcol in ['prediction_date', 'as_of_date']:
            if dtcol in df:
                df[dtcol] = pd.to_datetime(df[dtcol], errors='coerce')
                df = df.sort_values(dtcol)
                break
        df = df.drop_duplicates('uwi', keep='last')
        df['risk_source'] = path.name
        return df[['uwi', 'failure_risk_score', 'risk_source']]
    except Exception as exc:
        st.warning(f'Could not load predictions: {exc}')
        return pd.DataFrame(columns=['uwi', 'failure_risk_score', 'risk_source'])


def init_page(title, subtitle):
    app_header()
    st.title(title)
    st.caption(subtitle)
    st.divider()


def context():
    production, failures, demo, message = load_data()
    st.sidebar.markdown('### Data source')
    if demo:
        st.sidebar.warning('DEMO · Synthetic records and simulated risk scores')
    else:
        st.sidebar.success('REAL · Production dataset')
    st.sidebar.caption(message)
    st.sidebar.markdown('### Filters')
    wells = sorted(production['uwi'].dropna().astype(str).unique().tolist())
    picked = st.sidebar.multiselect('Wells', wells, default=[],
                                         placeholder=f'All {len(wells)} wells (select to filter)',
                                         key='filter_wells')
    chosen = picked if picked else wells
    st.sidebar.caption(f'{len(chosen)} of {len(wells)} wells shown')
    start = production['production_date'].min().date()
    end = production['production_date'].max().date()
    selected = st.sidebar.date_input('Production date range', value=(start, end), min_value=start, max_value=end, key='filter_dates')
    if not isinstance(selected, (list, tuple)) or len(selected) != 2:
        st.info('Select a start and end date to continue.')
        st.stop()
    first, last = pd.Timestamp(selected[0]), pd.Timestamp(selected[1])
    subset = production[production['uwi'].isin(chosen) & production['production_date'].between(first, last)].copy()
    fsub = failures[failures['uwi'].astype('string').isin(chosen)].copy() if 'uwi' in failures else failures
    if 'failure_start_date' in fsub:
        fsub = fsub[fsub['failure_start_date'].between(first, last)]
    if subset.empty:
        st.warning('No production observations match the selected filters.')
        st.stop()
    risks = load_risks(demo, production)
    risks = risks[risks['uwi'].isin(chosen)]
    if demo:
        st.info('**Demonstration mode:** all data and risk scores shown are synthetic. No operational conclusions should be drawn.')
    return subset, fsub, risks, demo


def metric(label, val, suffix=''):
    st.metric(label, f'{val:,.1f}{suffix}' if pd.notna(val) else 'N/A')


def line(data, x, y, title, labels=None):
    if y not in data or data[y].notna().sum() == 0:
        st.info(f'{title}: measurement not available in this dataset.')
        return
    chart = px.line(data, x=x, y=y, title=title, labels=labels or {}, markers=False)
    chart.update_traces(line_color=palette()['accent'], line_width=2.6)
    style_chart(chart, height=340)
    st.plotly_chart(chart, use_container_width=True)


def daily_sum(df, cols):
    cols = [c for c in cols if c in df]
    return df.groupby('production_date', as_index=False)[cols].sum(min_count=1)


def per_well(df):
    fields = {'oil_produced_m3': 'sum', 'hours_down': 'sum', 'pump_efficiency_pct': 'mean',
              'pcp_pump_rpm': 'mean', 'water_cut_pct': 'mean', 'oil_loss_m3': 'sum'}
    fields = {k: v for k, v in fields.items() if k in df}
    return df.groupby('uwi', as_index=False).agg(fields)


def downloadable(df, name):
    st.download_button('Download filtered CSV', df.to_csv(index=False).encode('utf-8'),
                       file_name=name, mime='text/csv')
