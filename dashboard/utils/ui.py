"""Consistent, accessible light styling and top navigation for the dashboard."""
import streamlit as st

PAGES = [
    ("pages/1_Overview.py", "Overview", "📊"),
    ("pages/2_Predictive_Maintenance.py", "Predictive", "🛡️"),
    ("pages/3_Production_Analysis.py", "Production", "📈"),
    ("pages/4_Operational_Optimization.py", "Operations", "⚙️"),
    ("pages/5_Maintenance_Priorities.py", "Priorities", "📋"),
    ("pages/6_Well_Details.py", "Well Details", "🔎"),
]

COLORS = {
    "bg": "#FFFFFF", "surface": "#FFFFFF", "raised": "#EAF6F5",
    "text": "#16243B", "muted": "#60748C", "border": "#DFE8F1",
    "accent": "#087F80", "blue": "#3475D5", "orange": "#C47A24",
    "red": "#CC5360", "grid": "#EDF2F7", "hover": "#F2F8FA",
}


def palette():
    """Shared palette for chart traces and page content."""
    return COLORS


def apply_theme():
    """Set a single light design; no runtime theme state or toggle."""
    st.markdown("""
    <style>
      :root { color-scheme: light; }
      html, body, .stApp, [data-testid="stAppViewContainer"],
      [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important; color: #16243B !important;
      }
      [data-testid="stHeader"] { background: #FFFFFF !important; }
      [data-testid="stSidebar"], [data-testid="stSidebarContent"],
      [data-testid="stSidebarUserContent"] { background: #F7FAFD !important; }
      [data-testid="stSidebar"] { border-right: 1px solid #DFE8F1; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #405771 !important; }
      .block-container {
          max-width: 1550px;
          padding-top: 5rem !important;
          padding-right: 2rem;
          padding-bottom: 4rem;
          padding-left: 2rem;
      }
      h1 { color: #14283F !important; font-size: 2.05rem !important;
           font-weight: 750 !important; letter-spacing: -.035em; }
      h2, h3 { color: #14283F !important; letter-spacing: -.02em; }
      [data-testid="stCaptionContainer"] { color: #60748C !important; }
      [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #DFE8F1;
        border-top: 3px solid #13A6A0; padding: 1.1rem 1.2rem;
        border-radius: 14px; box-shadow: 0 8px 26px rgba(23,52,86,.045);
      }
      [data-testid="stMetricLabel"] { color: #60748C !important; font-size: .88rem; }
      [data-testid="stMetricValue"] { color: #16243B !important; font-weight: 750; }
      [data-testid="stPlotlyChart"] {
        background: #FFFFFF; border: 1px solid #DFE8F1;
        padding: .45rem .6rem; border-radius: 14px;
        box-shadow: 0 5px 24px rgba(23,52,86,.035); overflow: hidden;
      }
      [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #DFE8F1; border-radius: 12px; overflow: hidden;
      }
      [data-testid="stVerticalBlockBorderWrapper"] { border-color: #DFE8F1; }
      hr { border-color: #DFE8F1 !important; }
      a[data-testid="stPageLink-NavLink"] {
        display: flex; justify-content: center; align-items: center; text-align: center;
        background: #FFFFFF; border: 1px solid #DFE8F1; border-radius: 12px;
        min-height: 50px; padding: .45rem .2rem; color: #28415D !important;
        font-weight: 650; transition: background .15s ease, border-color .15s ease;
        text-decoration: none;
      }
      a[data-testid="stPageLink-NavLink"]:hover {
        background: #F2F8FA; border-color: #13A6A0; color: #087F80 !important;
      }
      a[data-testid="stPageLink-NavLink"][aria-current="page"] {
        background: #EAF6F5; color: #087F80 !important; border-color: #13A6A0;
      }
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
      [data-testid="stDateInput"] input {
        background: #FFFFFF !important; color: #16243B !important;
        border-color: #D7E2EC !important;
      }
      div[data-baseweb="tag"] { background: #D9F2EE !important; }
      div[data-baseweb="tag"] span, div[data-baseweb="tag"] svg {
        color: #086F6A !important; fill: #086F6A !important;
      }
      [data-testid="stAlert"] {
        background: #ECF7F8 !important; color: #18394D !important;
        border: 1px solid #CBE8E8 !important; border-radius: 12px;
      }
      [data-testid="stAlert"] p,
      [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color: #18394D !important;
      }
      [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: #FFF7E8 !important; border-color: #F1DFB6 !important;
      }
      [data-testid="stSidebar"] [data-testid="stAlert"] p,
      [data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color: #785316 !important;
      }
      [data-testid="stSidebar"] hr { margin: .7rem 0; }
      @media (max-width: 900px) {
        .block-container { padding: 1.3rem .8rem 3rem; }
        h1 { font-size: 1.7rem !important; }
      }
    </style>
    """, unsafe_allow_html=True)


def app_header():
    """Render brand and horizontal navigation identically on every page."""
    apply_theme()
    st.markdown("""
      <div style="display:flex;align-items:center;gap:.75rem;
                  padding:.2rem 0 .3rem 0;">
        <span style="font-size:1.5rem">🛢️</span>
        <div>
          <div style="font-size:1.15rem;font-weight:800;letter-spacing:-.025em;
                      color:#14283F">INVISIBLE OIL <span style="color:#087F80">/ DETECTOR</span></div>
          <div style="font-size:.72rem;font-weight:650;letter-spacing:.13em;
                      color:#60748C">FIELD INTELLIGENCE · PRODUCTION & RELIABILITY</div>
        </div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    nav_cols = st.columns(6, gap="small")
    for col, (path, name, icon) in zip(nav_cols, PAGES):
        with col:
            st.page_link(path, label=f"{icon} {name}", use_container_width=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


def style_chart(fig, height=None):
    c = palette()
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color=c["text"], family="Arial, sans-serif", size=12),
        title_font=dict(color=c["text"], size=16),
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"], linecolor=c["border"]),
        yaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"], linecolor=c["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=22, r=22, t=58, b=24),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=c["text"]),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig
