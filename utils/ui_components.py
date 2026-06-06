import streamlit as st


THEME = {
    "bg": "#181A1F",
    "sidebar": "#202124",
    "card": "#242529",
    "card_hover": "#2A2B31",
    "border": "#3A3B40",
    "text": "#F5F5F5",
    "muted": "#A3A3A3",
    "subtle": "#737373",
    "primary": "#7C5CFF",
    "primary_soft": "#2D275B",
    "secondary": "#A78BFA",
    "success": "#00B894",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "info": "#3B82F6",
}


PAGE_OPTIONS_LOCKED = [
    "Import dataset",
    "Column selection",
]

PAGE_OPTIONS_UNLOCKED = [
    "Dataset overview",
    "Descriptive statistics",
    "Visualizations",
    "Normality tests",
    "T-tests",
    "ANOVA",
    "Chi-square tests",
    "Z-tests",
    "Distribution fitting",
    "Central limit theorem",
]


def inject_global_css():
    """
    Injects the dark Midnight Violet dashboard theme.
    """

    st.markdown(
        f"""
        <style>
        /* Main app background */
        .stApp {{
            background-color: {THEME["bg"]};
            color: {THEME["text"]};
        }}

        /* Reduce top padding */
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {THEME["sidebar"]};
            border-right: 1px solid {THEME["border"]};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.1rem;
        }}

        /* Headings */
        h1, h2, h3, h4 {{
            color: {THEME["text"]};
            letter-spacing: -0.02em;
        }}

        /* Labels and text */
        label, p, span, div {{
            color: inherit;
        }}

        /* Cards */
        .stat-card {{
            background: {THEME["card"]};
            border: 1px solid {THEME["border"]};
            border-radius: 14px;
            padding: 18px;
            min-height: 104px;
        }}

        .stat-card-title {{
            font-size: 13px;
            color: {THEME["muted"]};
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .stat-card-value {{
            font-size: 28px;
            color: {THEME["text"]};
            font-weight: 800;
            line-height: 1.1;
        }}

        .stat-card-subtitle {{
            font-size: 12px;
            color: {THEME["muted"]};
            margin-top: 6px;
        }}

        .info-card {{
            background: {THEME["card"]};
            border: 1px solid {THEME["border"]};
            border-radius: 14px;
            padding: 18px;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 6px 12px;
            background: {THEME["card"]};
            border: 1px solid {THEME["border"]};
            color: {THEME["muted"]};
            font-size: 13px;
            font-weight: 650;
        }}

        .status-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }}

        .dot-success {{
            background: {THEME["success"]};
            box-shadow: 0 0 8px {THEME["success"]};
        }}

        .dot-warning {{
            background: {THEME["warning"]};
            box-shadow: 0 0 8px {THEME["warning"]};
        }}

        .app-logo-box {{
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]});
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 900;
        }}

        .sidebar-title {{
            font-size: 19px;
            font-weight: 850;
            color: {THEME["text"]};
            margin-bottom: 0px;
        }}

        .sidebar-subtitle {{
            font-size: 12px;
            color: {THEME["muted"]};
            margin-top: -2px;
        }}

        .sidebar-section {{
            font-size: 11px;
            color: {THEME["subtle"]};
            font-weight: 800;
            margin-top: 18px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 12px;
            border: 1px solid {THEME["border"]};
            background-color: {THEME["card"]};
            color: {THEME["text"]};
            font-weight: 700;
            transition: 0.15s ease-in-out;
        }}

        .stButton > button:hover {{
            border-color: {THEME["primary"]};
            color: {THEME["text"]};
            background-color: {THEME["card_hover"]};
        }}

        /* Primary buttons */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]});
            border: 1px solid {THEME["primary"]};
            color: white;
        }}

        /* Inputs */
        div[data-baseweb="select"] > div {{
            background-color: {THEME["card"]};
            border-color: {THEME["border"]};
            border-radius: 12px;
        }}

        .stTextInput input,
        .stNumberInput input {{
            background-color: {THEME["card"]};
            border: 1px solid {THEME["border"]};
            color: {THEME["text"]};
            border-radius: 12px;
        }}

        /* Dataframes */
        div[data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {THEME["border"]};
        }}

        /* Expander */
        details {{
            background-color: {THEME["card"]} !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 14px !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: {THEME["card"]};
            border-radius: 999px;
            border: 1px solid {THEME["border"]};
            padding: 8px 18px;
            color: {THEME["muted"]};
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {THEME["primary_soft"]};
            border-color: {THEME["primary"]};
            color: {THEME["text"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """
    Renders the main sidebar navigation.
    Returns selected page name.
    """

    with st.sidebar:
        col_logo, col_title = st.columns([0.25, 0.75])

        with col_logo:
            st.markdown('<div class="app-logo-box">Σ</div>', unsafe_allow_html=True)

        with col_title:
            st.markdown('<div class="sidebar-title">StatKit</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-subtitle">Statistical toolkit</div>', unsafe_allow_html=True)

        st.divider()

        has_dataset = st.session_state.get("df") is not None
        has_selected_dataset = st.session_state.get("selected_df") is not None

        st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)

        data_page = st.radio(
            "Data navigation",
            PAGE_OPTIONS_LOCKED,
            label_visibility="collapsed",
            key="data_nav",
        )

        st.markdown('<div class="sidebar-section">Analysis</div>', unsafe_allow_html=True)

        if has_selected_dataset:
            analysis_page = st.radio(
                "Analysis navigation",
                PAGE_OPTIONS_UNLOCKED,
                label_visibility="collapsed",
                key="analysis_nav",
            )
        else:
            st.caption("Upload a dataset and select columns to unlock analysis pages.")
            analysis_page = None

        st.divider()

        if has_selected_dataset:
            rows = st.session_state.selected_df.shape[0]
            cols = st.session_state.selected_df.shape[1]
            st.success(f"{rows:,} rows · {cols} selected columns")
        elif has_dataset:
            rows = st.session_state.df.shape[0]
            cols = st.session_state.df.shape[1]
            st.warning(f"{rows:,} rows · {cols} detected columns")
        else:
            st.info("No file loaded")

        if st.button("Reset project"):
            reset_dataset_state()
            st.rerun()

    # Priority: if user recently selected an analysis page, use it.
    # Streamlit radio widgets both exist, so this logic keeps navigation simple.
    if has_selected_dataset and analysis_page is not None:
        return analysis_page

    return data_page


def reset_dataset_state():
    """
    Clears uploaded and selected dataset state.
    """

    keys_to_clear = [
        "df",
        "selected_df",
        "column_summary",
        "selected_columns",
        "uploaded_file_name",
        "dist_fit_results",
        "dist_fit_data",
        "dist_fit_column_name",
        "dist_fit_bins_used",
        "clt_data",
        "clt_sample_means",
        "clt_column_name",
        "clt_sample_size_used",
        "clt_number_of_samples_used",
        "clt_bins_used",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def render_top_bar(page_title):
    """
    Renders a compact top bar for every page.
    """

    has_selected_dataset = st.session_state.get("selected_df") is not None

    left, right = st.columns([0.65, 0.35])

    with left:
        st.markdown(f"### {page_title}")

    with right:
        if has_selected_dataset:
            rows = st.session_state.selected_df.shape[0]
            cols = st.session_state.selected_df.shape[1]
            st.markdown(
                f"""
                <div style="text-align:right;">
                    <span class="status-pill">
                        <span class="status-dot dot-success"></span>
                        {rows:,} rows · {cols} columns selected
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="text-align:right;">
                    <span class="status-pill">
                        <span class="status-dot dot-warning"></span>
                        No selected dataset
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()


def metric_card(title, value, subtitle=""):
    """
    Reusable metric card.
    """

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-card-title">{title}</div>
            <div class="stat-card-value">{value}</div>
            <div class="stat-card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title, lines):
    """
    Reusable information card.
    """

    bullet_lines = "".join([f"<li>{line}</li>" for line in lines])

    st.markdown(
        f"""
        <div class="info-card">
            <h4 style="margin-top:0;">{title}</h4>
            <ul style="margin-bottom:0; color:{THEME["muted"]};">
                {bullet_lines}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_locked_message():
    """
    Message shown when user tries analysis before selecting columns.
    """

    st.warning("Please upload a dataset and select columns before using this analysis page.")