import streamlit as st


THEME = {
    "bg": "#12131A",
    "bg_2": "#181A1F",
    "sidebar": "#17181F",
    "sidebar_2": "#202124",
    "card": "#242529",
    "card_hover": "#2A2B31",
    "card_soft": "#20222A",
    "border": "#3A3B40",
    "border_soft": "rgba(255,255,255,0.08)",
    "text": "#F5F5F5",
    "muted": "#A3A3A3",
    "subtle": "#737373",
    "primary": "#7C5CFF",
    "primary_soft": "#2D275B",
    "primary_glow": "rgba(124, 92, 255, 0.35)",
    "secondary": "#A78BFA",
    "success": "#00B894",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "info": "#3B82F6",
}


DATA_PAGES = [
    "Import dataset",
    "Column selection",
]

ANALYSIS_PAGES = [
    "Dataset overview",
    "Descriptive statistics",
    "Visualizations",
    "T-tests",
    "Non-parametric tests",
    "ANOVA",
    "Chi-square tests",
    "Z-tests",
    "Distribution fitting",
]

DIAGNOSTIC_PAGES = [
    "Normality tests",
    "Central limit theorem",
]


PAGE_ICONS = {
    "Import dataset": "⬆️",
    "Column selection": "▣",
    "Dataset overview": "📋",
    "Descriptive statistics": "▤",
    "Visualizations": "〽️",
    "T-tests": "𝑡",
    "Non-parametric tests": "↕️",
    "ANOVA": "A",
    "Chi-square tests": "χ²",
    "Z-tests": "𝑧",
    "Distribution fitting": "⚙️",
    "Normality tests": "〰️",
    "Central limit theorem": "↗️",
}


def inject_global_css():
    """
    Injects a polished Midnight Violet dashboard theme.
    """

    st.markdown(
        f"""
        <style>
        /* --------------------------------------------------
           Global app shell
        -------------------------------------------------- */

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(124, 92, 255, 0.12), transparent 32%),
                radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.10), transparent 30%),
                {THEME["bg"]};
            color: {THEME["text"]};
        }}

        .block-container {{
            padding-top: 1.15rem;
            padding-bottom: 2.4rem;
            max-width: 1240px;
        }}

        header[data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);
        }}

        h1, h2, h3, h4 {{
            color: {THEME["text"]};
            letter-spacing: -0.025em;
        }}

        label, p, span, div {{
            color: inherit;
        }}

        hr {{
            border-color: {THEME["border_soft"]};
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }}

        code {{
            background: rgba(124, 92, 255, 0.15) !important;
            color: {THEME["secondary"]} !important;
            border-radius: 8px !important;
            padding: 2px 6px !important;
        }}

        /* --------------------------------------------------
           Sidebar shell
        -------------------------------------------------- */

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, {THEME["sidebar"]}, {THEME["sidebar_2"]});
            border-right: 1px solid {THEME["border_soft"]};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.2rem;
            padding-left: 0.95rem;
            padding-right: 0.95rem;
        }}

        .app-logo-box {{
            width: 44px;
            height: 44px;
            border-radius: 15px;
            background:
                radial-gradient(circle at 30% 20%, rgba(255,255,255,0.50), transparent 22%),
                linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]});
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            font-weight: 950;
            color: white;
            box-shadow:
                0 12px 30px rgba(124, 92, 255, 0.35),
                inset 0 1px 0 rgba(255,255,255,0.28);
        }}

        .sidebar-title {{
            font-size: 20px;
            font-weight: 900;
            color: {THEME["text"]};
            margin-bottom: 0px;
            line-height: 1.05;
            letter-spacing: -0.035em;
        }}

        .sidebar-subtitle {{
            font-size: 12px;
            color: {THEME["muted"]};
            margin-top: 4px;
            line-height: 1.35;
        }}

        .sidebar-section {{
            font-size: 10.5px;
            color: {THEME["subtle"]};
            font-weight: 900;
            margin-top: 19px;
            margin-bottom: 9px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }}

        .nav-hint {{
            color: {THEME["muted"]};
            font-size: 12px;
            line-height: 1.5;
            background: rgba(255,255,255,0.035);
            border: 1px solid {THEME["border_soft"]};
            border-radius: 14px;
            padding: 12px;
        }}

        .footer-note {{
            color: {THEME["muted"]};
            font-size: 12px;
            margin-top: 8px;
        }}

        /* --------------------------------------------------
           Custom active navigation card
        -------------------------------------------------- */

        .nav-active-card {{
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            min-height: 42px;
            margin: 5px 0 7px 0;
            padding: 10px 13px;
            border-radius: 15px;
            color: #FFFFFF;
            font-size: 14px;
            font-weight: 850;
            letter-spacing: -0.01em;
            background:
                linear-gradient(135deg, rgba(124, 92, 255, 0.42), rgba(167, 139, 250, 0.18)),
                rgba(255,255,255,0.035);
            border: 1px solid rgba(167, 139, 250, 0.65);
            box-shadow:
                0 0 0 1px rgba(124, 92, 255, 0.16),
                0 14px 32px rgba(124, 92, 255, 0.23),
                inset 0 1px 0 rgba(255,255,255,0.15);
            overflow: hidden;
        }}

        .nav-active-card::before {{
            content: "";
            position: absolute;
            top: 8px;
            bottom: 8px;
            left: 0;
            width: 4px;
            border-radius: 999px;
            background: linear-gradient(180deg, #FFFFFF, {THEME["secondary"]});
            box-shadow: 0 0 14px rgba(167, 139, 250, 0.95);
        }}

        .nav-active-card::after {{
            content: "";
            position: absolute;
            inset: -50%;
            background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 34%);
            transform: translateX(-35%);
            pointer-events: none;
        }}

        .nav-active-icon {{
            position: relative;
            z-index: 1;
            width: 25px;
            height: 25px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.12);
            color: #FFFFFF;
            font-size: 14px;
            font-weight: 950;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.13);
        }}

        .nav-active-label {{
            position: relative;
            z-index: 1;
            color: #FFFFFF;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* --------------------------------------------------
           Sidebar buttons / inactive nav items
        -------------------------------------------------- */

        section[data-testid="stSidebar"] .stButton > button {{
            min-height: 42px;
            width: 100%;
            border-radius: 15px;
            border: 1px solid transparent;
            background: transparent;
            color: {THEME["muted"]};
            font-weight: 760;
            font-size: 14px;
            text-align: left;
            justify-content: flex-start;
            padding: 9px 13px;
            transition: all 0.18s ease;
            box-shadow: none;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(124, 92, 255, 0.13);
            color: {THEME["text"]};
            border-color: rgba(124, 92, 255, 0.28);
            transform: translateX(3px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20);
        }}

        section[data-testid="stSidebar"] .stButton > button:disabled {{
            opacity: 0.42;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}

        /* --------------------------------------------------
           General buttons
        -------------------------------------------------- */

        .stButton > button {{
            border-radius: 13px;
            border: 1px solid {THEME["border"]};
            background-color: {THEME["card"]};
            color: {THEME["text"]};
            font-weight: 760;
            transition: all 0.16s ease-in-out;
            min-height: 40px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }}

        .stButton > button:hover {{
            border-color: {THEME["primary"]};
            color: {THEME["text"]};
            background-color: {THEME["card_hover"]};
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(0,0,0,0.20);
        }}

        .stButton > button:disabled {{
            opacity: 0.45;
            cursor: not-allowed;
            transform: none;
        }}

        .stButton > button[kind="primary"] {{
            background:
                linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]});
            border: 1px solid rgba(167, 139, 250, 0.80);
            color: white;
            box-shadow:
                0 12px 28px rgba(124, 92, 255, 0.25),
                inset 0 1px 0 rgba(255,255,255,0.20);
        }}

        .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.06);
            box-shadow:
                0 16px 34px rgba(124, 92, 255, 0.34),
                inset 0 1px 0 rgba(255,255,255,0.25);
        }}

        /* --------------------------------------------------
           Cards
        -------------------------------------------------- */

        .stat-card {{
            position: relative;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)),
                {THEME["card"]};
            border: 1px solid {THEME["border_soft"]};
            border-radius: 18px;
            padding: 18px;
            min-height: 108px;
            box-shadow:
                0 16px 32px rgba(0,0,0,0.18),
                inset 0 1px 0 rgba(255,255,255,0.045);
            transition: all 0.18s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(124, 92, 255, 0.35);
            box-shadow:
                0 20px 40px rgba(0,0,0,0.24),
                0 0 0 1px rgba(124, 92, 255, 0.10);
        }}

        .stat-card-title {{
            font-size: 13px;
            color: {THEME["muted"]};
            margin-bottom: 8px;
            font-weight: 700;
        }}

        .stat-card-value {{
            font-size: 29px;
            color: {THEME["text"]};
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }}

        .stat-card-subtitle {{
            font-size: 12px;
            color: {THEME["muted"]};
            margin-top: 7px;
            line-height: 1.4;
        }}

        .info-card {{
            background:
                linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)),
                {THEME["card"]};
            border: 1px solid {THEME["border_soft"]};
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 16px 32px rgba(0,0,0,0.16);
        }}

        /* --------------------------------------------------
           Status pills
        -------------------------------------------------- */

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 7px 13px;
            background: rgba(255,255,255,0.035);
            border: 1px solid {THEME["border_soft"]};
            color: {THEME["muted"]};
            font-size: 13px;
            font-weight: 750;
            box-shadow: 0 10px 24px rgba(0,0,0,0.14);
        }}

        .status-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }}

        .dot-success {{
            background: {THEME["success"]};
            box-shadow: 0 0 12px {THEME["success"]};
        }}

        .dot-warning {{
            background: {THEME["warning"]};
            box-shadow: 0 0 12px {THEME["warning"]};
        }}

        .dot-error {{
            background: {THEME["error"]};
            box-shadow: 0 0 12px {THEME["error"]};
        }}

        /* --------------------------------------------------
           Inputs
        -------------------------------------------------- */

        div[data-baseweb="select"] > div {{
            background-color: {THEME["card"]};
            border-color: {THEME["border"]};
            border-radius: 13px;
        }}

        .stTextInput input,
        .stNumberInput input {{
            background-color: {THEME["card"]};
            border: 1px solid {THEME["border"]};
            color: {THEME["text"]};
            border-radius: 13px;
        }}

        .stTextInput input:focus,
        .stNumberInput input:focus {{
            border-color: {THEME["primary"]};
            box-shadow: 0 0 0 1px {THEME["primary_glow"]};
        }}

        .stSlider {{
            color: {THEME["primary"]};
        }}

        /* --------------------------------------------------
           Dataframes / expanders / alerts
        -------------------------------------------------- */

        div[data-testid="stDataFrame"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid {THEME["border_soft"]};
            box-shadow: 0 14px 28px rgba(0,0,0,0.14);
        }}

        details {{
            background:
                linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
                {THEME["card"]} !important;
            border: 1px solid {THEME["border_soft"]} !important;
            border-radius: 16px !important;
            box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 15px;
            border: 1px solid {THEME["border_soft"]};
            box-shadow: 0 12px 26px rgba(0,0,0,0.12);
        }}

        section[data-testid="stFileUploaderDropzone"] {{
            background:
                linear-gradient(180deg, rgba(124, 92, 255, 0.10), rgba(255,255,255,0.02)),
                {THEME["card"]};
            border: 1px dashed rgba(167, 139, 250, 0.45);
            border-radius: 20px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }}

        /* --------------------------------------------------
           Tabs — professional active highlight
        -------------------------------------------------- */

        .stTabs [data-baseweb="tab-list"] {{
            gap: 9px;
            background: rgba(255,255,255,0.025);
            padding: 7px;
            border: 1px solid {THEME["border_soft"]};
            border-radius: 999px;
            width: fit-content;
            max-width: 100%;
            overflow-x: auto;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 999px;
            border: 1px solid transparent;
            padding: 9px 18px;
            color: {THEME["muted"]};
            font-weight: 800;
            transition: all 0.16s ease;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background: rgba(124, 92, 255, 0.13);
            color: {THEME["text"]};
        }}

        .stTabs [aria-selected="true"] {{
            background:
                linear-gradient(135deg, rgba(124, 92, 255, 0.55), rgba(167, 139, 250, 0.28));
            border-color: rgba(167, 139, 250, 0.65);
            color: #FFFFFF;
            box-shadow:
                0 10px 24px rgba(124, 92, 255, 0.20),
                inset 0 1px 0 rgba(255,255,255,0.16);
        }}

        .stTabs [aria-selected="true"] p {{
            color: #FFFFFF !important;
        }}

        /* --------------------------------------------------
           Radio / checkbox little polish
        -------------------------------------------------- */

        div[role="radiogroup"] label,
        .stCheckbox label {{
            background: rgba(255,255,255,0.018);
            border-radius: 12px;
            padding: 4px 6px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_page_key(page_name):
    """
    Creates safe Streamlit widget keys from page names.
    """

    return (
        page_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("²", "2")
        .replace("/", "_")
    )


def _nav_button(page_name, disabled=False):
    """
    Sidebar navigation item.

    Active page is rendered as a custom highlighted card.
    Inactive pages are rendered as Streamlit buttons.
    """

    current_page = st.session_state.get("current_page", "Import dataset")
    is_active = current_page == page_name

    icon = PAGE_ICONS.get(page_name, "")

    if is_active:
        st.markdown(
            f"""
            <div class="nav-active-card">
                <span class="nav-active-icon">{icon}</span>
                <span class="nav-active-label">{page_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    button_label = f"{icon}  {page_name}"

    if st.button(
        button_label,
        key=f"nav_{_safe_page_key(page_name)}",
        disabled=disabled,
        use_container_width=True,
    ):
        st.session_state.current_page = page_name
        st.rerun()


def render_sidebar():
    """
    Renders sidebar navigation.
    Returns selected page name.
    """

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Import dataset"

    has_dataset = st.session_state.get("df") is not None
    has_selected_dataset = st.session_state.get("selected_df") is not None

    locked_pages = ANALYSIS_PAGES + DIAGNOSTIC_PAGES

    if st.session_state.current_page in locked_pages and not has_selected_dataset:
        st.session_state.current_page = "Column selection" if has_dataset else "Import dataset"

    with st.sidebar:
        col_logo, col_title = st.columns([0.25, 0.75])

        with col_logo:
            st.markdown('<div class="app-logo-box">Σ</div>', unsafe_allow_html=True)

        with col_title:
            st.markdown('<div class="sidebar-title">SandeepStician</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-subtitle">Smart statistics toolkit</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)

        for page in DATA_PAGES:
            if page == "Column selection":
                _nav_button(page, disabled=not has_dataset)
            else:
                _nav_button(page, disabled=False)

        st.markdown('<div class="sidebar-section">Analysis</div>', unsafe_allow_html=True)

        if has_selected_dataset:
            for page in ANALYSIS_PAGES:
                _nav_button(page, disabled=False)
        else:
            st.markdown(
                '<div class="nav-hint">Upload a dataset and select columns to unlock analysis pages.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-section">Diagnostics</div>', unsafe_allow_html=True)

        if has_selected_dataset:
            for page in DIAGNOSTIC_PAGES:
                _nav_button(page, disabled=False)
        else:
            st.markdown(
                '<div class="nav-hint">Diagnostics unlock after column selection.</div>',
                unsafe_allow_html=True,
            )

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

        if st.button("Reset project", use_container_width=True, key="reset_project_sidebar"):
            reset_dataset_state()
            st.session_state.current_page = "Import dataset"
            st.rerun()

    return st.session_state.current_page


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
        "uploaded_file_signature",

        "dist_fit_results",
        "dist_fit_data",
        "dist_fit_column_name",
        "dist_fit_bins_used",
        "dist_fit_signature",

        "clt_data",
        "clt_sample_means",
        "clt_column_name",
        "clt_sample_size_used",
        "clt_number_of_samples_used",
        "clt_bins_used",
        "clt_signature",
        "clt_sample_size_results",
        "clt_comparison_signature",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def render_top_bar(page_title):
    """
    Renders a compact top bar for every page.
    """

    has_dataset = st.session_state.get("df") is not None
    has_selected_dataset = st.session_state.get("selected_df") is not None

    left, right = st.columns([0.62, 0.38])

    with left:
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                margin-bottom:2px;
            ">
                <div style="
                    width:8px;
                    height:28px;
                    border-radius:999px;
                    background:linear-gradient(180deg, {THEME["primary"]}, {THEME["secondary"]});
                    box-shadow:0 0 18px rgba(124,92,255,0.55);
                "></div>
                <div>
                    <div style="
                        font-size:27px;
                        font-weight:900;
                        color:{THEME["text"]};
                        letter-spacing:-0.04em;
                        line-height:1.05;
                    ">
                        {page_title}
                    </div>
                    <div style="
                        font-size:13px;
                        color:{THEME["muted"]};
                        margin-top:3px;
                    ">
                        SandeepStician statistical workspace
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if has_selected_dataset:
            rows = st.session_state.selected_df.shape[0]
            cols = st.session_state.selected_df.shape[1]
            st.markdown(
                f"""
                <div style="text-align:right; margin-top:8px;">
                    <span class="status-pill">
                        <span class="status-dot dot-success"></span>
                        {rows:,} rows · {cols} columns selected
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif has_dataset:
            rows = st.session_state.df.shape[0]
            cols = st.session_state.df.shape[1]
            st.markdown(
                f"""
                <div style="text-align:right; margin-top:8px;">
                    <span class="status-pill">
                        <span class="status-dot dot-warning"></span>
                        {rows:,} rows · {cols} columns detected
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div style="text-align:right; margin-top:8px;">
                    <span class="status-pill">
                        <span class="status-dot dot-warning"></span>
                        No file loaded
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
            <h4 style="margin-top:0; margin-bottom:10px;">{title}</h4>
            <ul style="margin-bottom:0; color:{THEME["muted"]}; line-height:1.55;">
                {bullet_lines}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_status_card(title, status, message, status_type="info"):
    """
    Reusable status card for assumptions/results.

    status_type options:
    - success
    - warning
    - error
    - info
    """

    color_map = {
        "success": THEME["success"],
        "warning": THEME["warning"],
        "error": THEME["error"],
        "info": THEME["info"],
    }

    color = color_map.get(status_type, THEME["info"])

    st.markdown(
        f"""
        <div style="
            background:
                linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)),
                {THEME["card"]};
            border:1px solid {THEME["border_soft"]};
            border-left:5px solid {color};
            border-radius:18px;
            padding:17px;
            margin-bottom:12px;
            box-shadow:0 16px 32px rgba(0,0,0,0.16);
        ">
            <div style="font-size:13px; color:{THEME["muted"]}; font-weight:800;">
                {title}
            </div>
            <div style="font-size:21px; color:{color}; font-weight:900; margin-top:4px; letter-spacing:-0.03em;">
                {status}
            </div>
            <div style="font-size:13px; color:{THEME["muted"]}; margin-top:7px; line-height:1.55;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_column_card(column, dtype_text, unique_count, missing_count, badge="Column", badge_type="info"):
    """
    Reusable selected column display card.
    """

    color_map = {
        "success": THEME["success"],
        "warning": THEME["warning"],
        "error": THEME["error"],
        "info": THEME["info"],
        "primary": THEME["primary"],
        "secondary": THEME["secondary"],
    }

    badge_color = color_map.get(badge_type, THEME["info"])

    return f"""
    <div style="
        border: 1px solid {THEME["border_soft"]};
        background:
            linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)),
            {THEME["card"]};
        border-radius: 18px;
        padding: 16px;
        min-height: 130px;
        box-shadow: 0 14px 28px rgba(0,0,0,0.14);
    ">
        <div style="font-weight:900; font-size:16px; color:{THEME["text"]}; margin-bottom:8px;">
            {column}
        </div>
        <div style="
            display:inline-block;
            padding:5px 10px;
            border-radius:999px;
            background:{badge_color}22;
            color:{badge_color};
            font-size:12px;
            font-weight:850;
            margin-bottom:8px;
            border:1px solid {badge_color}44;
        ">
            {badge}
        </div>
        <div style="font-size:13px; color:{THEME["muted"]}; margin-top:8px;">
            Type: <b>{dtype_text}</b>
        </div>
        <div style="font-size:13px; color:{THEME["muted"]};">
            Unique values: <b>{unique_count}</b>
        </div>
        <div style="font-size:13px; color:{THEME["muted"]};">
            Missing values: <b>{missing_count}</b>
        </div>
    </div>
    """


def render_card_grid(card_html_list, min_width=220):
    """
    Renders a responsive grid of HTML cards.
    """

    cards = "".join(card_html_list)

    st.markdown(
        f"""
        <div style="
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax({min_width}px, 1fr));
            gap: 15px;
            margin-top: 12px;
            margin-bottom: 30px;
        ">
            {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_locked_message():
    """
    Message shown when user tries analysis before selecting columns.
    """

    st.warning("Please upload a dataset and select columns before using this analysis page.")