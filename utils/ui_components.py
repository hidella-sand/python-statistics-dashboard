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


DATA_PAGES = [
    "Import dataset",
    "Column selection",
]

ANALYSIS_PAGES = [
    "Dataset overview",
    "Descriptive statistics",
    "Visualizations",
    "T-tests",
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
    "ANOVA": "A",
    "Chi-square tests": "χ²",
    "Z-tests": "𝑧",
    "Distribution fitting": "⚙️",
    "Normality tests": "〰️",
    "Central limit theorem": "↗️",
}


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

        /* Main content container */
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1220px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {THEME["sidebar"]};
            border-right: 1px solid {THEME["border"]};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.1rem;
        }}

        /* Hide Streamlit default decoration spacing a little */
        header[data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);
        }}

        /* Headings */
        h1, h2, h3, h4 {{
            color: {THEME["text"]};
            letter-spacing: -0.02em;
        }}

        /* Text */
        label, p, span, div {{
            color: inherit;
        }}

        /* Horizontal divider */
        hr {{
            border-color: {THEME["border"]};
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

        .dot-error {{
            background: {THEME["error"]};
            box-shadow: 0 0 8px {THEME["error"]};
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
            color: white;
        }}

        .sidebar-title {{
            font-size: 19px;
            font-weight: 850;
            color: {THEME["text"]};
            margin-bottom: 0px;
            line-height: 1.1;
        }}

        .sidebar-subtitle {{
            font-size: 12px;
            color: {THEME["muted"]};
            margin-top: 2px;
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
            min-height: 40px;
        }}

        .stButton > button:hover {{
            border-color: {THEME["primary"]};
            color: {THEME["text"]};
            background-color: {THEME["card_hover"]};
        }}

        .stButton > button:disabled {{
            opacity: 0.45;
            cursor: not-allowed;
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

        /* Sliders */
        .stSlider {{
            color: {THEME["primary"]};
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
            font-weight: 700;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {THEME["primary_soft"]};
            border-color: {THEME["primary"]};
            color: {THEME["text"]};
        }}

        /* Alerts */
        div[data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid {THEME["border"]};
        }}

        /* File uploader */
        section[data-testid="stFileUploaderDropzone"] {{
            background-color: {THEME["card"]};
            border: 1px dashed {THEME["border"]};
            border-radius: 16px;
        }}

        /* Sidebar navigation active/inactive text markers */
        .nav-hint {{
            color: {THEME["muted"]};
            font-size: 12px;
            line-height: 1.5;
        }}

        .footer-note {{
            color: {THEME["muted"]};
            font-size: 12px;
            margin-top: 8px;
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
    Sidebar navigation button.
    Stores the selected page in st.session_state.current_page.
    """

    current_page = st.session_state.get("current_page", "Import dataset")
    is_active = current_page == page_name

    icon = PAGE_ICONS.get(page_name, "")

    if is_active:
        button_label = f"▸  {icon}  {page_name}"
    else:
        button_label = f"   {icon}  {page_name}"

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

    # Safety rule:
    # If selected dataset disappears while user is on an analysis page,
    # move user back to the correct setup page.
    if st.session_state.current_page in locked_pages and not has_selected_dataset:
        st.session_state.current_page = "Column selection" if has_dataset else "Import dataset"

    with st.sidebar:
        col_logo, col_title = st.columns([0.25, 0.75])

        with col_logo:
            st.markdown('<div class="app-logo-box">Σ</div>', unsafe_allow_html=True)

        with col_title:
            st.markdown('<div class="sidebar-title">StatKit</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-subtitle">Statistical toolkit</div>', unsafe_allow_html=True)

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

        if st.button("Reset project", use_container_width=True):
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

        # Distribution fitting session state
        "dist_fit_results",
        "dist_fit_data",
        "dist_fit_column_name",
        "dist_fit_bins_used",

        # CLT session state
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

    has_dataset = st.session_state.get("df") is not None
    has_selected_dataset = st.session_state.get("selected_df") is not None

    left, right = st.columns([0.62, 0.38])

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

        elif has_dataset:
            rows = st.session_state.df.shape[0]
            cols = st.session_state.df.shape[1]
            st.markdown(
                f"""
                <div style="text-align:right;">
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
                <div style="text-align:right;">
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
            <h4 style="margin-top:0;">{title}</h4>
            <ul style="margin-bottom:0; color:{THEME["muted"]};">
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
            background:{THEME["card"]};
            border:1px solid {THEME["border"]};
            border-left:5px solid {color};
            border-radius:14px;
            padding:16px;
            margin-bottom:10px;
        ">
            <div style="font-size:13px; color:{THEME["muted"]}; font-weight:700;">
                {title}
            </div>
            <div style="font-size:20px; color:{color}; font-weight:850; margin-top:4px;">
                {status}
            </div>
            <div style="font-size:13px; color:{THEME["muted"]}; margin-top:6px; line-height:1.5;">
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
        border: 1px solid {THEME["border"]};
        background: {THEME["card"]};
        border-radius: 14px;
        padding: 16px;
        min-height: 125px;
    ">
        <div style="font-weight:800; font-size:16px; color:{THEME["text"]}; margin-bottom:8px;">
            {column}
        </div>
        <div style="
            display:inline-block;
            padding:4px 9px;
            border-radius:999px;
            background:{badge_color}22;
            color:{badge_color};
            font-size:12px;
            font-weight:800;
            margin-bottom:8px;
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
            gap: 14px;
            margin-top: 12px;
            margin-bottom: 28px;
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