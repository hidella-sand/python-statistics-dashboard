import streamlit as st


# -----------------------------------------------------------------------------
# SandeepStician UI Theme
# -----------------------------------------------------------------------------
# Palette requested by user:
# Blue:       #56B4E9
# Vermillion: #D55E00
# Green:      #009E73
# Orange:     #E69F00
#
# Theme goal:
# Clean, calm, professional light dashboard for statistical analysis.
# -----------------------------------------------------------------------------

THEME = {
    # App shell
    "bg": "#F6F8FB",
    "bg_2": "#EEF3F8",
    "sidebar": "#FFFFFF",
    "sidebar_2": "#F8FAFC",

    # Surfaces
    "card": "#FFFFFF",
    "card_hover": "#F1F5F9",
    "card_soft": "#F8FAFC",
    "border": "#D9E2EC",
    "border_soft": "rgba(15, 23, 42, 0.10)",

    # Text
    "text": "#1F2937",
    "heading": "#111827",
    "muted": "#64748B",
    "subtle": "#94A3B8",

    # Requested color palette
    "primary": "#56B4E9",      # sky blue
    "secondary": "#009E73",    # green
    "accent": "#E69F00",       # orange
    "danger": "#D55E00",       # vermillion

    # Semantic aliases
    "success": "#009E73",
    "warning": "#E69F00",
    "error": "#D55E00",
    "info": "#56B4E9",

    # Soft fills / glows
    "primary_soft": "rgba(86, 180, 233, 0.14)",
    "secondary_soft": "rgba(0, 158, 115, 0.12)",
    "accent_soft": "rgba(230, 159, 0, 0.14)",
    "danger_soft": "rgba(213, 94, 0, 0.12)",
    "shadow": "rgba(15, 23, 42, 0.08)",
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
    Injects a soft professional light dashboard theme.
    """

    st.markdown(
        f"""
        <style>
        /* --------------------------------------------------
           Global app shell
        -------------------------------------------------- */

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(86, 180, 233, 0.16), transparent 32%),
                radial-gradient(circle at bottom right, rgba(0, 158, 115, 0.10), transparent 30%),
                linear-gradient(180deg, {THEME["bg"]}, {THEME["bg_2"]});
            color: {THEME["text"]};
        }}

        .block-container {{
            padding-top: 1.15rem;
            padding-bottom: 2.4rem;
            max-width: 1240px;
        }}

        header[data-testid="stHeader"] {{
            background-color: rgba(255, 255, 255, 0);
        }}

        h1, h2, h3, h4 {{
            color: {THEME["heading"]};
            letter-spacing: -0.025em;
        }}

        /* Base text color.
           Do NOT force every div to inherit color in a light theme, because
           Streamlit widgets can accidentally keep invisible white text. */
        label, p, span {{
            color: {THEME["text"]};
        }}

        div {{
            color: inherit;
        }}

        hr {{
            border-color: {THEME["border_soft"]};
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }}

        code {{
            background: {THEME["primary_soft"]} !important;
            color: #075985 !important;
            border-radius: 8px !important;
            padding: 2px 6px !important;
            border: 1px solid rgba(86, 180, 233, 0.20) !important;
        }}

        /* --------------------------------------------------
           Sidebar shell
        -------------------------------------------------- */

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, {THEME["sidebar"]}, {THEME["sidebar_2"]});
            border-right: 1px solid {THEME["border"]};
            box-shadow: 10px 0 30px rgba(15, 23, 42, 0.04);
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
                radial-gradient(circle at 30% 20%, rgba(255,255,255,0.70), transparent 24%),
                linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]});
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            font-weight: 950;
            color: white;
            box-shadow:
                0 12px 28px rgba(86, 180, 233, 0.24),
                inset 0 1px 0 rgba(255,255,255,0.55);
        }}

        .sidebar-title {{
            font-size: 20px;
            font-weight: 900;
            color: {THEME["heading"]};
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
            background: {THEME["card_soft"]};
            border: 1px solid {THEME["border"]};
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
            color: {THEME["heading"]};
            font-size: 14px;
            font-weight: 850;
            letter-spacing: -0.01em;
            background:
                linear-gradient(135deg, rgba(86, 180, 233, 0.26), rgba(0, 158, 115, 0.12)),
                #FFFFFF;
            border: 1px solid rgba(86, 180, 233, 0.48);
            box-shadow:
                0 0 0 1px rgba(86, 180, 233, 0.08),
                0 14px 32px rgba(15, 23, 42, 0.08),
                inset 0 1px 0 rgba(255,255,255,0.80);
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
            background: linear-gradient(180deg, {THEME["primary"]}, {THEME["secondary"]});
            box-shadow: 0 0 14px rgba(86, 180, 233, 0.55);
        }}

        .nav-active-card::after {{
            content: "";
            position: absolute;
            inset: -50%;
            background: radial-gradient(circle, rgba(255,255,255,0.65), transparent 35%);
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
            background: rgba(255,255,255,0.82);
            color: {THEME["heading"]};
            font-size: 14px;
            font-weight: 950;
            border: 1px solid rgba(86, 180, 233, 0.24);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.90);
        }}

        .nav-active-label {{
            position: relative;
            z-index: 1;
            color: {THEME["heading"]};
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
            background: rgba(86, 180, 233, 0.12);
            color: {THEME["heading"]};
            border-color: rgba(86, 180, 233, 0.28);
            transform: translateX(3px);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
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
            color: {THEME["heading"]};
            font-weight: 760;
            transition: all 0.16s ease-in-out;
            min-height: 40px;
            box-shadow: 0 8px 20px rgba(15,23,42,0.05);
        }}

        .stButton > button:hover {{
            border-color: {THEME["primary"]};
            color: {THEME["heading"]};
            background-color: {THEME["card_hover"]};
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(15,23,42,0.09);
        }}

        .stButton > button:disabled {{
            opacity: 0.45;
            cursor: not-allowed;
            transform: none;
        }}

        .stButton > button[kind="primary"] {{
            background:
                linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]});
            border: 1px solid rgba(86, 180, 233, 0.70);
            color: white;
            box-shadow:
                0 12px 28px rgba(86, 180, 233, 0.22),
                inset 0 1px 0 rgba(255,255,255,0.35);
        }}

        .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.03);
            box-shadow:
                0 16px 34px rgba(86, 180, 233, 0.28),
                inset 0 1px 0 rgba(255,255,255,0.42);
        }}

        /* --------------------------------------------------
           Cards
        -------------------------------------------------- */

        .stat-card {{
            position: relative;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.92)),
                {THEME["card"]};
            border: 1px solid {THEME["border"]};
            border-radius: 18px;
            padding: 18px;
            min-height: 108px;
            box-shadow:
                0 16px 32px rgba(15,23,42,0.06),
                inset 0 1px 0 rgba(255,255,255,0.85);
            transition: all 0.18s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(86, 180, 233, 0.42);
            box-shadow:
                0 20px 40px rgba(15,23,42,0.09),
                0 0 0 1px rgba(86, 180, 233, 0.08);
        }}

        .stat-card-title {{
            font-size: 13px;
            color: {THEME["muted"]};
            margin-bottom: 8px;
            font-weight: 700;
        }}

        .stat-card-value {{
            font-size: 29px;
            color: {THEME["heading"]};
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
                linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.92)),
                {THEME["card"]};
            border: 1px solid {THEME["border"]};
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 16px 32px rgba(15,23,42,0.05);
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
            background: rgba(255,255,255,0.78);
            border: 1px solid {THEME["border"]};
            color: {THEME["muted"]};
            font-size: 13px;
            font-weight: 750;
            box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        }}

        .status-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }}

        .dot-success {{
            background: {THEME["success"]};
            box-shadow: 0 0 10px rgba(0,158,115,0.45);
        }}

        .dot-warning {{
            background: {THEME["warning"]};
            box-shadow: 0 0 10px rgba(230,159,0,0.42);
        }}

        .dot-error {{
            background: {THEME["error"]};
            box-shadow: 0 0 10px rgba(213,94,0,0.42);
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
            box-shadow: 0 0 0 1px rgba(86, 180, 233, 0.28);
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
            border: 1px solid {THEME["border"]};
            box-shadow: 0 14px 28px rgba(15,23,42,0.05);
        }}

        details {{
            background:
                linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.92)),
                {THEME["card"]} !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 16px !important;
            box-shadow: 0 12px 28px rgba(15,23,42,0.05);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 15px;
            border: 1px solid {THEME["border"]};
            box-shadow: 0 12px 26px rgba(15,23,42,0.05);
        }}

        section[data-testid="stFileUploaderDropzone"] {{
            background:
                linear-gradient(180deg, rgba(86, 180, 233, 0.08), rgba(255,255,255,0.75)),
                {THEME["card"]};
            border: 1px dashed rgba(86, 180, 233, 0.55);
            border-radius: 20px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.80);
        }}

        /* --------------------------------------------------
           Tabs — soft active highlight
        -------------------------------------------------- */

        .stTabs [data-baseweb="tab-list"] {{
            gap: 9px;
            background: rgba(255,255,255,0.70);
            padding: 7px;
            border: 1px solid {THEME["border"]};
            border-radius: 999px;
            width: fit-content;
            max-width: 100%;
            overflow-x: auto;
            box-shadow: 0 10px 24px rgba(15,23,42,0.04);
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
            background: rgba(86, 180, 233, 0.12);
            color: {THEME["heading"]};
        }}

        .stTabs [aria-selected="true"] {{
            background:
                linear-gradient(135deg, rgba(86, 180, 233, 0.26), rgba(0, 158, 115, 0.12));
            border-color: rgba(86, 180, 233, 0.48);
            color: {THEME["heading"]};
            box-shadow:
                0 10px 24px rgba(86, 180, 233, 0.12),
                inset 0 1px 0 rgba(255,255,255,0.85);
        }}

        .stTabs [aria-selected="true"] p {{
            color: {THEME["heading"]} !important;
        }}

        /* --------------------------------------------------
           Radio / checkbox polish
        -------------------------------------------------- */

        div[role="radiogroup"] label,
        .stCheckbox label {{
            background: rgba(255,255,255,0.48);
            border-radius: 12px;
            padding: 4px 6px;
        }}


        /* --------------------------------------------------
           Light theme widget text visibility fixes
           These overrides prevent old dark-theme white labels
           from appearing on the new light background.
        -------------------------------------------------- */

        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li {{
            color: {THEME["text"]} !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {THEME["heading"]} !important;
        }}

        /* Streamlit widget labels */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label *,
        .stSelectbox label,
        .stSelectbox label *,
        .stMultiSelect label,
        .stMultiSelect label *,
        .stRadio > label,
        .stRadio > label *,
        .stCheckbox label,
        .stCheckbox label *,
        .stSlider label,
        .stSlider label *,
        .stNumberInput label,
        .stNumberInput label *,
        .stTextInput label,
        .stTextInput label *,
        .stTextArea label,
        .stTextArea label * {{
            color: {THEME["text"]} !important;
            font-weight: 750 !important;
        }}

        /* Captions and helper text */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaptionContainer,
        .stCaptionContainer *,
        small {{
            color: {THEME["muted"]} !important;
        }}

        /* Selectbox / multiselect input text */
        div[data-baseweb="select"],
        div[data-baseweb="select"] *,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {{
            color: {THEME["text"]} !important;
        }}

        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 13px !important;
        }}

        /* Text and number inputs */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {{
            background-color: #FFFFFF !important;
            color: {THEME["text"]} !important;
            border: 1px solid {THEME["border"]} !important;
        }}

        .stTextInput input::placeholder,
        .stNumberInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color: {THEME["subtle"]} !important;
            opacity: 1 !important;
        }}

        /* Radio options */
        div[role="radiogroup"] label {{
            background: rgba(255,255,255,0.86) !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 13px !important;
            padding: 7px 11px !important;
            color: {THEME["text"]} !important;
        }}

        div[role="radiogroup"] label *,
        div[role="radiogroup"] p,
        div[role="radiogroup"] span {{
            color: {THEME["text"]} !important;
            font-weight: 700 !important;
        }}

        /* Checkbox options */
        .stCheckbox label {{
            background: rgba(255,255,255,0.86) !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 13px !important;
            padding: 6px 10px !important;
            color: {THEME["text"]} !important;
        }}

        .stCheckbox label *,
        .stCheckbox p,
        .stCheckbox span {{
            color: {THEME["text"]} !important;
        }}

        /* Slider text, tick labels, and current value */
        .stSlider,
        .stSlider *,
        [data-testid="stTickBar"],
        [data-testid="stTickBar"] * {{
            color: {THEME["text"]} !important;
        }}

        /* File uploader visible text */
        section[data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] * {{
            color: {THEME["text"]} !important;
        }}

        section[data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] button {{
            background: #FFFFFF !important;
            color: {THEME["heading"]} !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
        }}

        /* Alerts should not inherit white text */
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] *,
        .stAlert,
        .stAlert * {{
            color: {THEME["text"]} !important;
        }}

        /* Expander headers and content */
        details,
        details *,
        summary,
        summary * {{
            color: {THEME["text"]} !important;
        }}

        /* Dataframe wrapper text safety */
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {{
            color: inherit;
        }}

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {{
            color: {THEME["heading"]} !important;
        }}

        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] *,
        .stDownloadButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] * {{
            color: #FFFFFF !important;
        }}

        /* Top bar/status pill */
        .status-pill,
        .status-pill * {{
            color: {THEME["muted"]} !important;
        }}

        /* Sidebar keeps its own controlled colors */
        section[data-testid="stSidebar"] {{
            color: {THEME["text"]} !important;
        }}

        section[data-testid="stSidebar"] .sidebar-title {{
            color: {THEME["heading"]} !important;
        }}

        section[data-testid="stSidebar"] .sidebar-subtitle,
        section[data-testid="stSidebar"] .sidebar-section,
        section[data-testid="stSidebar"] .nav-hint {{
            color: {THEME["muted"]} !important;
        }}

        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stButton > button * {{
            color: {THEME["muted"]} !important;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover,
        section[data-testid="stSidebar"] .stButton > button:hover * {{
            color: {THEME["heading"]} !important;
        }}

        .nav-active-card,
        .nav-active-card *,
        .nav-active-icon,
        .nav-active-label {{
            color: {THEME["heading"]} !important;
        }}

        .app-logo-box,
        .app-logo-box * {{
            color: #FFFFFF !important;
        }}

        
        

        /* --------------------------------------------------
           Selectbox / multiselect dropdown menu fix
           Fixes the black popup menu caused by BaseWeb popover.
        -------------------------------------------------- */

        /* Popover wrapper */
        div[data-baseweb="popover"] {{
            z-index: 999999 !important;
        }}

        /* Main dropdown surface */
        div[data-baseweb="popover"] > div {{
            background: #FFFFFF !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 14px !important;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16) !important;
            overflow: hidden !important;
        }}

        /* Dropdown list container */
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] [role="listbox"] {{
            background: #FFFFFF !important;
            color: {THEME["text"]} !important;
        }}

        /* Individual dropdown options */
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"] {{
            background: #FFFFFF !important;
            color: {THEME["text"]} !important;
            font-weight: 650 !important;
        }}

        /* Text inside options */
        div[data-baseweb="popover"] li *,
        div[data-baseweb="popover"] [role="option"] *,
        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] p {{
            color: {THEME["text"]} !important;
        }}

        /* Prevent the old dark background from leaking into dropdown option rows */
        div[data-baseweb="popover"] div {{
            color: {THEME["text"]} !important;
        }}

        /* Hovered option */
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] [role="option"]:hover {{
            background: rgba(86, 180, 233, 0.14) !important;
            color: {THEME["heading"]} !important;
        }}

        div[data-baseweb="popover"] li:hover *,
        div[data-baseweb="popover"] [role="option"]:hover * {{
            color: {THEME["heading"]} !important;
        }}

        /* Selected option */
        div[data-baseweb="popover"] [aria-selected="true"],
        div[data-baseweb="popover"] [aria-selected="true"] * {{
            background: rgba(0, 158, 115, 0.14) !important;
            color: #064E3B !important;
            font-weight: 850 !important;
        }}

        /* Search box inside searchable dropdowns */
        div[data-baseweb="popover"] input {{
            background: #F8FAFC !important;
            color: {THEME["text"]} !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 10px !important;
        }}

        div[data-baseweb="popover"] input::placeholder {{
            color: {THEME["subtle"]} !important;
            opacity: 1 !important;
        }}

        /* Dropdown scrollbar */
        div[data-baseweb="popover"] ::-webkit-scrollbar {{
            width: 10px;
        }}

        div[data-baseweb="popover"] ::-webkit-scrollbar-track {{
            background: #F1F5F9;
        }}

        div[data-baseweb="popover"] ::-webkit-scrollbar-thumb {{
            background: #CBD5E1;
            border-radius: 999px;
        }}

        div[data-baseweb="popover"] ::-webkit-scrollbar-thumb:hover {{
            background: #94A3B8;
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
                    box-shadow:0 0 18px rgba(86,180,233,0.36);
                "></div>
                <div>
                    <div style="
                        font-size:27px;
                        font-weight:900;
                        color:{THEME["heading"]};
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

    soft_map = {
        "success": THEME["secondary_soft"],
        "warning": THEME["accent_soft"],
        "error": THEME["danger_soft"],
        "info": THEME["primary_soft"],
    }

    soft = soft_map.get(status_type, THEME["primary_soft"])

    st.markdown(
        f"""
        <div style="
            background:
                linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92)),
                {THEME["card"]};
            border:1px solid {THEME["border"]};
            border-left:5px solid {color};
            border-radius:18px;
            padding:17px;
            margin-bottom:12px;
            box-shadow:0 16px 32px rgba(15,23,42,0.05);
        ">
            <div style="font-size:13px; color:{THEME["muted"]}; font-weight:800;">
                {title}
            </div>
            <div style="
                display:inline-block;
                font-size:21px;
                color:{color};
                background:{soft};
                font-weight:900;
                margin-top:7px;
                letter-spacing:-0.03em;
                padding:5px 10px;
                border-radius:999px;
            ">
                {status}
            </div>
            <div style="font-size:13px; color:{THEME["muted"]}; margin-top:8px; line-height:1.55;">
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
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92)),
            {THEME["card"]};
        border-radius: 18px;
        padding: 16px;
        min-height: 130px;
        box-shadow: 0 14px 28px rgba(15,23,42,0.05);
    ">
        <div style="font-weight:900; font-size:16px; color:{THEME["heading"]}; margin-bottom:8px;">
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
