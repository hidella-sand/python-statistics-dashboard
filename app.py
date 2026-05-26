import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_dataset,
    get_basic_dataset_info,
    get_column_summary,
    get_selected_dataframe,
    get_numerical_columns,
    get_categorical_columns
)


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Data Methods Lab",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------------------
# Small CSS styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
        color: inherit;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #d1d5db;
        background-color: #f9fafb;
        text-align: center;
        min-height: 95px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #111827 !important;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 14px;
        color: #4b5563 !important;
        margin-top: 6px;
    }

    .column-card-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .small-muted {
        color: #9ca3af;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------

if "step" not in st.session_state:
    st.session_state.step = "upload"

if "df" not in st.session_state:
    st.session_state.df = None

if "selected_df" not in st.session_state:
    st.session_state.selected_df = None

if "column_summary" not in st.session_state:
    st.session_state.column_summary = None

if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


def reset_project():
    st.session_state.step = "upload"
    st.session_state.df = None
    st.session_state.selected_df = None
    st.session_state.column_summary = None
    st.session_state.selected_columns = []
    st.session_state.uploaded_file_name = None


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown('<div class="main-title">📊 Data Methods Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a dataset, select useful columns, and prepare it for statistical analysis.</div>',
    unsafe_allow_html=True
)

st.divider()


# ------------------------------------------------------------
# STEP 1: Upload dataset
# ------------------------------------------------------------

if st.session_state.step == "upload":

    st.subheader("Step 1: Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel XLSX"
    )

    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)

            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.column_summary = get_column_summary(df)

            st.success("Dataset loaded successfully.")

            dataset_info = get_basic_dataset_info(df)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{dataset_info["rows"]}</div>
                        <div class="metric-label">Rows</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{dataset_info["columns"]}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{dataset_info["total_missing_values"]}</div>
                        <div class="metric-label">Missing Values</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{dataset_info["duplicate_rows"]}</div>
                        <div class="metric-label">Duplicate Rows</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("OK, Show Columns", type="primary"):
                st.session_state.step = "select_columns"
                st.rerun()

        except Exception as error:
            st.error(error)


# ------------------------------------------------------------
# STEP 2: Select columns using checkboxes
# ------------------------------------------------------------

elif st.session_state.step == "select_columns":

    df = st.session_state.df
    column_summary = st.session_state.column_summary

    st.subheader("Step 2: Select Columns for Analysis")

    st.info(
        "Uncheck columns that are not useful for analysis, such as ID numbers, names, card numbers, emails, or other identifier columns."
    )

    with st.expander("View Column Summary Table", expanded=False):
        st.dataframe(column_summary, use_container_width=True)

    st.write("### Column Selection")

    selected_columns = []

    with st.form("column_selection_form"):

        columns_per_row = 3
        all_columns = df.columns.tolist()

        for i in range(0, len(all_columns), columns_per_row):
            row_columns = st.columns(columns_per_row)

            for j, column_name in enumerate(all_columns[i:i + columns_per_row]):
                summary_row = column_summary[column_summary["Column"] == column_name].iloc[0]

                suggestion = summary_row["Suggestion"]
                detected_type = summary_row["Detected Type"]
                unique_values = summary_row["Unique Values"]
                missing_percentage = summary_row["Missing %"]
                reason = summary_row["Reason"]

                default_checked = True
                if suggestion == "Remove":
                    default_checked = False

                with row_columns[j]:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div class="column-card-title">{column_name}</div>
                            <div class="small-muted">Type: {detected_type}</div>
                            <div class="small-muted">Unique values: {unique_values}</div>
                            <div class="small-muted">Missing: {missing_percentage}%</div>
                            <div class="small-muted">Suggestion: <b>{suggestion}</b></div>
                            """,
                            unsafe_allow_html=True
                        )

                        checked = st.checkbox(
                            "Use this column",
                            value=default_checked,
                            key=f"select_{column_name}"
                        )

                        st.caption(reason)

                        if checked:
                            selected_columns.append(column_name)

        submitted = st.form_submit_button("Next: Continue with Selected Columns", type="primary")

    if submitted:
        final_selected_columns = []

        for column_name in df.columns:
            checkbox_key = f"select_{column_name}"

            if st.session_state.get(checkbox_key):
                final_selected_columns.append(column_name)

        if len(final_selected_columns) == 0:
            st.warning("Please select at least one column before continuing.")
        else:
            st.session_state.selected_columns = final_selected_columns
            st.session_state.selected_df = get_selected_dataframe(df, final_selected_columns)
            st.session_state.step = "selected_preview"
            st.rerun()

    st.button("Back to Upload", on_click=reset_project)


# ------------------------------------------------------------
# STEP 3: Show selected data nicely
# ------------------------------------------------------------

elif st.session_state.step == "selected_preview":

    selected_df = st.session_state.selected_df

    st.subheader("Step 3: Selected Dataset Overview")

    st.success("These are the columns that will be used for future analysis.")

    numerical_columns = get_numerical_columns(selected_df)
    categorical_columns = get_categorical_columns(selected_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{selected_df.shape[0]}</div>
                <div class="metric-label">Rows</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{selected_df.shape[1]}</div>
                <div class="metric-label">Selected Columns</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{len(numerical_columns)}</div>
                <div class="metric-label">Numerical Columns</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{len(categorical_columns)}</div>
                <div class="metric-label">Categorical Columns</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("### Selected Columns")
    st.write(st.session_state.selected_columns)

    st.write("### Selected Dataset Preview")
    st.dataframe(selected_df.head(20), use_container_width=True)

    st.write("### Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": selected_df.columns,
        "Data Type": selected_df.dtypes.astype(str).values,
        "Missing Values": selected_df.isna().sum().values,
        "Unique Values": selected_df.nunique(dropna=True).values
    })

    st.dataframe(dtype_df, use_container_width=True)

    col_back, col_continue = st.columns([1, 2])

    with col_back:
        if st.button("Back to Column Selection"):
            st.session_state.step = "select_columns"
            st.rerun()

    with col_continue:
        st.button("Continue to Analysis Coming Next", type="primary")