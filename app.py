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

from utils.descriptive_stats import (
    calculate_descriptive_statistics,
    create_descriptive_stats_table,
    interpret_descriptive_statistics
)

from utils.visualizations import (
    plot_histogram,
    plot_boxplot,
    plot_kde_pdf,
    plot_cdf,
    plot_qq,
    plot_pmf,
    is_discrete_numeric,
    prepare_numeric_data,
    get_plot_interpretation
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

    with col_continue:  # ← THIS LINE was de-indented in the original, causing the bug
        if st.button("Continue to Descriptive Statistics", type="primary"):
            st.session_state.step = "descriptive_stats"
            st.rerun()


# ------------------------------------------------------------
# STEP 4: Descriptive Statistics
# ------------------------------------------------------------

elif st.session_state.step == "descriptive_stats":
    selected_df = st.session_state.selected_df

    st.subheader("Step 4: Descriptive Statistics")

    st.info(
        "Select a numerical column to calculate mean, median, mode, variance, standard deviation, skewness, and kurtosis."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to Selected Dataset"):
            st.session_state.step = "selected_preview"
            st.rerun()

    else:
        selected_column = st.selectbox(
            "Choose a numerical column",
            numerical_columns
        )

        if selected_column:

            stats = calculate_descriptive_statistics(selected_df, selected_column)
            stats_table = create_descriptive_stats_table(stats)
            interpretations = interpret_descriptive_statistics(stats)

            st.write(f"### Descriptive Statistics for `{selected_column}`")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.dataframe(stats_table, use_container_width=True)

            with col2:
                st.write("### Quick Interpretation")

                for interpretation in interpretations:
                    st.write(f"- {interpretation}")

            st.write("### Selected Column Preview")

            preview_df = selected_df[[selected_column]].head(20)
            st.dataframe(preview_df, use_container_width=True)

            st.divider()

            col_back, col_next = st.columns([1, 2])

            with col_back:
                if st.button("Back to Selected Dataset"):
                    st.session_state.step = "selected_preview"
                    st.rerun()

            with col_next:
                if st.button("Continue to Visualizations", type="primary"):
                    st.session_state.step = "visualizations"
                    st.rerun()


# ------------------------------------------------------------
# STEP 5: Visualizations
# ------------------------------------------------------------

if st.session_state.step == "visualizations":

    selected_df = st.session_state.selected_df

    st.subheader("Step 5: Data Visualizations")

    st.info(
        "Select a numerical column to view histogram, boxplot, estimated PDF/KDE, CDF, Q-Q plot, and PMF if suitable."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to Descriptive Statistics"):
            st.session_state.step = "descriptive_stats"
            st.rerun()

    else:
        selected_column = st.selectbox(
            "Choose a numerical column for visualization",
            numerical_columns,
            key="visualization_column"
        )

        clean_data = prepare_numeric_data(selected_df, selected_column)

        st.write(f"### Visualizations for `{selected_column}`")

        plot_option = st.radio(
            "Choose visualization type",
            [
                "Histogram",
                "Boxplot",
                "Estimated PDF / KDE",
                "CDF",
                "Q-Q Plot",
                "PMF"
            ],
            horizontal=True
        )

        st.divider()

        plot_col, interpretation_col = st.columns([1.25, 1])

        fig = None

        with plot_col:
            st.write(f"#### {plot_option}")

            if plot_option == "Histogram":
                bins = st.slider("Number of bins", min_value=5, max_value=60, value=20)
                fig = plot_histogram(selected_df, selected_column, bins=bins)
                st.pyplot(fig)

            elif plot_option == "Boxplot":
                fig = plot_boxplot(selected_df, selected_column)
                st.pyplot(fig)

            elif plot_option == "Estimated PDF / KDE":
                fig = plot_kde_pdf(selected_df, selected_column)
                st.pyplot(fig)

            elif plot_option == "CDF":
                fig = plot_cdf(selected_df, selected_column)
                st.pyplot(fig)

            elif plot_option == "Q-Q Plot":
                fig = plot_qq(selected_df, selected_column)
                st.pyplot(fig)

            elif plot_option == "PMF":
                if is_discrete_numeric(clean_data):
                    fig = plot_pmf(selected_df, selected_column)
                    st.pyplot(fig)
                else:
                    st.warning(
                        "This column does not look discrete. PMF is mainly suitable for discrete numerical variables."
                    )

        with interpretation_col:
            with st.container(border=True):
                st.write("#### Plot Interpretation")

                plot_interpretations = get_plot_interpretation(
                    selected_df,
                    selected_column,
                    plot_option
                )

                for interpretation in plot_interpretations:
                    st.write(f"- {interpretation}")

                st.divider()

                st.write("#### What this plot is used for")

                if plot_option == "Histogram":
                    st.write(
                        "A histogram shows how often values fall into different ranges. "
                        "It helps identify shape, spread, skewness, and possible unusual patterns."
                    )

                elif plot_option == "Boxplot":
                    st.write(
                        "A boxplot summarizes the median, quartiles, spread, and possible outliers."
                    )

                elif plot_option == "Estimated PDF / KDE":
                    st.write(
                        "A KDE curve estimates the probability density of a continuous variable. "
                        "It is useful for understanding where values are concentrated."
                    )

                elif plot_option == "CDF":
                    st.write(
                        "A CDF shows the probability that a value is less than or equal to a certain point."
                    )

                elif plot_option == "Q-Q Plot":
                    st.write(
                        "A Q-Q plot compares the selected data against a normal distribution. "
                        "If points follow the line, the data is closer to normal."
                    )

                elif plot_option == "PMF":
                    st.write(
                        "A PMF shows the probability of each discrete value."
                    )

        st.divider()

        col_back, col_next = st.columns([1, 2])

        with col_back:
            if st.button("Back to Descriptive Statistics"):
                st.session_state.step = "descriptive_stats"
                st.rerun()

        with col_next:
            st.button("Continue to Normality Tests Coming Next", type="primary")