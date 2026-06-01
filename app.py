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


from utils.normality_tests import (
    run_all_normality_tests,
    create_normality_results_table,
    get_overall_normality_conclusion,
    get_test_explanation
)

from utils.t_tests import (
    run_one_sample_ttest,
    plot_one_sample_ttest,
    run_independent_ttest,
    plot_independent_ttest,
    run_paired_ttest,
    plot_paired_ttest,
    create_ttest_result_table,
    get_ttest_interpretation
)

from utils.anova_tests import (
    run_one_way_anova,
    run_tukey_hsd,
    plot_one_way_anova,
    run_two_way_anova,
    plot_two_way_interaction,
    get_one_way_anova_interpretation,
    get_two_way_anova_interpretation
)

from utils.chi_square_tests import (
    run_chi_square_independence,
    plot_chi_square_independence,
    get_independence_interpretation,
    run_chi_square_goodness_of_fit,
    plot_goodness_of_fit,
    get_goodness_of_fit_interpretation,
    create_chi_square_result_table
)

from utils.z_tests import (
    run_one_sample_ztest,
    plot_one_sample_ztest,
    run_two_sample_ztest,
    plot_two_sample_ztest,
    run_one_proportion_ztest,
    plot_one_proportion_ztest,
    run_two_proportion_ztest,
    plot_two_proportion_ztest,
    create_ztest_result_table,
    get_ztest_interpretation
)

from utils.distribution_fitting import (
    fit_all_distributions_least_squares,
    create_distribution_fit_table,
    plot_distribution_fits,
    plot_single_distribution_fit,
    plot_distribution_qq,
    get_distribution_fit_interpretation,
    get_selected_distribution_interpretation
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
            if st.button("Continue to Normality Tests", type="primary"):
                st.session_state.step = "normality_tests"
                st.rerun()


# ------------------------------------------------------------
# STEP 6: Normality Tests
# ------------------------------------------------------------

if st.session_state.step == "normality_tests":

    selected_df = st.session_state.selected_df

    st.subheader("Step 6: Normality Tests")

    st.info(
        "Select a numerical column and run Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to Visualizations"):
            st.session_state.step = "visualizations"
            st.rerun()

    else:
        selected_column = st.selectbox(
            "Choose a numerical column for normality testing",
            numerical_columns,
            key="normality_column"
        )

        alpha = st.selectbox(
            "Choose significance level (alpha)",
            [0.01, 0.05, 0.10],
            index=1
        )

        st.write(f"### Normality Test Results for `{selected_column}`")

        hypothesis = get_test_explanation()

        col_h0, col_h1 = st.columns(2)

        with col_h0:
            with st.container(border=True):
                st.write("#### Null Hypothesis H0")
                st.write(hypothesis["H0"])

        with col_h1:
            with st.container(border=True):
                st.write("#### Alternative Hypothesis H1")
                st.write(hypothesis["H1"])

        results = run_all_normality_tests(selected_df, selected_column, alpha)
        results_table = create_normality_results_table(results)
        overall_conclusion = get_overall_normality_conclusion(results)

        st.write("### Results Table")
        st.dataframe(results_table, use_container_width=True)

        st.write("### Overall Conclusion")

        with st.container(border=True):
            for line in overall_conclusion:
                st.write(f"- {line}")

        st.write("### Test-by-Test Notes")

        for result in results:
            with st.expander(result["Test"], expanded=False):
                st.write(f"**Decision:** {result['Decision']}")
                st.write(f"**Conclusion:** {result['Conclusion']}")
                st.write(f"**Note:** {result['Note']}")

                if result["Test"] == "Anderson-Darling":
                    st.write(f"**Critical Value Used:** {result['Critical Value Used']}")
                    st.write(f"**Significance Level Used:** {result['Significance Level Used']}")

        st.divider()

        st.write("### Visual Check")

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.write("#### Histogram")
            fig_hist = plot_histogram(selected_df, selected_column, bins=20)
            st.pyplot(fig_hist)

        with plot_col2:
            st.write("#### Q-Q Plot")
            fig_qq = plot_qq(selected_df, selected_column)
            st.pyplot(fig_qq)

        st.caption(
            "Statistical tests are useful, but plots are also important. "
            "For large datasets, even small deviations from normality may become statistically significant."
        )

        st.divider()

        col_back, col_next = st.columns([1, 2])

        with col_back:
            if st.button("Back to Visualizations"):
                st.session_state.step = "visualizations"
                st.rerun()

        with col_next:
            if st.button("Continue to T-Tests", type="primary"):
                st.session_state.step = "t_tests"
                st.rerun()



# ------------------------------------------------------------
# STEP 7: T-Tests
# ------------------------------------------------------------

if st.session_state.step == "t_tests":

    selected_df = st.session_state.selected_df

    st.subheader("Step 7: T-Tests")

    st.info(
        "Use t-tests to compare means. This section includes one-sample, independent two-sample, and paired t-tests."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to Normality Tests"):
            st.session_state.step = "normality_tests"
            st.rerun()

    else:
        alpha = st.selectbox(
            "Choose significance level (alpha)",
            [0.01, 0.05, 0.10],
            index=1,
            key="ttest_alpha"
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "One-sample t-test",
                "Independent two-sample t-test",
                "Paired t-test"
            ]
        )

        # ----------------------------------------------------
        # One-sample t-test
        # ----------------------------------------------------
        with tab1:
            st.write("### One-sample t-test")

            st.write(
                "Use this when you want to compare the mean of one numerical variable "
                "against a known or hypothesized value."
            )

            numeric_column = st.selectbox(
                "Choose numerical column",
                numerical_columns,
                key="one_sample_numeric"
            )

            default_mean = float(pd.to_numeric(selected_df[numeric_column], errors="coerce").mean())

            hypothesized_mean = st.number_input(
                "Enter hypothesized mean",
                value=default_mean,
                key="hypothesized_mean"
            )

            if st.button("Run One-sample t-test", type="primary"):
                try:
                    result = run_one_sample_ttest(
                        selected_df,
                        numeric_column,
                        hypothesized_mean,
                        alpha
                    )

                    result_table = create_ttest_result_table(result)
                    interpretation = get_ttest_interpretation(result)

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Visualization")
                        fig = plot_one_sample_ttest(
                            selected_df,
                            numeric_column,
                            hypothesized_mean
                        )
                        st.pyplot(fig)

                    with interpretation_col:
                        with st.container(border=True):
                            st.write("#### Interpretation")
                            for line in interpretation:
                                st.write(f"- {line}")

                    st.write("#### Result Table")
                    st.dataframe(result_table, use_container_width=True)

                except Exception as error:
                    st.error(error)

        # ----------------------------------------------------
        # Independent two-sample t-test
        # ----------------------------------------------------
        with tab2:
            st.write("### Independent two-sample t-test")

            st.write(
                "Use this when you want to compare the means of two independent groups."
            )

            numeric_column = st.selectbox(
                "Choose numerical outcome column",
                numerical_columns,
                key="independent_numeric"
            )

            possible_group_columns = [
                col for col in selected_df.columns
                if col != numeric_column and selected_df[col].nunique(dropna=True) <= 20
            ]

            if len(possible_group_columns) == 0:
                st.warning("No suitable grouping columns found. A grouping column should have a small number of categories.")

            else:
                group_column = st.selectbox(
                    "Choose grouping column",
                    possible_group_columns,
                    key="independent_group_column"
                )

                group_values = selected_df[group_column].dropna().unique().tolist()

                group1 = st.selectbox(
                    "Choose Group 1",
                    group_values,
                    key="independent_group_1"
                )

                group2 = st.selectbox(
                    "Choose Group 2",
                    group_values,
                    key="independent_group_2"
                )

                if group1 == group2:
                    st.warning("Group 1 and Group 2 must be different.")

                if st.button("Run Independent t-test", type="primary"):
                    try:
                        if group1 == group2:
                            st.error("Please select two different groups.")
                        else:
                            result = run_independent_ttest(
                                selected_df,
                                numeric_column,
                                group_column,
                                group1,
                                group2,
                                alpha
                            )

                            result_table = create_ttest_result_table(result)
                            interpretation = get_ttest_interpretation(result)

                            plot_col, interpretation_col = st.columns([1.2, 1])

                            with plot_col:
                                st.write("#### Visualization")
                                fig = plot_independent_ttest(
                                    selected_df,
                                    numeric_column,
                                    group_column,
                                    group1,
                                    group2
                                )
                                st.pyplot(fig)

                            with interpretation_col:
                                with st.container(border=True):
                                    st.write("#### Interpretation")
                                    for line in interpretation:
                                        st.write(f"- {line}")

                            st.write("#### Result Table")
                            st.dataframe(result_table, use_container_width=True)

                    except Exception as error:
                        st.error(error)

        # ----------------------------------------------------
        # Paired t-test
        # ----------------------------------------------------
        with tab3:
            st.write("### Paired t-test")

            st.write(
                "Use this when the two numerical columns are related, such as before-after measurements from the same subjects."
            )

            before_column = st.selectbox(
                "Choose before / first measurement column",
                numerical_columns,
                key="paired_before"
            )

            after_column = st.selectbox(
                "Choose after / second measurement column",
                numerical_columns,
                key="paired_after"
            )

            if before_column == after_column:
                st.warning("Before and after columns should be different.")

            if st.button("Run Paired t-test", type="primary"):
                try:
                    if before_column == after_column:
                        st.error("Please select two different columns.")
                    else:
                        result = run_paired_ttest(
                            selected_df,
                            before_column,
                            after_column,
                            alpha
                        )

                        result_table = create_ttest_result_table(result)
                        interpretation = get_ttest_interpretation(result)

                        plot_col, interpretation_col = st.columns([1.2, 1])

                        with plot_col:
                            st.write("#### Visualization")
                            fig = plot_paired_ttest(
                                selected_df,
                                before_column,
                                after_column
                            )
                            st.pyplot(fig)

                        with interpretation_col:
                            with st.container(border=True):
                                st.write("#### Interpretation")
                                for line in interpretation:
                                    st.write(f"- {line}")

                        st.write("#### Result Table")
                        st.dataframe(result_table, use_container_width=True)

                except Exception as error:
                    st.error(error)

        st.divider()

        col_back, col_next = st.columns([1, 2])

        with col_back:
            if st.button("Back to Normality Tests"):
                st.session_state.step = "normality_tests"
                st.rerun()



        with col_next:
            if st.button("Continue to ANOVA", type="primary"):
                st.session_state.step = "anova"
                st.rerun()




# ------------------------------------------------------------
# STEP 8: ANOVA
# ------------------------------------------------------------

if st.session_state.step == "anova":

    selected_df = st.session_state.selected_df

    st.subheader("Step 8: ANOVA Tests")

    st.info(
        "ANOVA is used to compare means across groups. This section includes one-way ANOVA and two-way ANOVA with interaction."
    )

    numerical_columns = get_numerical_columns(selected_df)

    categorical_columns = [
        col for col in selected_df.columns
        if selected_df[col].nunique(dropna=True) >= 2
        and selected_df[col].nunique(dropna=True) <= 20
    ]

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to T-Tests"):
            st.session_state.step = "t_tests"
            st.rerun()

    elif len(categorical_columns) == 0:
        st.warning("No suitable categorical columns found. ANOVA needs grouping/factor columns.")

        if st.button("Back to T-Tests"):
            st.session_state.step = "t_tests"
            st.rerun()

    else:
        alpha = st.selectbox(
            "Choose significance level (alpha)",
            [0.01, 0.05, 0.10],
            index=1,
            key="anova_alpha"
        )

        tab1, tab2 = st.tabs(["One-way ANOVA", "Two-way ANOVA"])

        # ----------------------------------------------------
        # One-way ANOVA
        # ----------------------------------------------------
        with tab1:
            st.write("### One-way ANOVA")

            st.write(
                "Use one-way ANOVA when you want to compare the mean of one numerical variable across 3 or more groups. "
                "It can also run with 2 groups, but t-test is usually simpler for exactly 2 groups."
            )

            numeric_column = st.selectbox(
                "Choose numerical outcome column",
                numerical_columns,
                key="one_way_numeric"
            )

            possible_factor_columns = [
                col for col in categorical_columns
                if col != numeric_column
            ]

            factor_column = st.selectbox(
                "Choose factor/group column",
                possible_factor_columns,
                key="one_way_factor"
            )

            group_count = selected_df[factor_column].nunique(dropna=True)

            st.caption(f"Detected number of groups in `{factor_column}`: {group_count}")

            if group_count < 2:
                st.warning("The selected factor must have at least 2 groups.")

            if st.button("Run One-way ANOVA", type="primary"):
                try:
                    result = run_one_way_anova(
                        selected_df,
                        numeric_column,
                        factor_column,
                        alpha
                    )

                    interpretation = get_one_way_anova_interpretation(result)

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Group Comparison Plot")
                        fig = plot_one_way_anova(
                            selected_df,
                            numeric_column,
                            factor_column
                        )
                        st.pyplot(fig)

                    with interpretation_col:
                        with st.container(border=True):
                            st.write("#### Interpretation")
                            for line in interpretation:
                                st.write(f"- {line}")

                    st.write("#### ANOVA Table")
                    st.dataframe(result["ANOVA Table"], use_container_width=True)

                    st.write("#### Group Summary")
                    st.dataframe(result["Group Summary"], use_container_width=True)

                    if result["Decision"] == "Reject H0":
                        st.write("#### Tukey HSD Post-hoc Test")
                        st.caption(
                            "Because ANOVA is significant, Tukey HSD helps identify which specific group pairs are different."
                        )

                        tukey_df = run_tukey_hsd(
                            selected_df,
                            numeric_column,
                            factor_column,
                            alpha
                        )

                        st.dataframe(tukey_df, use_container_width=True)

                except Exception as error:
                    st.error(error)

        # ----------------------------------------------------
        # Two-way ANOVA
        # ----------------------------------------------------
        with tab2:
            st.write("### Two-way ANOVA")

            st.write(
                "Use two-way ANOVA when you want to study the effect of two categorical factors on one numerical variable. "
                "This also checks whether the two factors interact with each other."
            )

            numeric_column = st.selectbox(
                "Choose numerical outcome column",
                numerical_columns,
                key="two_way_numeric"
            )

            possible_factor_columns = [
                col for col in categorical_columns
                if col != numeric_column
            ]

            factor1 = st.selectbox(
                "Choose Factor 1",
                possible_factor_columns,
                key="two_way_factor_1"
            )

            factor2_options = [
                col for col in possible_factor_columns
                if col != factor1
            ]

            if len(factor2_options) == 0:
                st.warning("Two-way ANOVA needs two different factor columns.")

            else:
                factor2 = st.selectbox(
                    "Choose Factor 2",
                    factor2_options,
                    key="two_way_factor_2"
                )

                st.caption(
                    f"Detected groups: `{factor1}` = {selected_df[factor1].nunique(dropna=True)}, "
                    f"`{factor2}` = {selected_df[factor2].nunique(dropna=True)}"
                )

                if st.button("Run Two-way ANOVA", type="primary"):
                    try:
                        result = run_two_way_anova(
                            selected_df,
                            numeric_column,
                            factor1,
                            factor2,
                            alpha
                        )

                        interpretation = get_two_way_anova_interpretation(result)

                        plot_col, interpretation_col = st.columns([1.2, 1])

                        with plot_col:
                            st.write("#### Interaction Plot")
                            fig = plot_two_way_interaction(
                                selected_df,
                                numeric_column,
                                factor1,
                                factor2
                            )
                            st.pyplot(fig)

                        with interpretation_col:
                            with st.container(border=True):
                                st.write("#### Interpretation")
                                for line in interpretation:
                                    st.write(f"- {line}")

                        st.write("#### ANOVA Table")
                        st.dataframe(result["ANOVA Table"], use_container_width=True)

                        st.write("#### Effects Summary")
                        st.dataframe(result["Effects Table"], use_container_width=True)

                        st.write("#### Group Combination Summary")
                        st.dataframe(result["Group Summary"], use_container_width=True)

                    except Exception as error:
                        st.error(error)

        st.divider()

        col_back, col_next = st.columns([1, 2])

        with col_back:
            if st.button("Back to T-Tests"):
                st.session_state.step = "t_tests"
                st.rerun()


        with col_next:
            if st.button("Continue to Chi-Square Tests", type="primary"):
                st.session_state.step = "chi_square"
                st.rerun()




# ------------------------------------------------------------
# STEP 9: Chi-Square Tests
# ------------------------------------------------------------

if st.session_state.step == "chi_square":

    selected_df = st.session_state.selected_df

    st.subheader("Step 9: Chi-Square Tests")

    st.info(
        "Chi-square tests are used for categorical data. "
        "This section includes the chi-square test of independence and chi-square goodness-of-fit test."
    )

    categorical_columns = [
        col for col in selected_df.columns
        if selected_df[col].nunique(dropna=True) >= 2
        and selected_df[col].nunique(dropna=True) <= 30
    ]

    if len(categorical_columns) == 0:
        st.warning("No suitable categorical columns found. Chi-square tests need categorical variables.")

        if st.button("Back to ANOVA"):
            st.session_state.step = "anova"
            st.rerun()

    else:
        alpha = st.selectbox(
            "Choose significance level (alpha)",
            [0.01, 0.05, 0.10],
            index=1,
            key="chi_square_alpha"
        )

        tab1, tab2 = st.tabs(
            [
                "Chi-square Test of Independence",
                "Chi-square Goodness-of-Fit"
            ]
        )

        # ----------------------------------------------------
        # Chi-square Test of Independence
        # ----------------------------------------------------
        with tab1:
            st.write("### Chi-square Test of Independence")

            st.write(
                "Use this test when you want to check whether two categorical variables are related."
            )

            column1 = st.selectbox(
                "Choose first categorical variable",
                categorical_columns,
                key="chi_ind_col1"
            )

            column2_options = [
                col for col in categorical_columns
                if col != column1
            ]

            if len(column2_options) == 0:
                st.warning("You need at least two different categorical columns for this test.")

            else:
                column2 = st.selectbox(
                    "Choose second categorical variable",
                    column2_options,
                    key="chi_ind_col2"
                )

                if st.button("Run Chi-square Test of Independence", type="primary"):
                    try:
                        result = run_chi_square_independence(
                            selected_df,
                            column1,
                            column2,
                            alpha
                        )

                        result_table = create_chi_square_result_table(result)
                        interpretation = get_independence_interpretation(result)

                        plot_col, interpretation_col = st.columns([1.2, 1])

                        with plot_col:
                            st.write("#### Observed Count Plot")
                            fig = plot_chi_square_independence(
                                selected_df,
                                column1,
                                column2
                            )
                            st.pyplot(fig)

                        with interpretation_col:
                            with st.container(border=True):
                                st.write("#### Interpretation")
                                for line in interpretation:
                                    st.write(f"- {line}")

                        st.write("#### Result Table")
                        st.dataframe(result_table, use_container_width=True)

                        col_obs, col_exp = st.columns(2)

                        with col_obs:
                            st.write("#### Observed Frequencies")
                            st.dataframe(result["Observed Table"], use_container_width=True)

                        with col_exp:
                            st.write("#### Expected Frequencies")
                            st.dataframe(result["Expected Table"], use_container_width=True)

                    except Exception as error:
                        st.error(error)

        # ----------------------------------------------------
        # Chi-square Goodness-of-Fit
        # ----------------------------------------------------
        with tab2:
            st.write("### Chi-square Goodness-of-Fit Test")

            st.write(
                "Use this test when you want to check whether one categorical variable follows an expected distribution."
            )

            categorical_column = st.selectbox(
                "Choose categorical variable",
                categorical_columns,
                key="chi_gof_col"
            )

            expected_method = st.radio(
                "Choose expected frequency method",
                [
                    "Equal expected frequencies",
                    "Custom expected frequencies"
                ],
                horizontal=True
            )

            observed_counts = (
                selected_df[categorical_column]
                .dropna()
                .astype(str)
                .value_counts()
                .sort_index()
            )

            expected_frequencies = None

            if expected_method == "Equal expected frequencies":
                st.caption(
                    "The test will compare the observed frequencies against equal expected frequencies across all categories."
                )

            else:
                st.caption(
                    "Edit the expected frequencies below. They will be automatically scaled to match the total observed count."
                )

                default_expected = observed_counts.sum() / len(observed_counts)

                expected_df = pd.DataFrame({
                    "Category": observed_counts.index,
                    "Observed Frequency": observed_counts.values,
                    "Expected Frequency": [default_expected] * len(observed_counts)
                })

                edited_expected_df = st.data_editor(
                    expected_df,
                    disabled=["Category", "Observed Frequency"],
                    use_container_width=True,
                    key="expected_frequency_editor"
                )

                expected_frequencies = dict(
                    zip(
                        edited_expected_df["Category"].astype(str),
                        edited_expected_df["Expected Frequency"]
                    )
                )

            if st.button("Run Chi-square Goodness-of-Fit Test", type="primary"):
                try:
                    result = run_chi_square_goodness_of_fit(
                        selected_df,
                        categorical_column,
                        expected_frequencies,
                        alpha
                    )

                    result_table = create_chi_square_result_table(result)
                    interpretation = get_goodness_of_fit_interpretation(result)

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Observed vs Expected Plot")
                        fig = plot_goodness_of_fit(result)
                        st.pyplot(fig)

                    with interpretation_col:
                        with st.container(border=True):
                            st.write("#### Interpretation")
                            for line in interpretation:
                                st.write(f"- {line}")

                    st.write("#### Result Table")
                    st.dataframe(result_table, use_container_width=True)

                    st.write("#### Observed vs Expected Frequencies")
                    st.dataframe(result["Observed Expected Table"], use_container_width=True)

                except Exception as error:
                    st.error(error)

        st.divider()

        col_back, col_next = st.columns([1, 2])

        with col_back:
            if st.button("Back to ANOVA"):
                st.session_state.step = "anova"
                st.rerun()


        with col_next:
            if st.button("Continue to Z-Tests", type="primary"):
                st.session_state.step = "z_tests"
                st.rerun()



# ------------------------------------------------------------
# STEP 10: Z-Tests
# ------------------------------------------------------------

if st.session_state.step == "z_tests":

    selected_df = st.session_state.selected_df

    st.subheader("Step 10: Z-Tests")

    st.info(
        "Z-tests are used for mean or proportion testing when the normal approximation is appropriate. "
        "For mean z-tests, the population standard deviation should be known."
    )

    numerical_columns = get_numerical_columns(selected_df)

    categorical_columns = [
        col for col in selected_df.columns
        if selected_df[col].nunique(dropna=True) >= 2
        and selected_df[col].nunique(dropna=True) <= 30
    ]

    alpha = st.selectbox(
        "Choose significance level (alpha)",
        [0.01, 0.05, 0.10],
        index=1,
        key="ztest_alpha"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "One-sample mean z-test",
            "Two-sample mean z-test",
            "One-proportion z-test",
            "Two-proportion z-test"
        ]
    )

    # --------------------------------------------------------
    # One-sample mean z-test
    # --------------------------------------------------------
    with tab1:
        st.write("### One-sample Mean Z-Test")

        st.write(
            "Use this when you want to compare one sample mean against a hypothesized population mean, "
            "and the population standard deviation is known."
        )

        if len(numerical_columns) == 0:
            st.warning("No numerical columns found.")

        else:
            numeric_column = st.selectbox(
                "Choose numerical column",
                numerical_columns,
                key="one_sample_z_numeric"
            )

            sample_mean_default = float(pd.to_numeric(selected_df[numeric_column], errors="coerce").mean())
            sample_std_default = float(pd.to_numeric(selected_df[numeric_column], errors="coerce").std())

            hypothesized_mean = st.number_input(
                "Enter hypothesized mean",
                value=sample_mean_default,
                key="one_sample_z_mean"
            )

            population_std = st.number_input(
                "Enter known population standard deviation",
                min_value=0.0001,
                value=sample_std_default if sample_std_default > 0 else 1.0,
                key="one_sample_z_std"
            )

            if st.button("Run One-sample Mean Z-Test", type="primary"):
                try:
                    result = run_one_sample_ztest(
                        selected_df,
                        numeric_column,
                        hypothesized_mean,
                        population_std,
                        alpha
                    )

                    result_table = create_ztest_result_table(result)
                    interpretation = get_ztest_interpretation(result)

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Visualization")
                        fig = plot_one_sample_ztest(
                            selected_df,
                            numeric_column,
                            hypothesized_mean
                        )
                        st.pyplot(fig)

                    with interpretation_col:
                        with st.container(border=True):
                            st.write("#### Interpretation")
                            for line in interpretation:
                                st.write(f"- {line}")

                    st.write("#### Result Table")
                    st.dataframe(result_table, use_container_width=True)

                except Exception as error:
                    st.error(error)

    # --------------------------------------------------------
    # Two-sample mean z-test
    # --------------------------------------------------------
    with tab2:
        st.write("### Two-sample Mean Z-Test")

        st.write(
            "Use this when you want to compare two independent sample means, "
            "and the population standard deviations for both groups are known."
        )

        if len(numerical_columns) == 0:
            st.warning("No numerical columns found.")

        else:
            numeric_column = st.selectbox(
                "Choose numerical outcome column",
                numerical_columns,
                key="two_sample_z_numeric"
            )

            possible_group_columns = [
                col for col in selected_df.columns
                if col != numeric_column
                and selected_df[col].nunique(dropna=True) >= 2
                and selected_df[col].nunique(dropna=True) <= 30
            ]

            if len(possible_group_columns) == 0:
                st.warning("No suitable grouping columns found.")

            else:
                group_column = st.selectbox(
                    "Choose grouping column",
                    possible_group_columns,
                    key="two_sample_z_group_col"
                )

                group_values = selected_df[group_column].dropna().astype(str).unique().tolist()

                group1 = st.selectbox(
                    "Choose Group 1",
                    group_values,
                    key="two_sample_z_group1"
                )

                group2 = st.selectbox(
                    "Choose Group 2",
                    group_values,
                    key="two_sample_z_group2"
                )

                default_std = float(pd.to_numeric(selected_df[numeric_column], errors="coerce").std())

                population_std1 = st.number_input(
                    "Enter known population standard deviation for Group 1",
                    min_value=0.0001,
                    value=default_std if default_std > 0 else 1.0,
                    key="two_sample_z_std1"
                )

                population_std2 = st.number_input(
                    "Enter known population standard deviation for Group 2",
                    min_value=0.0001,
                    value=default_std if default_std > 0 else 1.0,
                    key="two_sample_z_std2"
                )

                if group1 == group2:
                    st.warning("Group 1 and Group 2 must be different.")

                if st.button("Run Two-sample Mean Z-Test", type="primary"):
                    try:
                        if group1 == group2:
                            st.error("Please select two different groups.")
                        else:
                            result = run_two_sample_ztest(
                                selected_df,
                                numeric_column,
                                group_column,
                                group1,
                                group2,
                                population_std1,
                                population_std2,
                                alpha
                            )

                            result_table = create_ztest_result_table(result)
                            interpretation = get_ztest_interpretation(result)

                            plot_col, interpretation_col = st.columns([1.2, 1])

                            with plot_col:
                                st.write("#### Visualization")
                                fig = plot_two_sample_ztest(
                                    selected_df,
                                    numeric_column,
                                    group_column,
                                    group1,
                                    group2
                                )
                                st.pyplot(fig)

                            with interpretation_col:
                                with st.container(border=True):
                                    st.write("#### Interpretation")
                                    for line in interpretation:
                                        st.write(f"- {line}")

                            st.write("#### Result Table")
                            st.dataframe(result_table, use_container_width=True)

                    except Exception as error:
                        st.error(error)

    # --------------------------------------------------------
    # One-proportion z-test
    # --------------------------------------------------------
    with tab3:
        st.write("### One-proportion Z-Test")

        st.write(
            "Use this when you want to compare one sample proportion against a hypothesized population proportion."
        )

        if len(categorical_columns) == 0:
            st.warning("No suitable categorical columns found.")

        else:
            categorical_column = st.selectbox(
                "Choose categorical column",
                categorical_columns,
                key="one_prop_z_col"
            )

            categories = selected_df[categorical_column].dropna().astype(str).unique().tolist()

            success_category = st.selectbox(
                "Choose success category",
                categories,
                key="one_prop_z_success"
            )

            hypothesized_proportion = st.number_input(
                "Enter hypothesized proportion",
                min_value=0.0001,
                max_value=0.9999,
                value=0.5,
                key="one_prop_z_p0"
            )

            if st.button("Run One-proportion Z-Test", type="primary"):
                try:
                    result = run_one_proportion_ztest(
                        selected_df,
                        categorical_column,
                        success_category,
                        hypothesized_proportion,
                        alpha
                    )

                    result_table = create_ztest_result_table(result)
                    interpretation = get_ztest_interpretation(result)

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Visualization")
                        fig = plot_one_proportion_ztest(result)
                        st.pyplot(fig)

                    with interpretation_col:
                        with st.container(border=True):
                            st.write("#### Interpretation")
                            for line in interpretation:
                                st.write(f"- {line}")

                    st.write("#### Result Table")
                    st.dataframe(result_table, use_container_width=True)

                except Exception as error:
                    st.error(error)

    # --------------------------------------------------------
    # Two-proportion z-test
    # --------------------------------------------------------
    with tab4:
        st.write("### Two-proportion Z-Test")

        st.write(
            "Use this when you want to compare the proportion of a selected outcome between two independent groups."
        )

        if len(categorical_columns) < 2:
            st.warning("This test needs at least two suitable categorical columns.")

        else:
            outcome_column = st.selectbox(
                "Choose outcome categorical column",
                categorical_columns,
                key="two_prop_z_outcome"
            )

            success_categories = selected_df[outcome_column].dropna().astype(str).unique().tolist()

            success_category = st.selectbox(
                "Choose success category",
                success_categories,
                key="two_prop_z_success"
            )

            possible_group_columns = [
                col for col in categorical_columns
                if col != outcome_column
            ]

            group_column = st.selectbox(
                "Choose grouping column",
                possible_group_columns,
                key="two_prop_z_group_col"
            )

            group_values = selected_df[group_column].dropna().astype(str).unique().tolist()

            group1 = st.selectbox(
                "Choose Group 1",
                group_values,
                key="two_prop_z_group1"
            )

            group2 = st.selectbox(
                "Choose Group 2",
                group_values,
                key="two_prop_z_group2"
            )

            if group1 == group2:
                st.warning("Group 1 and Group 2 must be different.")

            if st.button("Run Two-proportion Z-Test", type="primary"):
                try:
                    if group1 == group2:
                        st.error("Please select two different groups.")
                    else:
                        result = run_two_proportion_ztest(
                            selected_df,
                            outcome_column,
                            success_category,
                            group_column,
                            group1,
                            group2,
                            alpha
                        )

                        result_table = create_ztest_result_table(result)
                        interpretation = get_ztest_interpretation(result)

                        plot_col, interpretation_col = st.columns([1.2, 1])

                        with plot_col:
                            st.write("#### Visualization")
                            fig = plot_two_proportion_ztest(result)
                            st.pyplot(fig)

                        with interpretation_col:
                            with st.container(border=True):
                                st.write("#### Interpretation")
                                for line in interpretation:
                                    st.write(f"- {line}")

                        st.write("#### Result Table")
                        st.dataframe(result_table, use_container_width=True)

                except Exception as error:
                    st.error(error)

    st.divider()

    col_back, col_next = st.columns([1, 2])

    with col_back:
        if st.button("Back to Chi-Square Tests"):
            st.session_state.step = "chi_square"
            st.rerun()

    with col_next:
        if st.button("Continue to Distribution Fitting", type="primary"):
            st.session_state.step = "distribution_fitting"
            st.rerun()


# ------------------------------------------------------------
# STEP 11: Distribution Fitting
# ------------------------------------------------------------

if st.session_state.step == "distribution_fitting":

    selected_df = st.session_state.selected_df

    st.subheader("Step 11: Distribution Fitting")

    st.info(
        "This section fits Normal, Exponential, and Uniform distributions using least-squares error. "
        "The observed histogram density is compared against each theoretical PDF curve."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to Z-Tests"):
            st.session_state.step = "z_tests"
            st.rerun()

    else:
        selected_column = st.selectbox(
            "Choose numerical column for distribution fitting",
            numerical_columns,
            key="dist_fit_column"
        )

        bins = st.slider(
            "Choose number of histogram bins",
            min_value=10,
            max_value=60,
            value=30,
            key="dist_fit_bins"
        )

        if st.button("Fit Distributions using Least Squares", type="primary"):
            try:
                results, data, bin_centers, observed_density = fit_all_distributions_least_squares(
                    selected_df,
                    selected_column,
                    bins=bins
                )

                st.session_state.dist_fit_results = results
                st.session_state.dist_fit_data = data
                st.session_state.dist_fit_column_name = selected_column
                st.session_state.dist_fit_bins_used = bins

            except Exception as error:
                st.error(error)

        if "dist_fit_results" in st.session_state:

            results = st.session_state.dist_fit_results
            data = st.session_state.dist_fit_data
            fitted_column = st.session_state.dist_fit_column_name
            bins_used = st.session_state.dist_fit_bins_used

            st.write(f"### Fitting Results for `{fitted_column}`")

            fit_table = create_distribution_fit_table(results)

            plot_col, interpretation_col = st.columns([1.2, 1])

            with plot_col:
                st.write("#### Histogram with Fitted PDFs")
                fig = plot_distribution_fits(data, results, bins=bins_used)
                st.pyplot(fig)

            with interpretation_col:
                with st.container(border=True):
                    st.write("#### Interpretation")

                    interpretation = get_distribution_fit_interpretation(
                        results,
                        fitted_column
                    )

                    for line in interpretation:
                        st.write(f"- {line}")

            st.write("#### Least-Squares Fit Comparison Table")
            st.dataframe(fit_table, use_container_width=True)

            st.divider()

            st.write("### Inspect One Fitted Distribution")

            distribution_names = [result["Distribution"] for result in results]

            selected_distribution_name = st.selectbox(
                "Choose fitted distribution to inspect",
                distribution_names,
                key="selected_distribution_inspect"
            )

            selected_result = None

            for result in results:
                if result["Distribution"] == selected_distribution_name:
                    selected_result = result
                    break

            if selected_result is not None:

                plot_col2, interpretation_col2 = st.columns([1.2, 1])

                with plot_col2:
                    st.write(f"#### {selected_distribution_name} Fit")
                    fig_single = plot_single_distribution_fit(
                        data,
                        selected_result,
                        bins=bins_used
                    )
                    st.pyplot(fig_single)

                    st.write(f"#### Q-Q Plot against {selected_distribution_name}")
                    fig_qq = plot_distribution_qq(data, selected_result)
                    st.pyplot(fig_qq)

                with interpretation_col2:
                    with st.container(border=True):
                        st.write("#### Distribution Notes")

                        selected_interpretation = get_selected_distribution_interpretation(
                            selected_result
                        )

                        for line in selected_interpretation:
                            st.write(f"- {line}")

                        st.divider()

                        st.write("#### Error Values")

                        st.write(f"**SSE:** {selected_result['SSE']:.6f}")
                        st.write(f"**MSE:** {selected_result['MSE']:.6f}")
                        st.write(f"**RMSE:** {selected_result['RMSE']:.6f}")

        st.divider()

        col_back, col_next = st.columns([1, 2])

        with col_back:
            if st.button("Back to Z-Tests"):
                st.session_state.step = "z_tests"
                st.rerun()

        with col_next:
            st.button("Continue to CLT Simulation Coming Next", type="primary")