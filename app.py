import streamlit as st
import pandas as pd
import numpy as np

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

from utils.clt_simulation import (
    simulate_sample_means,
    create_clt_summary_table,
    plot_original_distribution,
    plot_sampling_distribution,
    simulate_multiple_sample_sizes,
    plot_sample_size_comparison,
    run_normality_check_on_sample_means,
    create_sample_means_normality_table,
    get_clt_interpretation
)

from utils.ui_components import (
    inject_global_css,
    render_sidebar,
    render_top_bar,
    metric_card,
    info_card,
    page_locked_message,
    result_status_card
)


from utils.assumption_checks import (
    check_one_sample_ttest_assumptions,
    check_independent_ttest_assumptions,
    check_paired_ttest_assumptions,
    create_assumption_table,
    plot_assumption_histogram,
    plot_assumption_qq,
    plot_independent_groups_boxplot,
    plot_paired_differences
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

from utils.smart_descriptive import (
    detect_variable_type,
    get_descriptive_candidate_columns,
    create_categorical_summary,
    plot_categorical_bar,
    plot_categorical_percentage_bar,
    plot_categorical_donut,
    get_categorical_interpretation
)

from utils.nonparametric_tests import (
    run_mannwhitney_u_test,
    plot_mannwhitney_u,
    run_wilcoxon_signed_rank_test,
    plot_wilcoxon_signed_rank,
    plot_wilcoxon_differences,
    run_kruskal_wallis_test,
    plot_kruskal_wallis,
    run_friedman_test,
    plot_friedman_test,
    create_result_table,
    get_nonparametric_interpretation
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

inject_global_css()

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

if "uploaded_file_signature" not in st.session_state:
    st.session_state.uploaded_file_signature = None


def reset_project():
    st.session_state.step = "upload"
    st.session_state.df = None
    st.session_state.selected_df = None
    st.session_state.column_summary = None
    st.session_state.selected_columns = []
    st.session_state.uploaded_file_name = None


def clear_analysis_state_for_new_upload():
    """
    Clears selected dataset and analysis outputs when a new dataset is uploaded.
    This prevents old selected columns/results from staying after changing files.
    """

    keys_to_clear = [
        "selected_df",
        "selected_columns",
        "column_summary",

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



def render_assumption_panel(assumption_result, panel_key):
    """
    Renders a reusable assumption check panel.
    """

    st.write("### Assumption Checks")

    result_status_card(
        title="Recommended Action",
        status=assumption_result["recommendation_title"],
        message=assumption_result["recommendation"],
        status_type=assumption_result["recommendation_type"]
    )

    with st.expander("View Detailed Assumption Checks", expanded=True):
        checks_table = create_assumption_table(assumption_result)
        st.dataframe(checks_table, use_container_width=True)

    diagnostic_data = assumption_result.get("diagnostic_data", {})

    with st.expander("View Diagnostic Plots", expanded=False):

        if assumption_result["test"] == "One-sample t-test":
            data = diagnostic_data["data"]

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                fig_hist = plot_assumption_histogram(
                    data,
                    title="Distribution of Selected Variable",
                    x_label="Values"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with plot_col2:
                fig_qq = plot_assumption_qq(
                    data,
                    title="Q-Q Plot for Normality"
                )
                st.plotly_chart(fig_qq, use_container_width=True)

        elif assumption_result["test"] == "Independent two-sample t-test":
            group1_data = diagnostic_data["group1_data"]
            group2_data = diagnostic_data["group2_data"]
            group1 = diagnostic_data["group1"]
            group2 = diagnostic_data["group2"]
            numeric_column = diagnostic_data["numeric_column"]

            fig_box = plot_independent_groups_boxplot(
                group1_data,
                group2_data,
                group1,
                group2,
                numeric_column
            )
            st.plotly_chart(fig_box, use_container_width=True)

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                fig_qq_1 = plot_assumption_qq(
                    group1_data,
                    title=f"Q-Q Plot: {group1}"
                )
                st.plotly_chart(fig_qq_1, use_container_width=True)

            with plot_col2:
                fig_qq_2 = plot_assumption_qq(
                    group2_data,
                    title=f"Q-Q Plot: {group2}"
                )
                st.plotly_chart(fig_qq_2, use_container_width=True)

        elif assumption_result["test"] == "Paired t-test":
            differences = diagnostic_data["differences"]

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                fig_diff = plot_paired_differences(differences)
                st.plotly_chart(fig_diff, use_container_width=True)

            with plot_col2:
                fig_qq = plot_assumption_qq(
                    differences,
                    title="Q-Q Plot of Paired Differences"
                )
                st.plotly_chart(fig_qq, use_container_width=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

# st.markdown('<div class="main-title">📊 Data Methods Lab</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="subtitle">Upload a dataset, select useful columns, and prepare it for statistical analysis.</div>',
#     unsafe_allow_html=True
# )

# st.divider()

current_page = render_sidebar()
render_top_bar(current_page)


PAGE_TO_STEP = {
    "Import dataset": "upload",
    "Column selection": "select_columns",
    "Dataset overview": "selected_preview",
    "Descriptive statistics": "descriptive_stats",
    "Visualizations": "visualizations",
    "Normality tests": "normality_tests",
    "T-tests": "t_tests",
    "Non-parametric tests": "nonparametric_tests",
    "ANOVA": "anova",
    "Chi-square tests": "chi_square",
    "Z-tests": "z_tests",
    "Distribution fitting": "distribution_fitting",
    "Central limit theorem": "clt_simulation",
}

st.session_state.step = PAGE_TO_STEP.get(current_page, "upload")

# ------------------------------------------------------------
# STEP 1: Upload dataset
# ------------------------------------------------------------

if st.session_state.step == "upload":

    st.subheader("Import Dataset")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel XLSX"
    )

    if uploaded_file is not None:
        try:
            uploaded_file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

            if st.session_state.uploaded_file_signature != uploaded_file_signature:
                clear_analysis_state_for_new_upload()
                st.session_state.uploaded_file_signature = uploaded_file_signature

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

            if st.button("Continue to Column Selection", type="primary"):
                st.session_state.current_page = "Column selection"
                st.rerun()

        except Exception as error:
            st.error(error)


# ------------------------------------------------------------
# STEP 2: Select columns using checkboxes
# ------------------------------------------------------------

elif st.session_state.step == "select_columns":


    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        st.stop()


    df = st.session_state.df
    column_summary = st.session_state.column_summary

    st.subheader("Column Selection")

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
            st.session_state.current_page = "Dataset overview"
            st.rerun()

    st.button("Back to Upload", on_click=reset_project)


# ------------------------------------------------------------
# STEP 3: Show selected data nicely
# ------------------------------------------------------------

# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------

elif st.session_state.step == "selected_preview":

    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("Dataset Overview")

    st.info(
        "This page summarizes the selected dataset that will be used across all analysis modules."
    )

    numerical_columns = get_numerical_columns(selected_df)
    categorical_columns = get_categorical_columns(selected_df)

    total_missing = int(selected_df.isna().sum().sum())
    duplicate_rows = int(selected_df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Rows",
            f"{selected_df.shape[0]:,}",
            "records available"
        )

    with col2:
        metric_card(
            "Selected columns",
            selected_df.shape[1],
            "columns used for analysis"
        )

    with col3:
        metric_card(
            "Numerical columns",
            len(numerical_columns),
            "usable for mean-based tests"
        )

    with col4:
        metric_card(
            "Categorical columns",
            len(categorical_columns),
            "usable for grouping/tests"
        )

    st.write("### Selected Columns")

    selected_column_cards = ""

    for column in selected_df.columns:
        dtype_text = str(selected_df[column].dtype)
        missing_count = int(selected_df[column].isna().sum())
        unique_count = int(selected_df[column].nunique(dropna=True))

        if column in numerical_columns:
            badge = "Numerical"
            badge_color = "#00B894"
        elif column in categorical_columns:
            badge = "Categorical"
            badge_color = "#A78BFA"
        else:
            badge = "Other"
            badge_color = "#F59E0B"

        selected_column_cards += f"""
        <div style="
            border: 1px solid #3A3B40;
            background: #242529;
            border-radius: 14px;
            padding: 16px;
            min-height: 125px;
        ">
            <div style="font-weight:800; font-size:16px; color:#F5F5F5; margin-bottom:8px;">
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
            <div style="font-size:13px; color:#A3A3A3; margin-top:8px;">
                Type: <b>{dtype_text}</b>
            </div>
            <div style="font-size:13px; color:#A3A3A3;">
                Unique values: <b>{unique_count}</b>
            </div>
            <div style="font-size:13px; color:#A3A3A3;">
                Missing values: <b>{missing_count}</b>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div style="
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-top: 12px;
            margin-bottom: 28px;
        ">
            {selected_column_cards}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### Dataset Quality")

    q1, q2, q3 = st.columns(3)

    with q1:
        metric_card(
            "Missing values",
            total_missing,
            "total missing cells"
        )

    with q2:
        metric_card(
            "Duplicate rows",
            duplicate_rows,
            "fully duplicated records"
        )

    with q3:
        missing_percentage = round(
            (total_missing / (selected_df.shape[0] * selected_df.shape[1])) * 100,
            2
        ) if selected_df.shape[0] > 0 and selected_df.shape[1] > 0 else 0

        metric_card(
            "Missing %",
            f"{missing_percentage}%",
            "overall dataset missingness"
        )

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

    st.caption(
        "Use the sidebar to move directly to descriptive statistics, visualizations, tests, diagnostics, or distribution fitting."
    )


# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

elif st.session_state.step == "descriptive_stats":

    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("Descriptive Statistics")

    st.info(
        "Select a column to view a smart descriptive summary. "
        "The app automatically detects whether the variable is continuous numerical, discrete numerical, binary, or categorical."
    )

    candidate_columns = get_descriptive_candidate_columns(selected_df)

    if len(candidate_columns) == 0:
        st.warning("No suitable columns found for descriptive statistics.")

    else:
        selected_column = st.selectbox(
            "Choose a column",
            candidate_columns,
            key="smart_descriptive_column"
        )

        detected = detect_variable_type(selected_df, selected_column)
        variable_type = detected["type"]

        badge_color_map = {
            "success": "#00B894",
            "warning": "#F59E0B",
            "error": "#EF4444",
            "info": "#3B82F6",
            "primary": "#7C5CFF",
            "secondary": "#A78BFA",
        }

        badge_color = badge_color_map.get(detected["badge_type"], "#3B82F6")

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                margin-top:18px;
                margin-bottom:12px;
            ">
                <h3 style="margin:0;">Summary for <code>{selected_column}</code></h3>
                <span style="
                    padding:5px 10px;
                    border-radius:999px;
                    background:{badge_color}22;
                    color:{badge_color};
                    font-weight:800;
                    font-size:12px;
                ">
                    {detected["badge"]}
                </span>
            </div>
            <div style="color:#A3A3A3; margin-bottom:18px;">
                {detected["reason"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ------------------------------------------------------------
        # NUMERICAL SUMMARY
        # ------------------------------------------------------------

        if variable_type in ["Continuous numerical", "Discrete numerical"]:

            stats = calculate_descriptive_statistics(selected_df, selected_column)
            stats_table = create_descriptive_stats_table(stats)
            interpretations = interpret_descriptive_statistics(stats)

            mean_value = stats["Mean"]
            median_value = stats["Median"]
            mode_value = stats["Mode"]
            min_value = stats["Minimum"]
            max_value = stats["Maximum"]
            range_value = stats["Range"]
            variance_value = stats["Sample Variance"]
            std_value = stats["Sample Standard Deviation"]
            skewness_value = stats["Skewness"]
            kurtosis_value = stats["Excess Kurtosis"]
            valid_values = stats["Valid Numerical Values"]
            missing_percentage = stats["Missing / Invalid %"]

            row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

            with row1_col1:
                metric_card("Mean", f"{mean_value:.4f}", "average value")

            with row1_col2:
                metric_card("Median", f"{median_value:.4f}", "middle value")

            with row1_col3:
                metric_card("Std deviation", f"{std_value:.4f}", f"variance: {variance_value:.4f}")

            with row1_col4:
                metric_card("Valid values", f"{valid_values:,}", f"missing/invalid: {missing_percentage}%")

            row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)

            with row2_col1:
                metric_card("Minimum", f"{min_value:.4f}", "lowest value")

            with row2_col2:
                metric_card("Maximum", f"{max_value:.4f}", "highest value")

            with row2_col3:
                metric_card("Range", f"{range_value:.4f}", "max - min")

            with row2_col4:
                metric_card("Mode", mode_value, "most frequent value")

            st.divider()

            st.write("### Distribution Shape")

            shape_col1, shape_col2 = st.columns(2)

            with shape_col1:
                if skewness_value > 1:
                    skew_label = "Strong right-skew"
                    skew_note = "Longer tail on the right side"
                elif skewness_value > 0.5:
                    skew_label = "Moderate right-skew"
                    skew_note = "Some larger values pull the data right"
                elif skewness_value >= -0.5:
                    skew_label = "Approximately symmetric"
                    skew_note = "Skewness is close to 0"
                elif skewness_value >= -1:
                    skew_label = "Moderate left-skew"
                    skew_note = "Some smaller values pull the data left"
                else:
                    skew_label = "Strong left-skew"
                    skew_note = "Longer tail on the left side"

                metric_card(
                    "Skewness",
                    f"{skewness_value:.4f}",
                    f"{skew_label} · {skew_note}"
                )

            with shape_col2:
                if kurtosis_value > 1:
                    kurtosis_label = "Heavy tails"
                    kurtosis_note = "Possible extreme values"
                elif kurtosis_value < -1:
                    kurtosis_label = "Light tails / flatter"
                    kurtosis_note = "Flatter than normal"
                else:
                    kurtosis_label = "Near normal tail behavior"
                    kurtosis_note = "Not extremely different from normal"

                metric_card(
                    "Excess kurtosis",
                    f"{kurtosis_value:.4f}",
                    f"{kurtosis_label} · {kurtosis_note}"
                )

            st.divider()

            st.write("### Visual Summary")

            if variable_type == "Discrete numerical":
                plot_options = ["Histogram", "Boxplot", "PMF"]
            else:
                plot_options = ["Histogram", "Boxplot", "Estimated PDF / KDE"]

            plot_type = st.radio(
                "Choose plot type",
                plot_options,
                horizontal=True,
                key="smart_descriptive_plot_type"
            )

            plot_col, interpretation_col = st.columns([1.25, 1])

            with plot_col:
                if plot_type == "Histogram":
                    bins = st.slider(
                        "Number of bins",
                        min_value=5,
                        max_value=60,
                        value=20,
                        key="smart_descriptive_hist_bins"
                    )

                    fig = plot_histogram(selected_df, selected_column, bins=bins)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Boxplot":
                    fig = plot_boxplot(selected_df, selected_column)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Estimated PDF / KDE":
                    fig = plot_kde_pdf(selected_df, selected_column)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "PMF":
                    fig = plot_pmf(selected_df, selected_column)
                    st.plotly_chart(fig, use_container_width=True)

            with interpretation_col:
                with st.container(border=True):
                    st.write("#### Quick Interpretation")

                    for interpretation in interpretations:
                        st.write(f"- {interpretation}")

                    st.divider()

                    st.write("#### What to notice")

                    if plot_type == "Histogram":
                        st.write("- Look at the overall shape of the bars.")
                        st.write("- If one tail is longer, the data may be skewed.")
                        st.write("- Mean and median lines help compare balance.")

                    elif plot_type == "Boxplot":
                        st.write("- The box shows the middle 50% of the data.")
                        st.write("- The line inside the box is the median.")
                        st.write("- Points outside the whiskers may be potential outliers.")

                    elif plot_type == "Estimated PDF / KDE":
                        st.write("- Higher parts of the curve show where values are more concentrated.")
                        st.write("- Multiple peaks may suggest multiple clusters.")
                        st.write("- KDE is more suitable for continuous numerical data.")

                    elif plot_type == "PMF":
                        st.write("- PMF is useful for discrete values.")
                        st.write("- It shows the probability of each value.")
                        st.write("- This is better for count-like variables than KDE.")

            st.divider()

            with st.expander("View Detailed Numerical Statistics Table", expanded=False):
                st.dataframe(stats_table, use_container_width=True)

            with st.expander("View Selected Column Preview", expanded=False):
                preview_df = selected_df[[selected_column]].head(30)
                st.dataframe(preview_df, use_container_width=True)

        # ------------------------------------------------------------
        # CATEGORICAL / BINARY SUMMARY
        # ------------------------------------------------------------

        elif variable_type in ["Binary categorical", "Categorical"]:

            summary, frequency_table = create_categorical_summary(
                selected_df,
                selected_column
            )

            row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

            with row1_col1:
                metric_card(
                    "Valid values",
                    f"{summary['Valid Values']:,}",
                    f"missing: {summary['Missing %']:.2f}%"
                )

            with row1_col2:
                metric_card(
                    "Unique categories",
                    summary["Unique Categories"],
                    "distinct values"
                )

            with row1_col3:
                metric_card(
                    "Mode",
                    summary["Mode"],
                    "most common category"
                )

            with row1_col4:
                metric_card(
                    "Mode share",
                    f"{summary['Mode %']:.2f}%",
                    f"count: {summary['Mode Count']}"
                )

            st.divider()

            st.write("### Category Distribution")

            plot_type = st.radio(
                "Choose plot type",
                ["Count Bar Chart", "Percentage Bar Chart", "Donut Chart"],
                horizontal=True,
                key="categorical_descriptive_plot_type"
            )

            plot_col, interpretation_col = st.columns([1.25, 1])

            with plot_col:
                if plot_type == "Count Bar Chart":
                    fig = plot_categorical_bar(frequency_table, selected_column)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Percentage Bar Chart":
                    fig = plot_categorical_percentage_bar(frequency_table, selected_column)
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Donut Chart":
                    if summary["Unique Categories"] > 8:
                        st.warning(
                            "This column has many categories. A bar chart is easier to read than a donut chart."
                        )

                    fig = plot_categorical_donut(frequency_table, selected_column)
                    st.plotly_chart(fig, use_container_width=True)

            with interpretation_col:
                with st.container(border=True):
                    st.write("#### Quick Interpretation")

                    categorical_interpretation = get_categorical_interpretation(
                        summary,
                        frequency_table,
                        variable_type
                    )

                    for interpretation in categorical_interpretation:
                        st.write(f"- {interpretation}")

                    st.divider()

                    st.write("#### What to notice")
                    st.write("- Check which category appears most often.")
                    st.write("- Compare category percentages, not only raw counts.")
                    st.write("- If one category dominates, the variable is imbalanced.")

            st.divider()

            st.write("### Frequency Table")
            st.dataframe(frequency_table, use_container_width=True)

            with st.expander("View Selected Column Preview", expanded=False):
                preview_df = selected_df[[selected_column]].head(30)
                st.dataframe(preview_df, use_container_width=True)

        # ------------------------------------------------------------
        # OTHER TYPES
        # ------------------------------------------------------------

        else:
            st.warning(
                f"This column was detected as `{variable_type}`. "
                "It may not be suitable for standard descriptive statistics."
            )

            st.dataframe(
                selected_df[[selected_column]].head(30),
                use_container_width=True
            )

    st.caption(
        "Use the sidebar to continue to visualizations, normality tests, hypothesis tests, or diagnostics."
    )


# ------------------------------------------------------------
# STEP 5: Visualizations
# ------------------------------------------------------------

if st.session_state.step == "visualizations":


    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()


    selected_df = st.session_state.selected_df

    st.subheader("Data Visualizations")

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
                st.plotly_chart(fig, use_container_width=True)

            elif plot_option == "Boxplot":
                fig = plot_boxplot(selected_df, selected_column)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_option == "Estimated PDF / KDE":
                fig = plot_kde_pdf(selected_df, selected_column)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_option == "CDF":
                fig = plot_cdf(selected_df, selected_column)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_option == "Q-Q Plot":
                fig = plot_qq(selected_df, selected_column)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_option == "PMF":
                if is_discrete_numeric(clean_data):
                    fig = plot_pmf(selected_df, selected_column)
                    st.plotly_chart(fig, use_container_width=True)
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

    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()


    selected_df = st.session_state.selected_df

    st.subheader("Normality Tests")

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
            st.plotly_chart(fig_hist, use_container_width=True)

        with plot_col2:
            st.write("#### Q-Q Plot")
            fig_qq = plot_qq(selected_df, selected_column)
            st.plotly_chart(fig_qq, use_container_width=True)

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
# T-TESTS
# ------------------------------------------------------------

if st.session_state.step == "t_tests":

    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("T-Tests")

    st.info(
        "Use t-tests to compare means. This section includes one-sample, independent two-sample, "
        "and paired t-tests. Each test now includes assumption checks before running the test."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

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

            default_mean = float(
                pd.to_numeric(selected_df[numeric_column], errors="coerce").mean()
            )

            hypothesized_mean = st.number_input(
                "Enter hypothesized mean",
                value=default_mean,
                key="hypothesized_mean"
            )

            assumption_result = None

            try:
                assumption_result = check_one_sample_ttest_assumptions(
                    selected_df,
                    numeric_column,
                    alpha
                )

                render_assumption_panel(
                    assumption_result,
                    panel_key="one_sample_ttest_assumptions"
                )

            except Exception as error:
                st.error(f"Could not run assumption checks: {error}")

            run_disabled = (
                assumption_result is not None
                and assumption_result["recommendation_type"] == "error"
            )

            if st.button(
                "Run One-sample t-test",
                type="primary",
                disabled=run_disabled,
                key="run_one_sample_ttest"
            ):
                try:
                    result = run_one_sample_ttest(
                        selected_df,
                        numeric_column,
                        hypothesized_mean,
                        alpha
                    )

                    result_table = create_ttest_result_table(result)
                    interpretation = get_ttest_interpretation(result)

                    st.divider()
                    st.write("### Test Result")

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Visualization")
                        fig = plot_one_sample_ttest(
                            selected_df,
                            numeric_column,
                            hypothesized_mean
                        )
                        st.plotly_chart(fig, use_container_width=True)

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
                if col != numeric_column
                and selected_df[col].nunique(dropna=True) >= 2
                and selected_df[col].nunique(dropna=True) <= 30
            ]

            if len(possible_group_columns) == 0:
                st.warning(
                    "No suitable grouping columns found. "
                    "A grouping column should have a small number of categories."
                )

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

                assumption_result = None

                if group1 == group2:
                    st.warning("Group 1 and Group 2 must be different.")

                else:
                    try:
                        assumption_result = check_independent_ttest_assumptions(
                            selected_df,
                            numeric_column,
                            group_column,
                            group1,
                            group2,
                            alpha
                        )

                        render_assumption_panel(
                            assumption_result,
                            panel_key="independent_ttest_assumptions"
                        )

                    except Exception as error:
                        st.error(f"Could not run assumption checks: {error}")

                run_disabled = (
                    group1 == group2
                    or (
                        assumption_result is not None
                        and assumption_result["recommendation_type"] == "error"
                    )
                )

                if st.button(
                    "Run Independent t-test",
                    type="primary",
                    disabled=run_disabled,
                    key="run_independent_ttest"
                ):
                    try:
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

                        st.divider()
                        st.write("### Test Result")

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
                            st.plotly_chart(fig, use_container_width=True)

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
                "Use this when the two numerical columns are related, such as before-after "
                "measurements from the same subjects."
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

            assumption_result = None

            if before_column == after_column:
                st.warning("Before and after columns should be different.")

            else:
                try:
                    assumption_result = check_paired_ttest_assumptions(
                        selected_df,
                        before_column,
                        after_column,
                        alpha
                    )

                    render_assumption_panel(
                        assumption_result,
                        panel_key="paired_ttest_assumptions"
                    )

                except Exception as error:
                    st.error(f"Could not run assumption checks: {error}")

            run_disabled = (
                before_column == after_column
                or (
                    assumption_result is not None
                    and assumption_result["recommendation_type"] == "error"
                )
            )

            if st.button(
                "Run Paired t-test",
                type="primary",
                disabled=run_disabled,
                key="run_paired_ttest"
            ):
                try:
                    result = run_paired_ttest(
                        selected_df,
                        before_column,
                        after_column,
                        alpha
                    )

                    result_table = create_ttest_result_table(result)
                    interpretation = get_ttest_interpretation(result)

                    st.divider()
                    st.write("### Test Result")

                    plot_col, interpretation_col = st.columns([1.2, 1])

                    with plot_col:
                        st.write("#### Visualization")
                        fig = plot_paired_ttest(
                            selected_df,
                            before_column,
                            after_column
                        )
                        st.plotly_chart(fig, use_container_width=True)

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

        st.caption(
            "The assumption checks help decide whether the selected t-test is suitable. "
            "If assumptions are weak, consider the recommended non-parametric alternative."
        )

# ------------------------------------------------------------
# NON-PARAMETRIC TESTS
# ------------------------------------------------------------

elif st.session_state.step == "nonparametric_tests":

    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("Non-parametric Tests")

    st.info(
        "Non-parametric tests are useful when parametric assumptions such as normality are weak. "
        "These tests use ranks instead of relying directly on means and normal distributions."
    )

    numerical_columns = get_numerical_columns(selected_df)

    categorical_columns = [
        col for col in selected_df.columns
        if selected_df[col].nunique(dropna=True) >= 2
        and selected_df[col].nunique(dropna=True) <= 30
    ]

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

    else:
        alpha = st.selectbox(
            "Choose significance level (alpha)",
            [0.01, 0.05, 0.10],
            index=1,
            key="nonparametric_alpha"
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Mann-Whitney U",
                "Wilcoxon signed-rank",
                "Kruskal-Wallis",
                "Friedman"
            ]
        )

        # ----------------------------------------------------
        # Mann-Whitney U Test
        # ----------------------------------------------------
        with tab1:
            st.write("### Mann-Whitney U Test")

            st.write(
                "Use this as a non-parametric alternative to the independent two-sample t-test. "
                "It compares two independent groups using ranks."
            )

            if len(categorical_columns) == 0:
                st.warning("Mann-Whitney U needs one numerical column and one grouping column.")

            else:
                numeric_column = st.selectbox(
                    "Choose numerical outcome column",
                    numerical_columns,
                    key="mw_numeric_column"
                )

                possible_group_columns = [
                    col for col in categorical_columns
                    if col != numeric_column
                ]

                group_column = st.selectbox(
                    "Choose grouping column",
                    possible_group_columns,
                    key="mw_group_column"
                )

                group_values = selected_df[group_column].dropna().astype(str).unique().tolist()

                group1 = st.selectbox(
                    "Choose Group 1",
                    group_values,
                    key="mw_group1"
                )

                group2 = st.selectbox(
                    "Choose Group 2",
                    group_values,
                    key="mw_group2"
                )

                alternative = st.selectbox(
                    "Choose alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="mw_alternative"
                )

                if group1 == group2:
                    st.warning("Group 1 and Group 2 must be different.")

                if st.button("Run Mann-Whitney U Test", type="primary", key="run_mannwhitney"):
                    try:
                        if group1 == group2:
                            st.error("Please select two different groups.")
                        else:
                            result = run_mannwhitney_u_test(
                                selected_df,
                                numeric_column,
                                group_column,
                                group1,
                                group2,
                                alternative=alternative,
                                alpha=alpha
                            )

                            result_table = create_result_table(result)
                            interpretation = get_nonparametric_interpretation(result)

                            st.divider()
                            st.write("### Test Result")

                            plot_col, interpretation_col = st.columns([1.2, 1])

                            with plot_col:
                                fig = plot_mannwhitney_u(result)
                                st.plotly_chart(fig, use_container_width=True)

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
        # Wilcoxon Signed-Rank Test
        # ----------------------------------------------------
        with tab2:
            st.write("### Wilcoxon Signed-Rank Test")

            st.write(
                "Use this as a non-parametric alternative to the paired t-test. "
                "It compares two related numerical measurements using ranks of paired differences."
            )

            if len(numerical_columns) < 2:
                st.warning("Wilcoxon signed-rank test needs at least two numerical columns.")

            else:
                before_column = st.selectbox(
                    "Choose before / first measurement column",
                    numerical_columns,
                    key="wilcoxon_before_column"
                )

                after_options = [
                    col for col in numerical_columns
                    if col != before_column
                ]

                after_column = st.selectbox(
                    "Choose after / second measurement column",
                    after_options,
                    key="wilcoxon_after_column"
                )

                alternative = st.selectbox(
                    "Choose alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="wilcoxon_alternative"
                )

                if st.button("Run Wilcoxon Signed-Rank Test", type="primary", key="run_wilcoxon"):
                    try:
                        result = run_wilcoxon_signed_rank_test(
                            selected_df,
                            before_column,
                            after_column,
                            alternative=alternative,
                            alpha=alpha
                        )

                        result_table = create_result_table(result)
                        interpretation = get_nonparametric_interpretation(result)

                        st.divider()
                        st.write("### Test Result")

                        plot_col, interpretation_col = st.columns([1.2, 1])

                        with plot_col:
                            fig = plot_wilcoxon_signed_rank(result)
                            st.plotly_chart(fig, use_container_width=True)

                            fig_diff = plot_wilcoxon_differences(result)
                            st.plotly_chart(fig_diff, use_container_width=True)

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
        # Kruskal-Wallis Test
        # ----------------------------------------------------
        with tab3:
            st.write("### Kruskal-Wallis Test")

            st.write(
                "Use this as a non-parametric alternative to one-way ANOVA. "
                "It compares three or more independent groups using ranks."
            )

            if len(categorical_columns) == 0:
                st.warning("Kruskal-Wallis needs one numerical column and one grouping column.")

            else:
                numeric_column = st.selectbox(
                    "Choose numerical outcome column",
                    numerical_columns,
                    key="kw_numeric_column"
                )

                possible_group_columns = [
                    col for col in categorical_columns
                    if col != numeric_column
                ]

                group_column = st.selectbox(
                    "Choose grouping column",
                    possible_group_columns,
                    key="kw_group_column"
                )

                all_groups = selected_df[group_column].dropna().astype(str).unique().tolist()

                selected_groups = st.multiselect(
                    "Choose groups to include",
                    all_groups,
                    default=all_groups[: min(len(all_groups), 5)],
                    key="kw_selected_groups"
                )

                if len(selected_groups) < 2:
                    st.warning("Please select at least two groups.")

                if st.button("Run Kruskal-Wallis Test", type="primary", key="run_kruskal"):
                    try:
                        if len(selected_groups) < 2:
                            st.error("Kruskal-Wallis test requires at least two groups.")
                        else:
                            result = run_kruskal_wallis_test(
                                selected_df,
                                numeric_column,
                                group_column,
                                selected_groups=selected_groups,
                                alpha=alpha
                            )

                            result_table = create_result_table(result)
                            interpretation = get_nonparametric_interpretation(result)

                            st.divider()
                            st.write("### Test Result")

                            plot_col, interpretation_col = st.columns([1.2, 1])

                            with plot_col:
                                fig = plot_kruskal_wallis(result)
                                st.plotly_chart(fig, use_container_width=True)

                            with interpretation_col:
                                with st.container(border=True):
                                    st.write("#### Interpretation")
                                    for line in interpretation:
                                        st.write(f"- {line}")

                            st.write("#### Result Table")
                            st.dataframe(result_table, use_container_width=True)

                            st.write("#### Group Summary")
                            st.dataframe(result["Group Summary"], use_container_width=True)

                            if result["Decision"] == "Reject H0":
                                st.warning(
                                    "Kruskal-Wallis says at least one group differs, but it does not say which pair differs. "
                                    "A post-hoc test would be needed for pairwise comparison."
                                )

                    except Exception as error:
                        st.error(error)

        # ----------------------------------------------------
        # Friedman Test
        # ----------------------------------------------------
        with tab4:
            st.write("### Friedman Test")

            st.write(
                "Use this for three or more related/repeated numerical measurements. "
                "It is the non-parametric alternative to repeated-measures ANOVA."
            )

            if len(numerical_columns) < 3:
                st.warning("Friedman test needs at least three numerical measurement columns.")

            else:
                measurement_columns = st.multiselect(
                    "Choose three or more related measurement columns",
                    numerical_columns,
                    default=numerical_columns[:3],
                    key="friedman_measurement_columns"
                )

                if len(measurement_columns) < 3:
                    st.warning("Please select at least three related measurement columns.")

                if st.button("Run Friedman Test", type="primary", key="run_friedman"):
                    try:
                        if len(measurement_columns) < 3:
                            st.error("Friedman test requires at least three measurement columns.")
                        else:
                            result = run_friedman_test(
                                selected_df,
                                measurement_columns,
                                alpha=alpha
                            )

                            result_table = create_result_table(result)
                            interpretation = get_nonparametric_interpretation(result)

                            st.divider()
                            st.write("### Test Result")

                            plot_col, interpretation_col = st.columns([1.2, 1])

                            with plot_col:
                                fig = plot_friedman_test(result)
                                st.plotly_chart(fig, use_container_width=True)

                            with interpretation_col:
                                with st.container(border=True):
                                    st.write("#### Interpretation")
                                    for line in interpretation:
                                        st.write(f"- {line}")

                            st.write("#### Result Table")
                            st.dataframe(result_table, use_container_width=True)

                            st.write("#### Measurement Summary")
                            st.dataframe(result["Group Summary"], use_container_width=True)

                            if result["Decision"] == "Reject H0":
                                st.warning(
                                    "Friedman test says at least one repeated measurement differs, "
                                    "but a post-hoc test is needed to identify where the difference occurs."
                                )

                    except Exception as error:
                        st.error(error)

        st.divider()

        st.caption(
            "Non-parametric tests are especially useful when normality assumptions are weak, "
            "when data has outliers, or when the data is ordinal/rank-based."
        )


# ------------------------------------------------------------
# STEP 8: ANOVA
# ------------------------------------------------------------

if st.session_state.step == "anova":


    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("ANOVA Tests")

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
                        st.plotly_chart(fig, use_container_width=True)

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
                            st.plotly_chart(fig, use_container_width=True)

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


    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()


    selected_df = st.session_state.selected_df

    st.subheader("Chi-Square Tests")

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
                            st.plotly_chart(fig, use_container_width=True)

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
                        st.plotly_chart(fig, use_container_width=True)

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

    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("Z-Tests")

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
                        st.plotly_chart(fig, use_container_width=True)

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
                                st.plotly_chart(fig, use_container_width=True)

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
                        st.plotly_chart(fig, use_container_width=True)

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
                            st.plotly_chart(fig, use_container_width=True)

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


    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("Distribution Fitting")

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
                st.plotly_chart(fig, use_container_width=True)

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
            if st.button("Continue to CLT Simulation", type="primary"):
                st.session_state.step = "clt_simulation"
                st.rerun()


# ------------------------------------------------------------
# STEP 12: Central Limit Theorem Simulation
# ------------------------------------------------------------

if st.session_state.step == "clt_simulation":


    if st.session_state.selected_df is None:
        page_locked_message()
        st.stop()

    selected_df = st.session_state.selected_df

    st.subheader("Central Limit Theorem Simulation")

    st.info(
        "The Central Limit Theorem says that the sampling distribution of the sample mean becomes approximately normal "
        "as the sample size increases, even if the original data is not normally distributed."
    )

    numerical_columns = get_numerical_columns(selected_df)

    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the selected dataset.")

        if st.button("Back to Distribution Fitting"):
            st.session_state.step = "distribution_fitting"
            st.rerun()

    else:
        selected_column = st.selectbox(
            "Choose numerical column for CLT simulation",
            numerical_columns,
            key="clt_column"
        )

        st.write("### Simulation Settings")

        col_set1, col_set2, col_set3 = st.columns(3)

        with col_set1:
            sample_size = st.slider(
                "Sample size",
                min_value=2,
                max_value=200,
                value=30,
                key="clt_sample_size"
            )

        with col_set2:
            number_of_samples = st.slider(
                "Number of repeated samples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key="clt_number_of_samples"
            )

        with col_set3:
            random_seed = st.number_input(
                "Random seed",
                value=42,
                step=1,
                key="clt_random_seed"
            )

        bins = st.slider(
            "Histogram bins",
            min_value=10,
            max_value=60,
            value=30,
            key="clt_bins"
        )

        if st.button("Run CLT Simulation", type="primary"):
            try:
                data, sample_means = simulate_sample_means(
                    selected_df,
                    selected_column,
                    sample_size=sample_size,
                    number_of_samples=number_of_samples,
                    random_seed=int(random_seed)
                )

                st.session_state.clt_data = data
                st.session_state.clt_sample_means = sample_means
                st.session_state.clt_column_name = selected_column
                st.session_state.clt_sample_size_used = sample_size
                st.session_state.clt_number_of_samples_used = number_of_samples
                st.session_state.clt_bins_used = bins

            except Exception as error:
                st.error(error)

        if "clt_sample_means" in st.session_state:

            data = st.session_state.clt_data
            sample_means = st.session_state.clt_sample_means
            column_name = st.session_state.clt_column_name
            sample_size_used = st.session_state.clt_sample_size_used
            number_of_samples_used = st.session_state.clt_number_of_samples_used
            bins_used = st.session_state.clt_bins_used

            original_mean = data.mean()
            original_std = data.std(ddof=1)
            theoretical_standard_error = original_std / np.sqrt(sample_size_used)

            st.write(f"### CLT Results for `{column_name}`")

            plot_col, interpretation_col = st.columns([1.2, 1])

            with plot_col:
                st.write("#### Original Data Distribution")
                fig_original = plot_original_distribution(
                    data,
                    column_name,
                    bins=bins_used
                )
                st.pyplot(fig_original)

            with interpretation_col:
                with st.container(border=True):
                    st.write("#### What this original plot shows")
                    st.write(
                        "- This is the distribution of the actual selected column."
                    )
                    st.write(
                        "- It may be normal, skewed, flat, or irregular."
                    )
                    st.write(
                        "- CLT does not require the original distribution to be perfectly normal."
                    )

            plot_col2, interpretation_col2 = st.columns([1.2, 1])

            with plot_col2:
                st.write("#### Sampling Distribution of Sample Means")
                fig_sampling = plot_sampling_distribution(
                    sample_means,
                    original_mean,
                    theoretical_standard_error,
                    bins=bins_used
                )
                st.pyplot(fig_sampling)

            with interpretation_col2:
                with st.container(border=True):
                    st.write("#### CLT Interpretation")

                    interpretation = get_clt_interpretation(
                        data,
                        sample_means,
                        sample_size_used,
                        number_of_samples_used
                    )

                    for line in interpretation:
                        st.write(f"- {line}")

            st.write("#### CLT Summary Table")

            summary_table = create_clt_summary_table(
                data,
                sample_means,
                sample_size_used
            )

            st.dataframe(summary_table, use_container_width=True)

            st.divider()

            st.write("### Normality Check on Sample Means")

            shapiro_result = run_normality_check_on_sample_means(
                sample_means,
                alpha=0.05
            )

            shapiro_table = create_sample_means_normality_table(shapiro_result)

            st.dataframe(shapiro_table, use_container_width=True)

            st.caption(
                "This normality test is performed on the simulated sample means, not on the original data."
            )

            st.divider()

            st.write("### Compare Different Sample Sizes")

            st.caption(
                "This shows how the sampling distribution changes when sample size increases."
            )

            sample_size_options = st.multiselect(
                "Choose sample sizes to compare",
                [2, 5, 10, 20, 30, 50, 100],
                default=[5, 30, 100],
                key="clt_compare_sample_sizes"
            )

            comparison_samples = st.slider(
                "Number of samples for comparison plot",
                min_value=500,
                max_value=5000,
                value=1000,
                step=500,
                key="clt_comparison_samples"
            )

            if st.button("Run Sample Size Comparison"):
                try:
                    _, sample_size_results = simulate_multiple_sample_sizes(
                        selected_df,
                        column_name,
                        sample_size_options,
                        number_of_samples=comparison_samples,
                        random_seed=int(random_seed)
                    )

                    fig_comparison = plot_sample_size_comparison(sample_size_results)
                    st.pyplot(fig_comparison)

                    st.write("#### How to read this comparison")
                    st.write(
                        "- Larger sample sizes usually create a narrower sampling distribution."
                    )
                    st.write(
                        "- This happens because the standard error decreases as sample size increases."
                    )
                    st.write(
                        "- In simple terms, larger samples give more stable sample means."
                    )

                except Exception as error:
                    st.error(error)

        st.divider()

        col_back, col_finish = st.columns([1, 2])

        with col_back:
            if st.button("Back to Distribution Fitting"):
                st.session_state.step = "distribution_fitting"
                st.rerun()

        with col_finish:
            st.button("Toolkit Complete", type="primary")