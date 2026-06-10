import pandas as pd
from datetime import datetime


def safe_value(value):
    """
    Safely formats values for report text.
    """

    try:
        if pd.isna(value):
            return "N/A"

        if isinstance(value, float):
            return round(value, 6)

        return value

    except Exception:
        return value


def create_dataset_summary(selected_df):
    """
    Creates dataset summary text.
    """

    rows, columns = selected_df.shape

    numeric_columns = selected_df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        col for col in selected_df.columns
        if col not in numeric_columns
    ]

    missing_values = selected_df.isna().sum().sum()

    report = []

    report.append("## Dataset Summary")
    report.append("")
    report.append(f"- Number of rows: **{rows}**")
    report.append(f"- Number of columns: **{columns}**")
    report.append(f"- Numerical columns: **{len(numeric_columns)}**")
    report.append(f"- Non-numerical columns: **{len(categorical_columns)}**")
    report.append(f"- Total missing values: **{missing_values}**")
    report.append("")

    report.append("### Selected Columns")
    report.append("")

    for col in selected_df.columns:
        report.append(f"- `{col}`")

    report.append("")

    return "\n".join(report)


def create_distribution_fitting_section(session_state):
    """
    Adds distribution fitting results if available.
    """

    if "dist_fit_results" not in session_state:
        return ""

    results = session_state.dist_fit_results
    fitted_column = session_state.get("dist_fit_column_name", "Unknown column")
    bins_used = session_state.get("dist_fit_bins_used", "Unknown")

    report = []

    report.append("## Distribution Fitting")
    report.append("")
    report.append(f"- Fitted column: **{fitted_column}**")
    report.append(f"- Histogram bins used: **{bins_used}**")
    report.append("")

    if len(results) > 0:
        best = results[0]

        report.append("### Best Fitting Distribution")
        report.append("")
        report.append(f"- Distribution: **{best.get('Distribution', 'N/A')}**")
        report.append(f"- SSE: **{safe_value(best.get('SSE'))}**")
        report.append(f"- MSE: **{safe_value(best.get('MSE'))}**")
        report.append(f"- RMSE: **{safe_value(best.get('RMSE'))}**")
        report.append("")

        report.append("### Ranking Table")
        report.append("")
        report.append("| Rank | Distribution | SSE | MSE | RMSE |")
        report.append("|---:|---|---:|---:|---:|")

        for result in results:
            report.append(
                f"| {result.get('Rank', '')} "
                f"| {result.get('Distribution', '')} "
                f"| {safe_value(result.get('SSE'))} "
                f"| {safe_value(result.get('MSE'))} "
                f"| {safe_value(result.get('RMSE'))} |"
            )

        report.append("")

    return "\n".join(report)


def create_clt_section(session_state):
    """
    Adds CLT simulation results if available.
    """

    if "clt_sample_means" not in session_state:
        return ""

    data = session_state.clt_data
    sample_means = session_state.clt_sample_means

    column_name = session_state.get("clt_column_name", "Unknown column")
    sample_size = session_state.get("clt_sample_size_used", "Unknown")
    number_of_samples = session_state.get("clt_number_of_samples_used", "Unknown")

    original_mean = data.mean()
    original_std = data.std(ddof=1)

    sample_means_mean = sample_means.mean()
    sample_means_std = sample_means.std(ddof=1)

    report = []

    report.append("## Central Limit Theorem Simulation")
    report.append("")
    report.append(f"- Column used: **{column_name}**")
    report.append(f"- Sample size: **{sample_size}**")
    report.append(f"- Number of repeated samples: **{number_of_samples}**")
    report.append("")
    report.append("### CLT Summary")
    report.append("")
    report.append(f"- Original data mean: **{safe_value(original_mean)}**")
    report.append(f"- Original data standard deviation: **{safe_value(original_std)}**")
    report.append(f"- Mean of sample means: **{safe_value(sample_means_mean)}**")
    report.append(f"- Standard deviation of sample means: **{safe_value(sample_means_std)}**")
    report.append("")
    report.append(
        "The CLT simulation shows how repeated sample means become more stable and usually more normal-shaped as sample size increases."
    )
    report.append("")

    return "\n".join(report)


def generate_toolkit_report(selected_df, session_state):
    """
    Generates a Markdown report for the current toolkit session.
    """

    report = []

    report.append("# Statistical Toolkit Report")
    report.append("")
    report.append(f"Generated on: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    report.append("")

    report.append(
        "This report summarizes the dataset and the completed analysis sections available in the current Streamlit session."
    )
    report.append("")

    report.append(create_dataset_summary(selected_df))

    distribution_section = create_distribution_fitting_section(session_state)

    if distribution_section:
        report.append(distribution_section)

    clt_section = create_clt_section(session_state)

    if clt_section:
        report.append(clt_section)

    report.append("## Notes")
    report.append("")
    report.append("- Statistical results should be interpreted together with assumptions and visual diagnostics.")
    report.append("- A low p-value suggests statistical evidence against the null hypothesis, but it does not automatically mean practical importance.")
    report.append("- Visualizations, sample size, assumptions, and context should always be considered.")
    report.append("")

    return "\n".join(report)