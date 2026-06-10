import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go


PLOT_COLORS = {
    "primary": "#56B4E9",      # Sky blue
    "secondary": "#D55E00",    # Vermillion
    "green": "#009E73",        # Bluish green
    "warning": "#E69F00",      # Warm orange
    "error": "#D55E00",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
}

COLOR_SEQUENCE = [
    PLOT_COLORS["primary"],
    PLOT_COLORS["secondary"],
    PLOT_COLORS["green"],
    PLOT_COLORS["warning"],
]


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=430):
    """
    Applies a clean, light, professional Plotly theme.
    """

    fig.update_layout(
        title={
            "text": title if title else "",
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 18,
                "color": PLOT_COLORS["text"],
            },
        },
        paper_bgcolor=PLOT_COLORS["bg"],
        plot_bgcolor=PLOT_COLORS["card"],
        font={
            "color": PLOT_COLORS["text"],
            "family": "Arial",
        },
        height=height,
        margin={"l": 60, "r": 30, "t": 65, "b": 55},
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": PLOT_COLORS["muted"]},
        },
    )

    fig.update_xaxes(
        title_text=x_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["grid"],
        linecolor=PLOT_COLORS["axis"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
        mirror=False,
        showline=True,
    )

    fig.update_yaxes(
        title_text=y_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["grid"],
        linecolor=PLOT_COLORS["axis"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
        mirror=False,
        showline=True,
    )

    return fig


def format_number(value):
    """
    Formats numbers nicely for tables and interpretations.
    """

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if value == 0:
            return "0"

        if abs(value) < 0.00001:
            return f"{value:.2e}"

        return round(value, 5)

    except Exception:
        return value


def interpret_p_value(p_value, alpha=0.05):
    """
    Interprets p-value using alpha.
    """

    if p_value < alpha:
        return "Reject H0", "There is a statistically significant result."

    return "Fail to Reject H0", "There is not enough evidence for a statistically significant result."


def prepare_numeric_data(df, column):
    """
    Converts selected column to numeric and removes missing values.
    """

    data = pd.to_numeric(df[column], errors="coerce").dropna()

    if data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    return data


def two_tailed_p_value_from_z(z_statistic):
    """
    Calculates two-tailed p-value from z-statistic.
    """

    return 2 * (1 - stats.norm.cdf(abs(z_statistic)))


# ------------------------------------------------------------
# One-sample mean z-test
# ------------------------------------------------------------

def run_one_sample_ztest(df, numeric_column, hypothesized_mean, population_std, alpha=0.05):
    """
    One-sample mean z-test.

    H0: Population mean is equal to hypothesized mean.
    H1: Population mean is not equal to hypothesized mean.
    """

    data = prepare_numeric_data(df, numeric_column)

    if len(data) < 2:
        raise ValueError("One-sample z-test requires at least 2 valid numerical values.")

    if population_std <= 0:
        raise ValueError("Population standard deviation must be greater than 0.")

    sample_size = len(data)
    sample_mean = data.mean()
    standard_error = population_std / np.sqrt(sample_size)

    z_statistic = (sample_mean - hypothesized_mean) / standard_error
    p_value = two_tailed_p_value_from_z(z_statistic)

    decision, conclusion = interpret_p_value(p_value, alpha)

    return {
        "Test": "One-sample mean z-test",
        "H0": f"The population mean of {numeric_column} is equal to {hypothesized_mean}.",
        "H1": f"The population mean of {numeric_column} is not equal to {hypothesized_mean}.",
        "Sample Size": sample_size,
        "Sample Mean": sample_mean,
        "Hypothesized Mean": hypothesized_mean,
        "Population Standard Deviation": population_std,
        "Standard Error": standard_error,
        "Mean Difference": sample_mean - hypothesized_mean,
        "z-statistic": z_statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
    }


def plot_one_sample_ztest(df, numeric_column, hypothesized_mean):
    """
    Plotly histogram with sample mean and hypothesized mean.
    """

    data = prepare_numeric_data(df, numeric_column)
    sample_mean = data.mean()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=25,
            marker={
                "color": PLOT_COLORS["primary"],
                "line": {"color": "#FFFFFF", "width": 1},
            },
            opacity=0.88,
            name="Observed values",
            hovertemplate=f"{numeric_column}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    fig.add_vline(
        x=sample_mean,
        line_width=3,
        line_dash="dash",
        line_color=PLOT_COLORS["green"],
        annotation_text="Sample Mean",
        annotation_position="top left",
    )

    fig.add_vline(
        x=hypothesized_mean,
        line_width=3,
        line_dash="dot",
        line_color=PLOT_COLORS["warning"],
        annotation_text="Hypothesized Mean",
        annotation_position="top right",
    )

    fig = apply_plotly_theme(
        fig,
        title=f"One-sample z-test: {numeric_column}",
        x_title=numeric_column,
        y_title="Frequency",
        height=430,
    )

    return fig


# ------------------------------------------------------------
# Two-sample mean z-test
# ------------------------------------------------------------

def run_two_sample_ztest(
    df,
    numeric_column,
    group_column,
    group1,
    group2,
    population_std1,
    population_std2,
    alpha=0.05,
):
    """
    Two-sample mean z-test.

    H0: The two population means are equal.
    H1: The two population means are not equal.
    """

    group1_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group1)][numeric_column],
        errors="coerce",
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group2)][numeric_column],
        errors="coerce",
    ).dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Each group must have at least 2 valid numerical values.")

    if population_std1 <= 0 or population_std2 <= 0:
        raise ValueError("Population standard deviations must be greater than 0.")

    n1 = len(group1_data)
    n2 = len(group2_data)

    mean1 = group1_data.mean()
    mean2 = group2_data.mean()

    standard_error = np.sqrt((population_std1 ** 2 / n1) + (population_std2 ** 2 / n2))
    z_statistic = (mean1 - mean2) / standard_error
    p_value = two_tailed_p_value_from_z(z_statistic)

    decision, conclusion = interpret_p_value(p_value, alpha)

    return {
        "Test": "Two-sample mean z-test",
        "H0": f"The population mean of {numeric_column} is equal for {group1} and {group2}.",
        "H1": f"The population mean of {numeric_column} is different between {group1} and {group2}.",
        "Numeric Column": numeric_column,
        "Group Column": group_column,
        "Group 1": group1,
        "Group 2": group2,
        "Group 1 Sample Size": n1,
        "Group 2 Sample Size": n2,
        "Group 1 Mean": mean1,
        "Group 2 Mean": mean2,
        "Mean Difference": mean1 - mean2,
        "Population Std Group 1": population_std1,
        "Population Std Group 2": population_std2,
        "Standard Error": standard_error,
        "z-statistic": z_statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
    }


def plot_two_sample_ztest(df, numeric_column, group_column, group1, group2):
    """
    Plotly boxplot comparing two groups.
    """

    group1_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group1)][numeric_column],
        errors="coerce",
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group2)][numeric_column],
        errors="coerce",
    ).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=group1_data,
            name=str(group1),
            marker_color=PLOT_COLORS["primary"],
            line_color=PLOT_COLORS["primary"],
            fillcolor="rgba(86, 180, 233, 0.28)",
            boxmean=True,
            hovertemplate=f"{group1}<br>{numeric_column}: %{{y}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Box(
            y=group2_data,
            name=str(group2),
            marker_color=PLOT_COLORS["secondary"],
            line_color=PLOT_COLORS["secondary"],
            fillcolor="rgba(213, 94, 0, 0.24)",
            boxmean=True,
            hovertemplate=f"{group2}<br>{numeric_column}: %{{y}}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Two-sample z-test: {numeric_column} by {group_column}",
        x_title=group_column,
        y_title=numeric_column,
        height=430,
    )

    return fig


# ------------------------------------------------------------
# One-proportion z-test
# ------------------------------------------------------------

def run_one_proportion_ztest(df, categorical_column, success_category, hypothesized_proportion, alpha=0.05):
    """
    One-proportion z-test.

    H0: Population proportion is equal to hypothesized proportion.
    H1: Population proportion is not equal to hypothesized proportion.
    """

    data = df[categorical_column].dropna().astype(str)

    if data.empty:
        raise ValueError(f"Column '{categorical_column}' does not contain valid categorical data.")

    if hypothesized_proportion <= 0 or hypothesized_proportion >= 1:
        raise ValueError("Hypothesized proportion must be between 0 and 1.")

    success_category = str(success_category)

    n = len(data)
    successes = int((data == success_category).sum())

    sample_proportion = successes / n
    standard_error = np.sqrt((hypothesized_proportion * (1 - hypothesized_proportion)) / n)

    z_statistic = (sample_proportion - hypothesized_proportion) / standard_error
    p_value = two_tailed_p_value_from_z(z_statistic)

    decision, conclusion = interpret_p_value(p_value, alpha)

    return {
        "Test": "One-proportion z-test",
        "H0": f"The population proportion of `{success_category}` in {categorical_column} is equal to {hypothesized_proportion}.",
        "H1": f"The population proportion of `{success_category}` in {categorical_column} is not equal to {hypothesized_proportion}.",
        "Categorical Column": categorical_column,
        "Success Category": success_category,
        "Sample Size": n,
        "Success Count": successes,
        "Sample Proportion": sample_proportion,
        "Hypothesized Proportion": hypothesized_proportion,
        "Standard Error": standard_error,
        "z-statistic": z_statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
    }


def plot_one_proportion_ztest(result):
    """
    Plotly bar chart comparing sample proportion and hypothesized proportion.
    """

    labels = ["Sample Proportion", "Hypothesized Proportion"]
    values = [result["Sample Proportion"], result["Hypothesized Proportion"]]
    text_values = [format_number(value) for value in values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=text_values,
            textposition="outside",
            marker={
                "color": [PLOT_COLORS["primary"], PLOT_COLORS["warning"]],
                "line": {"color": "#FFFFFF", "width": 1},
            },
            hovertemplate="Type: %{x}<br>Proportion: %{y:.5f}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title="Sample vs Hypothesized Proportion",
        x_title="Proportion type",
        y_title="Proportion",
        height=430,
    )

    fig.update_yaxes(range=[0, 1.08])
    fig.update_xaxes(type="category")

    return fig


# ------------------------------------------------------------
# Two-proportion z-test
# ------------------------------------------------------------

def run_two_proportion_ztest(df, outcome_column, success_category, group_column, group1, group2, alpha=0.05):
    """
    Two-proportion z-test.

    H0: The two population proportions are equal.
    H1: The two population proportions are not equal.
    """

    clean_df = df[[outcome_column, group_column]].copy()
    clean_df = clean_df.dropna()

    clean_df[outcome_column] = clean_df[outcome_column].astype(str)
    clean_df[group_column] = clean_df[group_column].astype(str)

    group1 = str(group1)
    group2 = str(group2)
    success_category = str(success_category)

    group1_df = clean_df[clean_df[group_column] == group1]
    group2_df = clean_df[clean_df[group_column] == group2]

    n1 = len(group1_df)
    n2 = len(group2_df)

    if n1 == 0 or n2 == 0:
        raise ValueError("Both selected groups must contain data.")

    success1 = int((group1_df[outcome_column] == success_category).sum())
    success2 = int((group2_df[outcome_column] == success_category).sum())

    p1 = success1 / n1
    p2 = success2 / n2

    pooled_proportion = (success1 + success2) / (n1 + n2)
    standard_error = np.sqrt(
        pooled_proportion * (1 - pooled_proportion) * ((1 / n1) + (1 / n2))
    )

    if standard_error == 0:
        raise ValueError("Standard error is 0, so the z-test cannot be calculated.")

    z_statistic = (p1 - p2) / standard_error
    p_value = two_tailed_p_value_from_z(z_statistic)

    decision, conclusion = interpret_p_value(p_value, alpha)

    return {
        "Test": "Two-proportion z-test",
        "H0": f"The proportion of `{success_category}` is equal for {group1} and {group2}.",
        "H1": f"The proportion of `{success_category}` is different between {group1} and {group2}.",
        "Outcome Column": outcome_column,
        "Success Category": success_category,
        "Group Column": group_column,
        "Group 1": group1,
        "Group 2": group2,
        "Group 1 Sample Size": n1,
        "Group 2 Sample Size": n2,
        "Group 1 Success Count": success1,
        "Group 2 Success Count": success2,
        "Group 1 Proportion": p1,
        "Group 2 Proportion": p2,
        "Proportion Difference": p1 - p2,
        "Pooled Proportion": pooled_proportion,
        "Standard Error": standard_error,
        "z-statistic": z_statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
    }


def plot_two_proportion_ztest(result):
    """
    Plotly bar chart comparing two sample proportions.
    """

    labels = [str(result["Group 1"]), str(result["Group 2"])]
    values = [result["Group 1 Proportion"], result["Group 2 Proportion"]]
    text_values = [format_number(value) for value in values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=text_values,
            textposition="outside",
            marker={
                "color": [PLOT_COLORS["primary"], PLOT_COLORS["secondary"]],
                "line": {"color": "#FFFFFF", "width": 1},
            },
            hovertemplate="Group: %{x}<br>Proportion: %{y:.5f}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title="Group Proportion Comparison",
        x_title=result["Group Column"],
        y_title=f"Proportion of {result['Success Category']}",
        height=430,
    )

    fig.update_yaxes(range=[0, 1.08])
    fig.update_xaxes(type="category")

    return fig


# ------------------------------------------------------------
# Shared output helpers
# ------------------------------------------------------------

def create_ztest_result_table(result):
    """
    Converts z-test result dictionary into a clean table.
    """

    hidden_keys = ["H0", "H1", "Conclusion"]

    rows = []

    for key, value in result.items():
        if key in hidden_keys:
            continue

        rows.append({
            "Metric": key,
            "Value": format_number(value),
        })

    return pd.DataFrame(rows)


def get_ztest_interpretation(result):
    """
    Creates plain-English interpretation for z-test result.
    """

    interpretation = []

    interpretation.append(f"**H0:** {result['H0']}")
    interpretation.append(f"**H1:** {result['H1']}")

    interpretation.append(
        f"The z-statistic is {format_number(result['z-statistic'])}."
    )

    interpretation.append(
        f"The p-value is {format_number(result['p-value'])}, and alpha is {result['Alpha']}."
    )

    if result["Decision"] == "Reject H0":
        interpretation.append("Because the p-value is less than alpha, we reject H0.")
    else:
        interpretation.append("Because the p-value is greater than or equal to alpha, we fail to reject H0.")

    interpretation.append(result["Conclusion"])

    if "Population Standard Deviation" in result:
        interpretation.append(
            "This test assumes the population standard deviation is known."
        )

    if "Population Std Group 1" in result:
        interpretation.append(
            "This test assumes the population standard deviations for both groups are known."
        )

    if "Sample Proportion" in result or "Group 1 Proportion" in result:
        interpretation.append(
            "For proportion z-tests, the sample size should be large enough for the normal approximation to be reasonable."
        )

    return interpretation
