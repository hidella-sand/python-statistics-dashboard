import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


CHART_PALETTE = ["#56B4E9", "#D55E00", "#009E73", "#E69F00"]

PLOT_COLORS = {
    "primary": "#56B4E9",      # sky blue
    "secondary": "#D55E00",    # vermillion
    "green": "#009E73",        # bluish green
    "warning": "#E69F00",      # warm orange
    "blue": "#56B4E9",
    "error": "#CC3311",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "grid": "#E5E7EB",
    "text": "#1F2937",
    "muted": "#6B7280",
    "border": "#D1D5DB",
}


def format_number(value, decimals=5):
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

        return round(value, decimals)

    except Exception:
        return value


def format_p_value(p_value):
    """
    Formats p-values in a clean academic way.
    """

    try:
        p_value = float(p_value)

        if p_value == 0:
            return "< 1e-300"

        if p_value < 0.00001:
            return f"{p_value:.2e}"

        return round(p_value, 5)

    except Exception:
        return p_value


def interpret_p_value(p_value, alpha=0.05):
    """
    General p-value interpretation.
    """
    if p_value < alpha:
        return "Reject H0", "There is a statistically significant result."
    else:
        return "Fail to Reject H0", "There is not enough evidence for a statistically significant result."


def prepare_numeric_data(df, column):
    """
    Converts a selected column into numeric data and removes missing values.
    """
    numeric_data = pd.to_numeric(df[column], errors="coerce").dropna()

    if numeric_data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    return numeric_data


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=420):
    """
    Applies a soft professional light theme to Plotly figures.
    """

    fig.update_layout(
        title={
            "text": title if title else "",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 18, "color": PLOT_COLORS["text"]},
        },
        paper_bgcolor=PLOT_COLORS["bg"],
        plot_bgcolor=PLOT_COLORS["card"],
        font={"color": PLOT_COLORS["text"], "family": "Arial"},
        height=height,
        margin={"l": 48, "r": 28, "t": 62, "b": 48},
        hovermode="closest",
        colorway=CHART_PALETTE,
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
        linecolor=PLOT_COLORS["border"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
        showline=True,
        linewidth=1,
    )

    fig.update_yaxes(
        title_text=y_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["grid"],
        linecolor=PLOT_COLORS["border"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
        showline=True,
        linewidth=1,
    )

    return fig


# ------------------------------------------------------------
# One-sample t-test
# ------------------------------------------------------------

def run_one_sample_ttest(df, numeric_column, hypothesized_mean, alpha=0.05):
    """
    One-sample t-test.

    H0: Population mean is equal to hypothesized mean.
    H1: Population mean is not equal to hypothesized mean.
    """

    data = prepare_numeric_data(df, numeric_column)

    if len(data) < 2:
        raise ValueError("One-sample t-test requires at least 2 valid numerical values.")

    t_statistic, p_value = stats.ttest_1samp(data, popmean=hypothesized_mean)

    sample_mean = data.mean()
    sample_std = data.std(ddof=1)
    mean_difference = sample_mean - hypothesized_mean
    df_value = len(data) - 1

    decision, conclusion = interpret_p_value(p_value, alpha)

    result = {
        "Test": "One-sample t-test",
        "H0": f"The mean of {numeric_column} is equal to {hypothesized_mean}.",
        "H1": f"The mean of {numeric_column} is not equal to {hypothesized_mean}.",
        "Sample Size": len(data),
        "Sample Mean": sample_mean,
        "Hypothesized Mean": hypothesized_mean,
        "Mean Difference": mean_difference,
        "Sample Standard Deviation": sample_std,
        "t-statistic": t_statistic,
        "Degrees of Freedom": df_value,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion
    }

    return result


def plot_one_sample_ttest(df, numeric_column, hypothesized_mean):
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
                "line": {"color": PLOT_COLORS["card"], "width": 1},
            },
            opacity=0.88,
            name="Observed values",
            hovertemplate=f"{numeric_column}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    fig.add_vline(
        x=sample_mean,
        line_width=2,
        line_dash="dash",
        line_color=PLOT_COLORS["green"],
        annotation_text="Sample Mean",
        annotation_position="top left",
    )

    fig.add_vline(
        x=hypothesized_mean,
        line_width=2,
        line_dash="dot",
        line_color=PLOT_COLORS["warning"],
        annotation_text="Hypothesized Mean",
        annotation_position="top right",
    )

    fig = apply_plotly_theme(
        fig,
        title=f"One-sample t-test: {numeric_column}",
        x_title=numeric_column,
        y_title="Frequency",
        height=420,
    )

    return fig


# ------------------------------------------------------------
# Independent two-sample t-test
# ------------------------------------------------------------

def calculate_welch_df(group1_data, group2_data):
    """
    Calculates Welch-Satterthwaite degrees of freedom.
    """

    n1 = len(group1_data)
    n2 = len(group2_data)

    s1_squared = group1_data.var(ddof=1)
    s2_squared = group2_data.var(ddof=1)

    numerator = ((s1_squared / n1) + (s2_squared / n2)) ** 2

    denominator = ((s1_squared / n1) ** 2 / (n1 - 1)) + ((s2_squared / n2) ** 2 / (n2 - 1))

    return numerator / denominator


def run_independent_ttest(df, numeric_column, group_column, group1, group2, alpha=0.05):
    """
    Independent two-sample t-test.

    H0: The two group means are equal.
    H1: The two group means are not equal.
    """

    group1_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group1)][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group2)][numeric_column],
        errors="coerce"
    ).dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Each group must have at least 2 valid numerical values.")

    levene_statistic, levene_p_value = stats.levene(group1_data, group2_data)

    equal_variance_assumed = levene_p_value >= alpha

    t_statistic, p_value = stats.ttest_ind(
        group1_data,
        group2_data,
        equal_var=equal_variance_assumed
    )

    if equal_variance_assumed:
        df_value = len(group1_data) + len(group2_data) - 2
        variance_note = "Levene's test did not reject equal variances, so Student's independent t-test was used."
    else:
        df_value = calculate_welch_df(group1_data, group2_data)
        variance_note = "Levene's test rejected equal variances, so Welch's t-test was used."

    mean_difference = group1_data.mean() - group2_data.mean()

    decision, conclusion = interpret_p_value(p_value, alpha)

    result = {
        "Test": "Independent two-sample t-test",
        "H0": f"The mean of {numeric_column} is equal for {group1} and {group2}.",
        "H1": f"The mean of {numeric_column} is different between {group1} and {group2}.",
        "Group Column": group_column,
        "Group 1": group1,
        "Group 2": group2,
        "Group 1 Sample Size": len(group1_data),
        "Group 2 Sample Size": len(group2_data),
        "Group 1 Mean": group1_data.mean(),
        "Group 2 Mean": group2_data.mean(),
        "Mean Difference": mean_difference,
        "Levene Statistic": levene_statistic,
        "Levene p-value": levene_p_value,
        "Equal Variance Assumed": equal_variance_assumed,
        "Variance Note": variance_note,
        "t-statistic": t_statistic,
        "Degrees of Freedom": df_value,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion
    }

    return result


def plot_independent_ttest(df, numeric_column, group_column, group1, group2):
    """
    Plotly boxplot comparing two independent groups.
    """

    group1_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group1)][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group2)][numeric_column],
        errors="coerce"
    ).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=group1_data,
            name=str(group1),
            marker_color=PLOT_COLORS["primary"],
            line_color=PLOT_COLORS["primary"],
            fillcolor="rgba(86, 180, 233, 0.24)",
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
            fillcolor="rgba(213, 94, 0, 0.22)",
            boxmean=True,
            hovertemplate=f"{group2}<br>{numeric_column}: %{{y}}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Independent t-test: {numeric_column} by {group_column}",
        x_title=group_column,
        y_title=numeric_column,
        height=420,
    )

    return fig


# ------------------------------------------------------------
# Paired t-test
# ------------------------------------------------------------

def run_paired_ttest(df, before_column, after_column, alpha=0.05):
    """
    Paired t-test.

    H0: The mean difference between paired values is 0.
    H1: The mean difference between paired values is not 0.
    """

    paired_df = df[[before_column, after_column]].copy()

    paired_df[before_column] = pd.to_numeric(paired_df[before_column], errors="coerce")
    paired_df[after_column] = pd.to_numeric(paired_df[after_column], errors="coerce")

    paired_df = paired_df.dropna()

    if len(paired_df) < 2:
        raise ValueError("Paired t-test requires at least 2 complete pairs.")

    before_data = paired_df[before_column]
    after_data = paired_df[after_column]
    differences = after_data - before_data

    t_statistic, p_value = stats.ttest_rel(before_data, after_data)

    decision, conclusion = interpret_p_value(p_value, alpha)

    result = {
        "Test": "Paired t-test",
        "H0": f"The mean difference between {before_column} and {after_column} is 0.",
        "H1": f"The mean difference between {before_column} and {after_column} is not 0.",
        "Number of Pairs": len(paired_df),
        "Before Mean": before_data.mean(),
        "After Mean": after_data.mean(),
        "Mean Difference": differences.mean(),
        "Standard Deviation of Differences": differences.std(ddof=1),
        "t-statistic": t_statistic,
        "Degrees of Freedom": len(paired_df) - 1,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion
    }

    return result


def plot_paired_ttest(df, before_column, after_column):
    """
    Plotly paired before-after line plot.
    """

    paired_df = df[[before_column, after_column]].copy()

    paired_df[before_column] = pd.to_numeric(paired_df[before_column], errors="coerce")
    paired_df[after_column] = pd.to_numeric(paired_df[after_column], errors="coerce")

    paired_df = paired_df.dropna()

    plot_df = paired_df.head(80)

    fig = go.Figure()

    for index, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[before_column, after_column],
                y=[row[before_column], row[after_column]],
                mode="lines+markers",
                line={"color": "rgba(86, 180, 233, 0.30)", "width": 1.5},
                marker={"size": 6, "color": PLOT_COLORS["primary"]},
                showlegend=False,
                hovertemplate="Measurement: %{x}<br>Value: %{y}<extra></extra>",
            )
        )

    before_mean = paired_df[before_column].mean()
    after_mean = paired_df[after_column].mean()

    fig.add_trace(
        go.Scatter(
            x=[before_column, after_column],
            y=[before_mean, after_mean],
            mode="lines+markers",
            line={"color": PLOT_COLORS["green"], "width": 4},
            marker={"size": 10},
            name="Mean change",
            hovertemplate="Measurement: %{x}<br>Mean: %{y:.4f}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Paired t-test: {before_column} vs {after_column}",
        x_title="Measurement",
        y_title="Value",
        height=440,
    )

    return fig


# ------------------------------------------------------------
# Shared output helpers
# ------------------------------------------------------------

def create_ttest_result_table(result):
    """
    Converts t-test result dictionary into a clean table.
    """

    rows = []

    hidden_keys = ["H0", "H1", "Conclusion", "Variance Note"]

    for key, value in result.items():
        if key in hidden_keys:
            continue

        if key == "p-value":
            formatted_value = format_p_value(value)
        elif key == "Levene p-value":
            formatted_value = format_p_value(value)
        else:
            formatted_value = format_number(value)

        rows.append({
            "Metric": key,
            "Value": formatted_value
        })

    return pd.DataFrame(rows)


def get_ttest_interpretation(result):
    """
    Creates a plain-English interpretation for t-test result.
    """

    interpretation = []

    interpretation.append(f"**H0:** {result['H0']}")
    interpretation.append(f"**H1:** {result['H1']}")

    interpretation.append(
        f"The p-value is {format_p_value(result['p-value'])}, and alpha is {result['Alpha']}."
    )

    if result["Decision"] == "Reject H0":
        interpretation.append(
            "Because p-value is less than alpha, we reject H0."
        )
    else:
        interpretation.append(
            "Because p-value is greater than or equal to alpha, we fail to reject H0."
        )

    interpretation.append(result["Conclusion"])

    if "Variance Note" in result:
        interpretation.append(result["Variance Note"])

    return interpretation