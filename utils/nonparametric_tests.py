import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


CHART_PALETTE = ["#56B4E9", "#D55E00", "#009E73", "#E69F00"]

PLOT_COLORS = {
    "primary": "#56B4E9",       # sky blue
    "secondary": "#D55E00",     # vermillion
    "green": "#009E73",         # bluish green
    "warning": "#E69F00",       # warm orange
    "error": "#D55E00",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "grid": "#D9E2EC",
    "axis": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
    "bar_line": "#FFFFFF",
    "soft_primary": "rgba(86, 180, 233, 0.24)",
    "soft_secondary": "rgba(213, 94, 0, 0.20)",
    "soft_green": "rgba(0, 158, 115, 0.20)",
    "soft_warning": "rgba(230, 159, 0, 0.22)",
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
    Formats p-values cleanly.
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
    Interprets p-value.
    """

    if p_value < alpha:
        return "Reject H0", "There is a statistically significant result."
    else:
        return "Fail to Reject H0", "There is not enough evidence for a statistically significant result."


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=420):
    """
    Applies the soft professional SandeepStician Plotly theme.
    """

    fig.update_layout(
        title={
            "text": title if title else "",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 18, "color": PLOT_COLORS["text"], "family": "Arial"},
        },
        paper_bgcolor=PLOT_COLORS["bg"],
        plot_bgcolor=PLOT_COLORS["card"],
        font={"color": PLOT_COLORS["text"], "family": "Arial"},
        height=height,
        margin={"l": 55, "r": 30, "t": 62, "b": 52},
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": PLOT_COLORS["muted"]},
            "bgcolor": "rgba(255,255,255,0)",
        },
    )

    fig.update_xaxes(
        title_text=x_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["axis"],
        linecolor=PLOT_COLORS["axis"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
        showline=True,
        linewidth=1,
        mirror=False,
    )

    fig.update_yaxes(
        title_text=y_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["axis"],
        linecolor=PLOT_COLORS["axis"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
        showline=True,
        linewidth=1,
        mirror=False,
    )

    fig.update_traces(
        selector=dict(type="box"),
        line={"width": 2},
        whiskerwidth=0.65,
    )

    return fig


def create_result_table(result):
    """
    Converts result dictionary into clean dataframe.
    """

    hidden_keys = [
        "H0",
        "H1",
        "Conclusion",
        "Group Summary",
        "Frequency Table",
        "Clean Data",
        "Difference Data",
        "Selected Columns",
    ]

    rows = []

    for key, value in result.items():
        if key in hidden_keys:
            continue

        if "p-value" in key:
            formatted_value = format_p_value(value)
        else:
            formatted_value = format_number(value)

        rows.append(
            {
                "Metric": key,
                "Value": formatted_value,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Mann-Whitney U Test
# ------------------------------------------------------------

def run_mannwhitney_u_test(
    df,
    numeric_column,
    group_column,
    group1,
    group2,
    alternative="two-sided",
    alpha=0.05
):
    """
    Mann-Whitney U test.

    Non-parametric alternative to independent two-sample t-test.

    H0: The distributions of the two groups are the same.
    H1: The distributions of the two groups are different.
    """

    group1_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group1)][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group2)][numeric_column],
        errors="coerce"
    ).dropna()

    if len(group1_data) < 1 or len(group2_data) < 1:
        raise ValueError("Both groups must contain at least 1 valid numerical value.")

    statistic, p_value = stats.mannwhitneyu(
        group1_data,
        group2_data,
        alternative=alternative
    )

    n1 = len(group1_data)
    n2 = len(group2_data)

    rank_biserial = (2 * statistic / (n1 * n2)) - 1

    decision, conclusion = interpret_p_value(p_value, alpha)

    result = {
        "Test": "Mann-Whitney U test",
        "H0": f"The distribution of {numeric_column} is the same for {group1} and {group2}.",
        "H1": f"The distribution of {numeric_column} is different between {group1} and {group2}.",
        "Numeric Column": numeric_column,
        "Group Column": group_column,
        "Group 1": group1,
        "Group 2": group2,
        "Group 1 Sample Size": n1,
        "Group 2 Sample Size": n2,
        "Group 1 Median": group1_data.median(),
        "Group 2 Median": group2_data.median(),
        "U Statistic": statistic,
        "Rank-biserial Effect Size": rank_biserial,
        "p-value": p_value,
        "Alpha": alpha,
        "Alternative": alternative,
        "Decision": decision,
        "Conclusion": conclusion,
        "Clean Data": {
            "group1_data": group1_data,
            "group2_data": group2_data,
        }
    }

    return result


def plot_mannwhitney_u(result):
    """
    Boxplot for Mann-Whitney U test.
    """

    group1_data = result["Clean Data"]["group1_data"]
    group2_data = result["Clean Data"]["group2_data"]

    group1 = result["Group 1"]
    group2 = result["Group 2"]
    numeric_column = result["Numeric Column"]

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=group1_data,
            name=str(group1),
            marker_color=PLOT_COLORS["primary"],
            line_color=PLOT_COLORS["primary"],
            fillcolor=PLOT_COLORS["soft_primary"],
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Box(
            y=group2_data,
            name=str(group2),
            marker_color=PLOT_COLORS["secondary"],
            line_color=PLOT_COLORS["secondary"],
            fillcolor=PLOT_COLORS["soft_secondary"],
            boxmean=True,
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Mann-Whitney U: {numeric_column} by group",
        x_title="Group",
        y_title=numeric_column,
        height=420,
    )

    return fig


# ------------------------------------------------------------
# Wilcoxon Signed-Rank Test
# ------------------------------------------------------------

def run_wilcoxon_signed_rank_test(
    df,
    before_column,
    after_column,
    alternative="two-sided",
    alpha=0.05
):
    """
    Wilcoxon signed-rank test.

    Non-parametric alternative to paired t-test.

    H0: The median paired difference is 0.
    H1: The median paired difference is not 0.
    """

    clean_df = df[[before_column, after_column]].copy()

    clean_df[before_column] = pd.to_numeric(clean_df[before_column], errors="coerce")
    clean_df[after_column] = pd.to_numeric(clean_df[after_column], errors="coerce")

    clean_df = clean_df.dropna()

    if len(clean_df) < 2:
        raise ValueError("Wilcoxon signed-rank test requires at least 2 complete pairs.")

    differences = clean_df[after_column] - clean_df[before_column]

    non_zero_differences = differences[differences != 0]

    if len(non_zero_differences) == 0:
        raise ValueError("All paired differences are zero, so Wilcoxon test cannot be calculated.")

    statistic, p_value = stats.wilcoxon(
        clean_df[before_column],
        clean_df[after_column],
        alternative=alternative,
        zero_method="wilcox"
    )

    decision, conclusion = interpret_p_value(p_value, alpha)

    result = {
        "Test": "Wilcoxon signed-rank test",
        "H0": f"The median paired difference between {before_column} and {after_column} is 0.",
        "H1": f"The median paired difference between {before_column} and {after_column} is not 0.",
        "Before Column": before_column,
        "After Column": after_column,
        "Number of Complete Pairs": len(clean_df),
        "Non-zero Differences": len(non_zero_differences),
        "Before Median": clean_df[before_column].median(),
        "After Median": clean_df[after_column].median(),
        "Median Difference": differences.median(),
        "W Statistic": statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Alternative": alternative,
        "Decision": decision,
        "Conclusion": conclusion,
        "Clean Data": clean_df,
        "Difference Data": differences,
    }

    return result


def plot_wilcoxon_signed_rank(result):
    """
    Before-after paired line plot.
    """

    clean_df = result["Clean Data"]
    before_column = result["Before Column"]
    after_column = result["After Column"]

    plot_df = clean_df.head(80)

    fig = go.Figure()

    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[before_column, after_column],
                y=[row[before_column], row[after_column]],
                mode="lines+markers",
                line={"color": "rgba(86, 180, 233, 0.28)", "width": 1.5},
                marker={"size": 6, "color": PLOT_COLORS["primary"]},
                showlegend=False,
                hovertemplate="Measurement: %{x}<br>Value: %{y}<extra></extra>",
            )
        )

    before_median = clean_df[before_column].median()
    after_median = clean_df[after_column].median()

    fig.add_trace(
        go.Scatter(
            x=[before_column, after_column],
            y=[before_median, after_median],
            mode="lines+markers",
            line={"color": PLOT_COLORS["green"], "width": 4},
            marker={"size": 10},
            name="Median change",
            hovertemplate="Measurement: %{x}<br>Median: %{y:.4f}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Wilcoxon Signed-Rank: {before_column} vs {after_column}",
        x_title="Measurement",
        y_title="Value",
        height=440,
    )

    return fig


def plot_wilcoxon_differences(result):
    """
    Histogram of paired differences.
    """

    differences = result["Difference Data"]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=differences,
            nbinsx=25,
            marker={
                "color": PLOT_COLORS["primary"],
                "line": {"color": PLOT_COLORS["bar_line"], "width": 1},
            },
            opacity=0.85,
            name="Differences",
        )
    )

    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color=PLOT_COLORS["warning"],
        annotation_text="No difference",
        annotation_position="top right",
    )

    fig = apply_plotly_theme(
        fig,
        title="Distribution of Paired Differences",
        x_title="After - Before",
        y_title="Frequency",
        height=420,
    )

    return fig


# ------------------------------------------------------------
# Kruskal-Wallis Test
# ------------------------------------------------------------

def run_kruskal_wallis_test(
    df,
    numeric_column,
    group_column,
    selected_groups=None,
    alpha=0.05
):
    """
    Kruskal-Wallis test.

    Non-parametric alternative to one-way ANOVA.

    H0: The group distributions are the same.
    H1: At least one group distribution is different.
    """

    clean_df = df[[numeric_column, group_column]].copy()
    clean_df[numeric_column] = pd.to_numeric(clean_df[numeric_column], errors="coerce")
    clean_df[group_column] = clean_df[group_column].astype(str)
    clean_df = clean_df.dropna()

    if selected_groups is not None and len(selected_groups) > 0:
        selected_groups = [str(group) for group in selected_groups]
        clean_df = clean_df[clean_df[group_column].isin(selected_groups)]

    groups = []

    for group_name, group_data in clean_df.groupby(group_column):
        values = group_data[numeric_column].dropna()

        if len(values) > 0:
            groups.append((group_name, values))

    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis test requires at least 2 valid groups.")

    statistic, p_value = stats.kruskal(*[values for _, values in groups])

    total_n = sum(len(values) for _, values in groups)
    k = len(groups)

    epsilon_squared = (statistic - k + 1) / (total_n - k) if total_n > k else np.nan

    decision, conclusion = interpret_p_value(p_value, alpha)

    group_summary = clean_df.groupby(group_column)[numeric_column].agg(
        ["count", "median", "mean", "std", "min", "max"]
    ).reset_index()

    result = {
        "Test": "Kruskal-Wallis test",
        "H0": f"The distribution of {numeric_column} is the same across groups of {group_column}.",
        "H1": f"At least one group distribution of {numeric_column} is different.",
        "Numeric Column": numeric_column,
        "Group Column": group_column,
        "Number of Groups": k,
        "Total Sample Size": total_n,
        "H Statistic": statistic,
        "Epsilon-squared Effect Size": epsilon_squared,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
        "Group Summary": group_summary,
        "Clean Data": clean_df,
    }

    return result


def plot_kruskal_wallis(result):
    """
    Boxplot for Kruskal-Wallis test.
    """

    clean_df = result["Clean Data"]
    numeric_column = result["Numeric Column"]
    group_column = result["Group Column"]

    fig = go.Figure()

    for index, (group_name, group_data) in enumerate(clean_df.groupby(group_column)):
        color = CHART_PALETTE[index % len(CHART_PALETTE)]

        fig.add_trace(
            go.Box(
                y=group_data[numeric_column],
                name=str(group_name),
                marker_color=color,
                line_color=color,
                fillcolor="rgba(86, 180, 233, 0.16)" if index % len(CHART_PALETTE) == 0 else "rgba(213, 94, 0, 0.14)" if index % len(CHART_PALETTE) == 1 else "rgba(0, 158, 115, 0.14)" if index % len(CHART_PALETTE) == 2 else "rgba(230, 159, 0, 0.16)",
                boxmean=True,
            )
        )

    fig = apply_plotly_theme(
        fig,
        title=f"Kruskal-Wallis: {numeric_column} by {group_column}",
        x_title=group_column,
        y_title=numeric_column,
        height=460,
    )

    return fig


# ------------------------------------------------------------
# Friedman Test
# ------------------------------------------------------------

def run_friedman_test(
    df,
    measurement_columns,
    alpha=0.05
):
    """
    Friedman test.

    Non-parametric test for 3 or more related/repeated measurements.

    H0: The repeated measurement distributions are the same.
    H1: At least one repeated measurement distribution is different.
    """

    if len(measurement_columns) < 3:
        raise ValueError("Friedman test requires at least 3 related measurement columns.")

    clean_df = df[measurement_columns].copy()

    for column in measurement_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.dropna()

    if len(clean_df) < 2:
        raise ValueError("Friedman test requires at least 2 complete rows.")

    statistic, p_value = stats.friedmanchisquare(
        *[clean_df[column] for column in measurement_columns]
    )

    n = len(clean_df)
    k = len(measurement_columns)

    kendalls_w = statistic / (n * (k - 1)) if n > 0 and k > 1 else np.nan

    decision, conclusion = interpret_p_value(p_value, alpha)

    summary_rows = []

    for column in measurement_columns:
        summary_rows.append(
            {
                "Measurement": column,
                "Count": clean_df[column].count(),
                "Median": clean_df[column].median(),
                "Mean": clean_df[column].mean(),
                "Std": clean_df[column].std(ddof=1),
                "Min": clean_df[column].min(),
                "Max": clean_df[column].max(),
            }
        )

    group_summary = pd.DataFrame(summary_rows)

    result = {
        "Test": "Friedman test",
        "H0": "The repeated measurement distributions are the same.",
        "H1": "At least one repeated measurement distribution is different.",
        "Number of Complete Rows": n,
        "Number of Measurements": k,
        "Chi-square Statistic": statistic,
        "Kendall's W Effect Size": kendalls_w,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
        "Selected Columns": measurement_columns,
        "Group Summary": group_summary,
        "Clean Data": clean_df,
    }

    return result


def plot_friedman_test(result):
    """
    Boxplot for repeated measurement columns.
    """

    clean_df = result["Clean Data"]
    measurement_columns = result["Selected Columns"]

    fig = go.Figure()

    for index, column in enumerate(measurement_columns):
        color = CHART_PALETTE[index % len(CHART_PALETTE)]

        fig.add_trace(
            go.Box(
                y=clean_df[column],
                name=str(column),
                marker_color=color,
                line_color=color,
                fillcolor="rgba(86, 180, 233, 0.16)" if index % len(CHART_PALETTE) == 0 else "rgba(213, 94, 0, 0.14)" if index % len(CHART_PALETTE) == 1 else "rgba(0, 158, 115, 0.14)" if index % len(CHART_PALETTE) == 2 else "rgba(230, 159, 0, 0.16)",
                boxmean=True,
            )
        )

    fig = apply_plotly_theme(
        fig,
        title="Friedman Test: Repeated Measurements",
        x_title="Measurement",
        y_title="Value",
        height=460,
    )

    return fig


# ------------------------------------------------------------
# Interpretations
# ------------------------------------------------------------

def get_nonparametric_interpretation(result):
    """
    Plain-English interpretation.
    """

    interpretation = []

    interpretation.append(f"**H0:** {result['H0']}")
    interpretation.append(f"**H1:** {result['H1']}")

    interpretation.append(
        f"The p-value is {format_p_value(result['p-value'])}, and alpha is {result['Alpha']}."
    )

    if result["Decision"] == "Reject H0":
        interpretation.append("Because the p-value is less than alpha, we reject H0.")
    else:
        interpretation.append("Because the p-value is greater than or equal to alpha, we fail to reject H0.")

    interpretation.append(result["Conclusion"])

    if result["Test"] == "Mann-Whitney U test":
        interpretation.append(
            "This test compares two independent groups using ranks instead of raw means."
        )

    elif result["Test"] == "Wilcoxon signed-rank test":
        interpretation.append(
            "This test compares paired measurements using the ranks of the paired differences."
        )

    elif result["Test"] == "Kruskal-Wallis test":
        interpretation.append(
            "This test compares three or more independent groups using ranks. "
            "If significant, a post-hoc test is needed to identify which groups differ."
        )

    elif result["Test"] == "Friedman test":
        interpretation.append(
            "This test compares three or more related/repeated measurements using ranks."
        )

    return interpretation