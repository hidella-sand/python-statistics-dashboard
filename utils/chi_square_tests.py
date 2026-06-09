import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go


PLOT_COLORS = {
    "primary": "#7C5CFF",
    "secondary": "#A78BFA",
    "blue": "#60A5FA",
    "green": "#00B894",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "bg": "#181A1F",
    "card": "#242529",
    "grid": "#3A3B40",
    "text": "#F5F5F5",
    "muted": "#A3A3A3",
}


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=460):
    """
    Applies the dark dashboard theme to Plotly figures.
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
        margin={"l": 70, "r": 35, "t": 70, "b": 65},
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
        linecolor=PLOT_COLORS["grid"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
    )

    fig.update_yaxes(
        title_text=y_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["grid"],
        linecolor=PLOT_COLORS["grid"],
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
    )

    return fig


def format_number(value):
    """
    Formats numbers nicely for tables and interpretation text.
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
    Interprets p-value for chi-square tests.
    """

    if p_value < alpha:
        return "Reject H0", "There is a statistically significant result."

    return "Fail to Reject H0", "There is not enough evidence for a statistically significant result."


def prepare_categorical_data(df, columns):
    """
    Keeps selected categorical columns and removes missing values.
    """

    clean_df = df[columns].copy()

    clean_df = clean_df.dropna()

    for col in columns:
        clean_df[col] = clean_df[col].astype(str)

    if clean_df.empty:
        raise ValueError("No valid data available after removing missing values.")

    for col in columns:
        if clean_df[col].nunique() < 2:
            raise ValueError(f"Column '{col}' must contain at least two categories.")

    return clean_df


# ------------------------------------------------------------
# Chi-square Test of Independence
# ------------------------------------------------------------

def run_chi_square_independence(df, column1, column2, alpha=0.05):
    """
    Chi-square test of independence.

    H0: The two categorical variables are independent.
    H1: The two categorical variables are associated/dependent.
    """

    clean_df = prepare_categorical_data(df, [column1, column2])

    observed_table = pd.crosstab(clean_df[column1], clean_df[column2])

    chi2_statistic, p_value, dof, expected_values = stats.chi2_contingency(
        observed_table
    )

    expected_table = pd.DataFrame(
        expected_values,
        index=observed_table.index,
        columns=observed_table.columns
    )

    decision, conclusion = interpret_p_value(p_value, alpha)

    total_n = observed_table.values.sum()
    min_dimension = min(observed_table.shape)

    if min_dimension > 1:
        cramers_v = np.sqrt(chi2_statistic / (total_n * (min_dimension - 1)))
    else:
        cramers_v = np.nan

    low_expected_count = int((expected_values < 5).sum())
    total_cells = expected_values.size

    result = {
        "Test": "Chi-square Test of Independence",
        "H0": f"{column1} and {column2} are independent.",
        "H1": f"{column1} and {column2} are associated/dependent.",
        "Variable 1": column1,
        "Variable 2": column2,
        "Chi-square Statistic": chi2_statistic,
        "Degrees of Freedom": dof,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
        "Cramer's V": cramers_v,
        "Total Observations": total_n,
        "Low Expected Frequency Cells": low_expected_count,
        "Total Cells": total_cells,
        "Observed Table": observed_table,
        "Expected Table": expected_table,
    }

    return result


def plot_chi_square_independence(df, column1, column2):
    """
    Creates a modern Plotly heatmap for the observed contingency table.
    """

    clean_df = prepare_categorical_data(df, [column1, column2])
    observed_table = pd.crosstab(clean_df[column1], clean_df[column2])

    fig = go.Figure(
        data=go.Heatmap(
            z=observed_table.values,
            x=observed_table.columns.astype(str),
            y=observed_table.index.astype(str),
            colorscale=[
                [0, PLOT_COLORS["card"]],
                [0.5, PLOT_COLORS["secondary"]],
                [1, PLOT_COLORS["primary"]],
            ],
            text=observed_table.values,
            texttemplate="%{text}",
            textfont={"color": PLOT_COLORS["text"], "size": 14},
            colorbar={"title": "Count"},
            hovertemplate=(
                f"{column1}: %{{y}}<br>"
                f"{column2}: %{{x}}<br>"
                "Observed Count: %{z}<extra></extra>"
            ),
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Observed Counts: {column1} vs {column2}",
        x_title=column2,
        y_title=column1,
        height=520,
    )

    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")

    return fig


def get_independence_interpretation(result):
    """
    Plain-English interpretation for chi-square test of independence.
    """

    interpretation = []

    interpretation.append(f"**H0:** {result['H0']}")
    interpretation.append(f"**H1:** {result['H1']}")

    interpretation.append(
        f"The chi-square statistic is {format_number(result['Chi-square Statistic'])}, "
        f"with {result['Degrees of Freedom']} degrees of freedom."
    )

    interpretation.append(
        f"The p-value is {format_p_value(result['p-value'])}, and alpha is {result['Alpha']}."
    )

    if result["Decision"] == "Reject H0":
        interpretation.append(
            "Because the p-value is less than alpha, we reject H0."
        )
        interpretation.append(
            f"This suggests `{result['Variable 1']}` and `{result['Variable 2']}` are associated."
        )
    else:
        interpretation.append(
            "Because the p-value is greater than or equal to alpha, we fail to reject H0."
        )
        interpretation.append(
            f"This means there is not enough evidence to say `{result['Variable 1']}` and `{result['Variable 2']}` are associated."
        )

    cramers_v = result["Cramer's V"]

    if not pd.isna(cramers_v):
        if cramers_v < 0.1:
            interpretation.append("Cramer's V suggests a very weak association.")
        elif cramers_v < 0.3:
            interpretation.append("Cramer's V suggests a weak association.")
        elif cramers_v < 0.5:
            interpretation.append("Cramer's V suggests a moderate association.")
        else:
            interpretation.append("Cramer's V suggests a strong association.")

    if result["Low Expected Frequency Cells"] > 0:
        interpretation.append(
            f"{result['Low Expected Frequency Cells']} out of {result['Total Cells']} expected frequency cells are below 5. "
            "This may weaken the reliability of the chi-square test."
        )
    else:
        interpretation.append(
            "All expected frequencies are at least 5, so the chi-square assumption is reasonably satisfied."
        )

    return interpretation


# ------------------------------------------------------------
# Chi-square Goodness-of-Fit Test
# ------------------------------------------------------------

def run_chi_square_goodness_of_fit(
    df,
    categorical_column,
    expected_frequencies=None,
    alpha=0.05
):
    """
    Chi-square goodness-of-fit test.

    H0: Observed frequencies match the expected frequencies.
    H1: Observed frequencies do not match the expected frequencies.
    """

    clean_df = prepare_categorical_data(df, [categorical_column])

    observed_counts = clean_df[categorical_column].value_counts().sort_index()

    if len(observed_counts) < 2:
        raise ValueError("Goodness-of-fit test needs at least two categories.")

    if expected_frequencies is None:
        expected_values = np.repeat(
            observed_counts.sum() / len(observed_counts),
            len(observed_counts)
        )

    else:
        expected_values = []

        for category in observed_counts.index:
            expected_values.append(float(expected_frequencies.get(category, 0)))

        expected_values = np.array(expected_values)

        if np.any(expected_values <= 0):
            raise ValueError("All expected frequencies must be greater than 0.")

        expected_values = expected_values * (
            observed_counts.sum() / expected_values.sum()
        )

    chi2_statistic, p_value = stats.chisquare(
        f_obs=observed_counts.values,
        f_exp=expected_values
    )

    dof = len(observed_counts) - 1

    decision, conclusion = interpret_p_value(p_value, alpha)

    observed_expected_table = pd.DataFrame({
        "Category": observed_counts.index,
        "Observed Frequency": observed_counts.values,
        "Expected Frequency": expected_values,
    })

    low_expected_count = int((expected_values < 5).sum())

    result = {
        "Test": "Chi-square Goodness-of-Fit Test",
        "H0": f"The observed distribution of {categorical_column} matches the expected distribution.",
        "H1": f"The observed distribution of {categorical_column} does not match the expected distribution.",
        "Categorical Column": categorical_column,
        "Chi-square Statistic": chi2_statistic,
        "Degrees of Freedom": dof,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
        "Low Expected Frequency Cells": low_expected_count,
        "Observed Expected Table": observed_expected_table,
    }

    return result


def plot_goodness_of_fit(result):
    """
    Creates a modern Plotly grouped bar chart comparing observed and expected frequencies.
    """

    table = result["Observed Expected Table"]

    categories = table["Category"].astype(str).values
    observed = table["Observed Frequency"].values
    expected = table["Expected Frequency"].values

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=observed,
            name="Observed",
            marker_color=PLOT_COLORS["primary"],
            hovertemplate="Category: %{x}<br>Observed: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=expected,
            name="Expected",
            marker_color=PLOT_COLORS["secondary"],
            hovertemplate="Category: %{x}<br>Expected: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")

    fig = apply_plotly_theme(
        fig,
        title=f"Observed vs Expected: {result['Categorical Column']}",
        x_title="Category",
        y_title="Frequency",
        height=500,
    )

    fig.update_xaxes(type="category")

    return fig


def get_goodness_of_fit_interpretation(result):
    """
    Plain-English interpretation for chi-square goodness-of-fit test.
    """

    interpretation = []

    interpretation.append(f"**H0:** {result['H0']}")
    interpretation.append(f"**H1:** {result['H1']}")

    interpretation.append(
        f"The chi-square statistic is {format_number(result['Chi-square Statistic'])}, "
        f"with {result['Degrees of Freedom']} degrees of freedom."
    )

    interpretation.append(
        f"The p-value is {format_p_value(result['p-value'])}, and alpha is {result['Alpha']}."
    )

    if result["Decision"] == "Reject H0":
        interpretation.append(
            "Because the p-value is less than alpha, we reject H0."
        )
        interpretation.append(
            "This suggests the observed frequencies are significantly different from the expected frequencies."
        )
    else:
        interpretation.append(
            "Because the p-value is greater than or equal to alpha, we fail to reject H0."
        )
        interpretation.append(
            "This means there is not enough evidence to say the observed frequencies differ from the expected frequencies."
        )

    if result["Low Expected Frequency Cells"] > 0:
        interpretation.append(
            f"{result['Low Expected Frequency Cells']} expected frequency cell(s) are below 5. "
            "This may weaken the reliability of the goodness-of-fit test."
        )
    else:
        interpretation.append(
            "All expected frequencies are at least 5, so the chi-square assumption is reasonably satisfied."
        )

    return interpretation


# ------------------------------------------------------------
# Shared table helper
# ------------------------------------------------------------

def create_chi_square_result_table(result):
    """
    Converts chi-square result dictionary into a clean table.
    """

    hidden_keys = [
        "H0",
        "H1",
        "Conclusion",
        "Observed Table",
        "Expected Table",
        "Observed Expected Table",
    ]

    rows = []

    for key, value in result.items():
        if key in hidden_keys:
            continue

        if key == "p-value":
            formatted_value = format_p_value(value)
        else:
            formatted_value = format_number(value)

        rows.append({
            "Metric": key,
            "Value": formatted_value,
        })

    return pd.DataFrame(rows)