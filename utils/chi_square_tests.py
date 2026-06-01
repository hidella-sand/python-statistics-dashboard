import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def format_number(value):
    """
    Formats numbers nicely for tables.
    """
    try:
        return round(float(value), 5)
    except Exception:
        return value


def interpret_p_value(p_value, alpha=0.05):
    """
    Interprets p-value for chi-square tests.
    """

    if p_value < alpha:
        return "Reject H0", "There is a statistically significant result."
    else:
        return "Fail to Reject H0", "There is not enough evidence for a statistically significant result."


def prepare_categorical_data(df, columns):
    """
    Keeps selected categorical columns and removes missing values.
    """

    clean_df = df[columns].copy()

    for col in columns:
        clean_df[col] = clean_df[col].astype(str)

    clean_df = clean_df.dropna()

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

    chi2_statistic, p_value, dof, expected_values = stats.chi2_contingency(observed_table)

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
        "Expected Table": expected_table
    }

    return result


def plot_chi_square_independence(df, column1, column2):
    """
    Heatmap-style plot for observed contingency table.
    """

    clean_df = prepare_categorical_data(df, [column1, column2])
    observed_table = pd.crosstab(clean_df[column1], clean_df[column2])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    image = ax.imshow(observed_table.values)

    ax.set_xticks(np.arange(len(observed_table.columns)))
    ax.set_yticks(np.arange(len(observed_table.index)))

    ax.set_xticklabels(observed_table.columns)
    ax.set_yticklabels(observed_table.index)

    ax.set_xlabel(column2)
    ax.set_ylabel(column1)
    ax.set_title(f"Observed Counts: {column1} vs {column2}", fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for i in range(len(observed_table.index)):
        for j in range(len(observed_table.columns)):
            ax.text(
                j,
                i,
                observed_table.values[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

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
        f"The p-value is {format_number(result['p-value'])}, and alpha is {result['Alpha']}."
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

def run_chi_square_goodness_of_fit(df, categorical_column, expected_frequencies=None, alpha=0.05):
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
        expected_values = np.repeat(observed_counts.sum() / len(observed_counts), len(observed_counts))
    else:
        expected_values = []

        for category in observed_counts.index:
            expected_values.append(float(expected_frequencies.get(category, 0)))

        expected_values = np.array(expected_values)

        if np.any(expected_values <= 0):
            raise ValueError("All expected frequencies must be greater than 0.")

        # Scale expected values to match total observed count
        expected_values = expected_values * (observed_counts.sum() / expected_values.sum())

    chi2_statistic, p_value = stats.chisquare(
        f_obs=observed_counts.values,
        f_exp=expected_values
    )

    dof = len(observed_counts) - 1

    decision, conclusion = interpret_p_value(p_value, alpha)

    observed_expected_table = pd.DataFrame({
        "Category": observed_counts.index,
        "Observed Frequency": observed_counts.values,
        "Expected Frequency": expected_values
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
        "Observed Expected Table": observed_expected_table
    }

    return result


def plot_goodness_of_fit(result):
    """
    Bar chart comparing observed and expected frequencies.
    """

    table = result["Observed Expected Table"]

    categories = table["Category"].astype(str).values
    observed = table["Observed Frequency"].values
    expected = table["Expected Frequency"].values

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.bar(x - width / 2, observed, width, label="Observed")
    ax.bar(x + width / 2, expected, width, label="Expected")

    ax.set_title(f"Observed vs Expected: {result['Categorical Column']}", fontsize=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

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
        f"The p-value is {format_number(result['p-value'])}, and alpha is {result['Alpha']}."
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
        "Observed Expected Table"
    ]

    rows = []

    for key, value in result.items():
        if key in hidden_keys:
            continue

        rows.append({
            "Metric": key,
            "Value": format_number(value)
        })

    return pd.DataFrame(rows)