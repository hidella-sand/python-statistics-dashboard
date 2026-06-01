import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def format_number(value):
    """
    Formats numbers nicely for tables and interpretations.
    """
    try:
        return round(float(value), 5)
    except Exception:
        return value


def interpret_p_value(p_value, alpha=0.05):
    """
    Interprets p-value using alpha.
    """
    if p_value < alpha:
        return "Reject H0", "There is a statistically significant result."
    else:
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

    result = {
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
        "Conclusion": conclusion
    }

    return result


def plot_one_sample_ztest(df, numeric_column, hypothesized_mean):
    """
    Histogram with sample mean and hypothesized mean.
    """

    data = prepare_numeric_data(df, numeric_column)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.hist(data, bins=20, edgecolor="black", alpha=0.75)

    ax.axvline(data.mean(), linestyle="--", linewidth=2, label="Sample Mean")
    ax.axvline(hypothesized_mean, linestyle="-", linewidth=2, label="Hypothesized Mean")

    ax.set_title(f"One-sample z-test: {numeric_column}", fontsize=12)
    ax.set_xlabel(numeric_column)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()

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
    alpha=0.05
):
    """
    Two-sample mean z-test.

    H0: The two population means are equal.
    H1: The two population means are not equal.
    """

    group1_data = pd.to_numeric(
        df[df[group_column] == group1][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column] == group2][numeric_column],
        errors="coerce"
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

    result = {
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
        "Conclusion": conclusion
    }

    return result


def plot_two_sample_ztest(df, numeric_column, group_column, group1, group2):
    """
    Boxplot comparing two groups.
    """

    group1_data = pd.to_numeric(
        df[df[group_column] == group1][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column] == group2][numeric_column],
        errors="coerce"
    ).dropna()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.boxplot([group1_data, group2_data], tick_labels=[str(group1), str(group2)])

    ax.set_title(f"{numeric_column} by {group_column}", fontsize=12)
    ax.set_xlabel(group_column)
    ax.set_ylabel(numeric_column)
    ax.grid(True, alpha=0.3)

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

    result = {
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
        "Conclusion": conclusion
    }

    return result


def plot_one_proportion_ztest(result):
    """
    Bar chart comparing sample proportion and hypothesized proportion.
    """

    labels = ["Sample Proportion", "Hypothesized Proportion"]
    values = [result["Sample Proportion"], result["Hypothesized Proportion"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.bar(labels, values, edgecolor="black")

    ax.set_ylim(0, 1)
    ax.set_title("Sample vs Hypothesized Proportion", fontsize=12)
    ax.set_ylabel("Proportion")
    ax.grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, value, str(format_number(value)), ha="center", va="bottom")

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

    result = {
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
        "Conclusion": conclusion
    }

    return result


def plot_two_proportion_ztest(result):
    """
    Bar chart comparing two sample proportions.
    """

    labels = [str(result["Group 1"]), str(result["Group 2"])]
    values = [result["Group 1 Proportion"], result["Group 2 Proportion"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.bar(labels, values, edgecolor="black")

    ax.set_ylim(0, 1)
    ax.set_title("Group Proportion Comparison", fontsize=12)
    ax.set_xlabel(result["Group Column"])
    ax.set_ylabel(f"Proportion of {result['Success Category']}")
    ax.grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, value, str(format_number(value)), ha="center", va="bottom")

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
            "Value": format_number(value)
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