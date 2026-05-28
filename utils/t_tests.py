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
    Histogram with sample mean and hypothesized mean.
    """

    data = prepare_numeric_data(df, numeric_column)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.hist(data, bins=20, edgecolor="black", alpha=0.75)

    ax.axvline(data.mean(), linestyle="--", linewidth=2, label="Sample Mean")
    ax.axvline(hypothesized_mean, linestyle="-", linewidth=2, label="Hypothesized Mean")

    ax.set_title(f"One-sample t-test: {numeric_column}", fontsize=12)
    ax.set_xlabel(numeric_column)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()

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
        df[df[group_column] == group1][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column] == group2][numeric_column],
        errors="coerce"
    ).dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Each group must have at least 2 valid numerical values.")

    # Levene's test checks equal variance assumption
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
    Boxplot comparing two independent groups.
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
    Line plot showing before-after paired values.
    """

    paired_df = df[[before_column, after_column]].copy()

    paired_df[before_column] = pd.to_numeric(paired_df[before_column], errors="coerce")
    paired_df[after_column] = pd.to_numeric(paired_df[after_column], errors="coerce")

    paired_df = paired_df.dropna()

    # Limit lines for readability
    plot_df = paired_df.head(50)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for _, row in plot_df.iterrows():
        ax.plot([0, 1], [row[before_column], row[after_column]], marker="o", alpha=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([before_column, after_column])

    ax.set_title("Paired t-test: Before vs After", fontsize=12)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

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

        rows.append({
            "Metric": key,
            "Value": format_number(value)
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
        f"The p-value is {format_number(result['p-value'])}, and alpha is {result['Alpha']}."
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