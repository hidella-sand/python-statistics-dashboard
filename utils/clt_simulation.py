import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def format_number(value):
    """
    Formats numbers nicely for tables and interpretations.
    """
    try:
        return round(float(value), 6)
    except Exception:
        return value


def prepare_numeric_data(df, column):
    """
    Converts selected column into numeric data and removes missing values.
    """

    data = pd.to_numeric(df[column], errors="coerce").dropna()

    if data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    if data.nunique() < 2:
        raise ValueError("CLT simulation needs at least two different numerical values.")

    return data


def simulate_sample_means(df, column, sample_size=30, number_of_samples=1000, random_seed=42):
    """
    Simulates the Central Limit Theorem.

    Steps:
    1. Select a numerical column.
    2. Repeatedly draw random samples with replacement.
    3. Calculate the mean of each sample.
    4. Return the sample means.
    """

    data = prepare_numeric_data(df, column)

    if sample_size < 2:
        raise ValueError("Sample size must be at least 2.")

    if number_of_samples < 10:
        raise ValueError("Number of samples should be at least 10.")

    rng = np.random.default_rng(random_seed)

    sample_means = []

    data_values = data.values

    for _ in range(number_of_samples):
        sample = rng.choice(data_values, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    return data, sample_means


def create_clt_summary_table(data, sample_means, sample_size):
    """
    Creates summary table comparing original data and sampling distribution.
    """

    original_mean = data.mean()
    original_std = data.std(ddof=1)

    sample_means_mean = sample_means.mean()
    sample_means_std = sample_means.std(ddof=1)

    theoretical_standard_error = original_std / np.sqrt(sample_size)

    rows = [
        {
            "Metric": "Original Data Mean",
            "Value": format_number(original_mean)
        },
        {
            "Metric": "Original Data Standard Deviation",
            "Value": format_number(original_std)
        },
        {
            "Metric": "Sample Size Used",
            "Value": sample_size
        },
        {
            "Metric": "Mean of Sample Means",
            "Value": format_number(sample_means_mean)
        },
        {
            "Metric": "Standard Deviation of Sample Means",
            "Value": format_number(sample_means_std)
        },
        {
            "Metric": "Theoretical Standard Error",
            "Value": format_number(theoretical_standard_error)
        },
        {
            "Metric": "Difference: Original Mean vs Mean of Sample Means",
            "Value": format_number(abs(original_mean - sample_means_mean))
        }
    ]

    return pd.DataFrame(rows)


def plot_original_distribution(data, column, bins=30):
    """
    Plots the original data distribution.
    """

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.hist(data, bins=bins, edgecolor="black", alpha=0.75)

    ax.axvline(
        data.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Mean = {format_number(data.mean())}"
    )

    ax.set_title(f"Original Distribution of {column}", fontsize=12)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig


def plot_sampling_distribution(sample_means, original_mean, theoretical_standard_error, bins=30):
    """
    Plots the distribution of sample means with a normal curve overlay.
    """

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.hist(
        sample_means,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.60,
        label="Sample Means"
    )

    x_values = np.linspace(sample_means.min(), sample_means.max(), 400)

    normal_curve = stats.norm.pdf(
        x_values,
        loc=original_mean,
        scale=theoretical_standard_error
    )

    ax.plot(
        x_values,
        normal_curve,
        linewidth=2,
        label="Normal Curve using CLT"
    )

    ax.axvline(
        sample_means.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Mean of Sample Means = {format_number(sample_means.mean())}"
    )

    ax.set_title("Sampling Distribution of the Mean", fontsize=12)
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig


def simulate_multiple_sample_sizes(df, column, sample_sizes, number_of_samples=1000, random_seed=42):
    """
    Simulates sample means for multiple sample sizes.
    Useful to show how the sampling distribution changes when sample size increases.
    """

    data = prepare_numeric_data(df, column)

    results = {}

    for sample_size in sample_sizes:
        _, sample_means = simulate_sample_means(
            df,
            column,
            sample_size=sample_size,
            number_of_samples=number_of_samples,
            random_seed=random_seed
        )

        results[sample_size] = sample_means

    return data, results


def plot_sample_size_comparison(sample_size_results):
    """
    Plots sampling distributions for different sample sizes.
    """

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    for sample_size, sample_means in sample_size_results.items():
        kde = stats.gaussian_kde(sample_means)
        x_values = np.linspace(sample_means.min(), sample_means.max(), 300)
        y_values = kde(x_values)

        ax.plot(
            x_values,
            y_values,
            linewidth=2,
            label=f"n = {sample_size}"
        )

    ax.set_title("Effect of Sample Size on Sampling Distribution", fontsize=12)
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig


def run_normality_check_on_sample_means(sample_means, alpha=0.05):
    """
    Runs Shapiro-Wilk normality test on sample means.
    Uses max 5000 sample means for stability.
    """

    test_data = sample_means

    note = "Test performed on all sample means."

    if len(sample_means) > 5000:
        rng = np.random.default_rng(42)
        test_data = rng.choice(sample_means, size=5000, replace=False)
        note = "More than 5000 sample means detected, so Shapiro-Wilk was run on a random subset of 5000."

    statistic, p_value = stats.shapiro(test_data)

    if p_value < alpha:
        decision = "Reject H0"
        conclusion = "The sample means do not appear perfectly normally distributed."
    else:
        decision = "Fail to Reject H0"
        conclusion = "There is not enough evidence to say the sample means are non-normal."

    result = {
        "Test": "Shapiro-Wilk on Sample Means",
        "Statistic": statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
        "Note": note
    }

    return result


def create_sample_means_normality_table(result):
    """
    Converts Shapiro result into a table.
    """

    rows = []

    for key, value in result.items():
        rows.append({
            "Metric": key,
            "Value": format_number(value)
        })

    return pd.DataFrame(rows)


def get_clt_interpretation(data, sample_means, sample_size, number_of_samples):
    """
    Gives plain-English CLT interpretation.
    """

    original_mean = data.mean()
    original_std = data.std(ddof=1)

    sample_means_mean = sample_means.mean()
    sample_means_std = sample_means.std(ddof=1)

    theoretical_standard_error = original_std / np.sqrt(sample_size)

    interpretation = []

    interpretation.append(
        f"We repeatedly took {number_of_samples} random samples, each with sample size {sample_size}."
    )

    interpretation.append(
        "For each random sample, we calculated the sample mean."
    )

    interpretation.append(
        f"The original data mean is {format_number(original_mean)}, while the mean of the sample means is {format_number(sample_means_mean)}."
    )

    interpretation.append(
        "According to the Central Limit Theorem, the mean of the sample means should get close to the original population/sample mean."
    )

    interpretation.append(
        f"The standard deviation of the original data is {format_number(original_std)}."
    )

    interpretation.append(
        f"The theoretical standard error is {format_number(theoretical_standard_error)}, and the observed standard deviation of sample means is {format_number(sample_means_std)}."
    )

    interpretation.append(
        "As sample size increases, the sampling distribution of the mean usually becomes narrower and more normal-shaped."
    )

    interpretation.append(
        "This can happen even when the original data distribution is skewed or not perfectly normal."
    )

    return interpretation