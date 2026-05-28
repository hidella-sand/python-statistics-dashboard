import pandas as pd
import numpy as np
from scipy import stats


def prepare_normality_data(df, column):
    """
    Converts selected column into numeric data and removes missing values.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")

    numeric_data = pd.to_numeric(df[column], errors="coerce")
    clean_data = numeric_data.dropna()

    if clean_data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    return clean_data


def interpret_p_value(p_value, alpha=0.05):
    """
    Interprets a p-value using the selected alpha level.
    """

    if p_value < alpha:
        return "Reject H0", "The data does not appear to be normally distributed."
    else:
        return "Fail to Reject H0", "There is not enough evidence to say the data is non-normal."


def run_shapiro_test(clean_data, alpha=0.05):
    """
    Runs Shapiro-Wilk normality test.

    H0: Data is normally distributed.
    H1: Data is not normally distributed.
    """

    if len(clean_data) < 3:
        return {
            "Test": "Shapiro-Wilk",
            "Statistic": None,
            "p-value": None,
            "Decision": "Not enough data",
            "Conclusion": "Shapiro-Wilk test requires at least 3 valid values.",
            "Note": "Minimum sample size requirement not met."
        }

    note = "Test performed on full data."

    # Shapiro can be less reliable for very large samples.
    # We use a sample of 5000 values to keep it stable.
    test_data = clean_data

    if len(clean_data) > 5000:
        test_data = clean_data.sample(n=5000, random_state=42)
        note = "Dataset has more than 5000 values, so Shapiro-Wilk was run on a random sample of 5000."

    statistic, p_value = stats.shapiro(test_data)
    decision, conclusion = interpret_p_value(p_value, alpha)

    return {
        "Test": "Shapiro-Wilk",
        "Statistic": statistic,
        "p-value": p_value,
        "Decision": decision,
        "Conclusion": conclusion,
        "Note": note
    }


def run_ks_test(clean_data, alpha=0.05):
    """
    Runs Kolmogorov-Smirnov test against a normal distribution
    using the sample mean and standard deviation.

    H0: Data follows a normal distribution.
    H1: Data does not follow a normal distribution.
    """

    if len(clean_data) < 2:
        return {
            "Test": "Kolmogorov-Smirnov",
            "Statistic": None,
            "p-value": None,
            "Decision": "Not enough data",
            "Conclusion": "KS test requires at least 2 valid values.",
            "Note": "Minimum sample size requirement not met."
        }

    mean_value = clean_data.mean()
    std_value = clean_data.std(ddof=1)

    if std_value == 0:
        return {
            "Test": "Kolmogorov-Smirnov",
            "Statistic": None,
            "p-value": None,
            "Decision": "Cannot test",
            "Conclusion": "All values are the same, so normality cannot be tested meaningfully.",
            "Note": "Standard deviation is 0."
        }

    statistic, p_value = stats.kstest(
        clean_data,
        "norm",
        args=(mean_value, std_value)
    )

    decision, conclusion = interpret_p_value(p_value, alpha)

    return {
        "Test": "Kolmogorov-Smirnov",
        "Statistic": statistic,
        "p-value": p_value,
        "Decision": decision,
        "Conclusion": conclusion,
        "Note": "Test compares data against a normal distribution using the sample mean and standard deviation."
    }


def run_anderson_darling_test(clean_data, alpha=0.05):
    """
    Runs Anderson-Darling normality test.

    H0: Data is normally distributed.
    H1: Data is not normally distributed.

    Anderson-Darling uses critical values instead of a p-value.
    """

    if len(clean_data) < 3:
        return {
            "Test": "Anderson-Darling",
            "Statistic": None,
            "p-value": "Not provided",
            "Decision": "Not enough data",
            "Conclusion": "Anderson-Darling test requires at least 3 valid values.",
            "Note": "Minimum sample size requirement not met.",
            "Critical Value Used": None,
            "Significance Level Used": None
        }

    result = stats.anderson(clean_data, dist="norm")

    statistic = result.statistic
    critical_values = result.critical_values
    significance_levels = result.significance_level

    alpha_percentage = alpha * 100

    # Find the closest available Anderson-Darling significance level
    closest_index = np.argmin(np.abs(significance_levels - alpha_percentage))

    critical_value_used = critical_values[closest_index]
    significance_level_used = significance_levels[closest_index]

    if statistic > critical_value_used:
        decision = "Reject H0"
        conclusion = "The data does not appear to be normally distributed."
    else:
        decision = "Fail to Reject H0"
        conclusion = "There is not enough evidence to say the data is non-normal."

    return {
        "Test": "Anderson-Darling",
        "Statistic": statistic,
        "p-value": "Not provided",
        "Decision": decision,
        "Conclusion": conclusion,
        "Note": "This test uses critical values instead of a p-value.",
        "Critical Value Used": critical_value_used,
        "Significance Level Used": f"{significance_level_used}%"
    }


def run_all_normality_tests(df, column, alpha=0.05):
    """
    Runs Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests.
    """

    clean_data = prepare_normality_data(df, column)

    results = [
        run_shapiro_test(clean_data, alpha),
        run_ks_test(clean_data, alpha),
        run_anderson_darling_test(clean_data, alpha)
    ]

    return results


def create_normality_results_table(results):
    """
    Converts normality test results into a clean table.
    """

    rows = []

    for result in results:
        statistic = result["Statistic"]
        p_value = result["p-value"]

        if isinstance(statistic, (float, np.floating)):
            statistic = round(statistic, 5)

        if isinstance(p_value, (float, np.floating)):
            p_value = round(p_value, 5)

        rows.append({
            "Test": result["Test"],
            "Statistic": statistic,
            "p-value": p_value,
            "Decision": result["Decision"],
            "Conclusion": result["Conclusion"]
        })

    return pd.DataFrame(rows)


def get_overall_normality_conclusion(results):
    """
    Creates an overall conclusion based on all normality tests.
    """

    reject_count = 0
    valid_test_count = 0

    for result in results:
        if result["Decision"] in ["Reject H0", "Fail to Reject H0"]:
            valid_test_count += 1

            if result["Decision"] == "Reject H0":
                reject_count += 1

    if valid_test_count == 0:
        return [
            "Normality could not be tested because there were not enough valid values.",
            "Try selecting another numerical column."
        ]

    if reject_count == 0:
        return [
            "None of the valid normality tests rejected H0.",
            "The selected variable may be approximately normally distributed.",
            "Still, always check the histogram and Q-Q plot before making a final decision."
        ]

    if reject_count == valid_test_count:
        return [
            "All valid normality tests rejected H0.",
            "This strongly suggests the selected variable is not normally distributed.",
            "For tests that require normality, consider transformation, non-parametric tests, or checking sample size effects."
        ]

    return [
        f"{reject_count} out of {valid_test_count} valid test(s) rejected H0.",
        "The evidence is mixed.",
        "Use the Q-Q plot, histogram, sample size, and subject knowledge before making a final decision."
    ]


def get_test_explanation():
    """
    Gives explanation of the null and alternative hypotheses.
    """

    return {
        "H0": "The selected variable follows a normal distribution.",
        "H1": "The selected variable does not follow a normal distribution."
    }