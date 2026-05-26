import pandas as pd
import numpy as np


def prepare_numeric_data(df, column):
    """
    Converts a selected column into numeric data and removes missing values.
    This makes sure the calculations work safely.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")

    numeric_data = pd.to_numeric(df[column], errors="coerce")
    clean_data = numeric_data.dropna()

    if clean_data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    return numeric_data, clean_data


def get_mode_value(clean_data):
    """
    Returns the mode of the selected numerical column.
    If there is no clear mode, returns a suitable message.
    """

    mode_values = clean_data.mode()

    if mode_values.empty:
        return "No mode"

    # If every value appears only once, there is no useful mode
    if len(mode_values) == len(clean_data):
        return "No clear mode"

    # If there are multiple modes, show first few
    if len(mode_values) > 3:
        return ", ".join([str(round(value, 4)) for value in mode_values.head(3)]) + " ..."

    return ", ".join([str(round(value, 4)) for value in mode_values])


def calculate_descriptive_statistics(df, column):
    """
    Calculates main descriptive statistics for one numerical column.
    """

    numeric_data, clean_data = prepare_numeric_data(df, column)

    total_count = len(numeric_data)
    valid_count = len(clean_data)
    missing_count = int(numeric_data.isna().sum())
    missing_percentage = round((missing_count / total_count) * 100, 2) if total_count > 0 else 0

    minimum = clean_data.min()
    maximum = clean_data.max()

    stats = {
        "Column": column,
        "Total Rows": total_count,
        "Valid Numerical Values": valid_count,
        "Missing / Invalid Values": missing_count,
        "Missing / Invalid %": missing_percentage,
        "Mean": clean_data.mean(),
        "Median": clean_data.median(),
        "Mode": get_mode_value(clean_data),
        "Minimum": minimum,
        "Maximum": maximum,
        "Range": maximum - minimum,
        "Sample Variance": clean_data.var(ddof=1),
        "Sample Standard Deviation": clean_data.std(ddof=1),
        "Skewness": clean_data.skew(),
        "Excess Kurtosis": clean_data.kurt()
    }

    return stats


def create_descriptive_stats_table(stats):
    """
    Converts the descriptive statistics dictionary into a clean table.
    """

    rows = []

    for key, value in stats.items():
        if key == "Column":
            continue

        if isinstance(value, (int, np.integer)):
            formatted_value = value
        elif isinstance(value, (float, np.floating)):
            formatted_value = round(value, 4)
        else:
            formatted_value = value

        rows.append({
            "Statistic": key,
            "Value": formatted_value
        })

    return pd.DataFrame(rows)


def interpret_skewness(skewness):
    """
    Gives a simple interpretation of skewness.
    """

    if skewness > 1:
        return "The data is strongly right-skewed. This means most values are lower, with a longer tail on the right side."
    elif skewness > 0.5:
        return "The data is moderately right-skewed. Some larger values are pulling the distribution to the right."
    elif skewness >= -0.5:
        return "The data is approximately symmetric based on skewness."
    elif skewness >= -1:
        return "The data is moderately left-skewed. Some smaller values are pulling the distribution to the left."
    else:
        return "The data is strongly left-skewed. This means most values are higher, with a longer tail on the left side."


def interpret_kurtosis(kurtosis):
    """
    Gives a simple interpretation of excess kurtosis.
    """

    if kurtosis > 1:
        return "The distribution has heavy tails or possible extreme values compared with a normal distribution."
    elif kurtosis < -1:
        return "The distribution is flatter and has lighter tails compared with a normal distribution."
    else:
        return "The tail behavior is not extremely different from a normal distribution."


def interpret_mean_median(mean, median, std):
    """
    Compares mean and median to explain possible skewness.
    """

    if std == 0:
        return "The standard deviation is 0, meaning all values are the same."

    difference = abs(mean - median)

    if difference < 0.1 * std:
        return "The mean and median are close, suggesting the data may be fairly balanced."

    if mean > median:
        return "The mean is greater than the median, suggesting possible right-skewness."

    return "The mean is less than the median, suggesting possible left-skewness."


def interpret_descriptive_statistics(stats):
    """
    Creates simple plain-English interpretations from the statistics.
    """

    interpretations = []

    mean = stats["Mean"]
    median = stats["Median"]
    std = stats["Sample Standard Deviation"]
    skewness = stats["Skewness"]
    kurtosis = stats["Excess Kurtosis"]
    missing_percentage = stats["Missing / Invalid %"]

    interpretations.append(interpret_mean_median(mean, median, std))
    interpretations.append(interpret_skewness(skewness))
    interpretations.append(interpret_kurtosis(kurtosis))

    if missing_percentage > 0:
        interpretations.append(
            f"{missing_percentage}% of the values are missing or invalid, so this should be considered before deeper analysis."
        )
    else:
        interpretations.append("There are no missing or invalid values in this selected numerical column.")

    return interpretations