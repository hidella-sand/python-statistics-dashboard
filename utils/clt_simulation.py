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


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=500):
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
        margin={"l": 70, "r": 35, "t": 70, "b": 60},
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
    Formats numbers nicely for tables and interpretations.
    """

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if value == 0:
            return "0"

        if abs(value) < 0.000001:
            return f"{value:.2e}"

        return round(value, 6)

    except Exception:
        return value


def format_p_value(value):
    """
    Formats p-values nicely.
    """

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if value == 0:
            return "< 1e-300"

        if value < 0.000001:
            return f"{value:.2e}"

        return round(value, 6)

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


def simulate_sample_means(
    df,
    column,
    sample_size=30,
    number_of_samples=1000,
    random_seed=42
):
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

    data_values = data.values

    sample_means = rng.choice(
        data_values,
        size=(number_of_samples, sample_size),
        replace=True
    ).mean(axis=1)

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
            "Value": format_number(original_mean),
        },
        {
            "Metric": "Original Data Standard Deviation",
            "Value": format_number(original_std),
        },
        {
            "Metric": "Sample Size Used",
            "Value": sample_size,
        },
        {
            "Metric": "Mean of Sample Means",
            "Value": format_number(sample_means_mean),
        },
        {
            "Metric": "Standard Deviation of Sample Means",
            "Value": format_number(sample_means_std),
        },
        {
            "Metric": "Theoretical Standard Error",
            "Value": format_number(theoretical_standard_error),
        },
        {
            "Metric": "Difference: Original Mean vs Mean of Sample Means",
            "Value": format_number(abs(original_mean - sample_means_mean)),
        },
    ]

    return pd.DataFrame(rows)


def plot_original_distribution(data, column, bins=30):
    """
    Plotly histogram for the original data distribution.
    """

    data_array = np.asarray(data)

    mean_value = np.mean(data_array)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data_array,
            nbinsx=bins,
            name="Original Data",
            marker={
                "color": PLOT_COLORS["primary"],
                "opacity": 0.65,
                "line": {
                    "color": PLOT_COLORS["grid"],
                    "width": 1,
                },
            },
            hovertemplate="Value: %{x}<br>Frequency: %{y}<extra></extra>",
        )
    )

    fig.add_vline(
        x=mean_value,
        line_width=3,
        line_dash="dash",
        line_color=PLOT_COLORS["green"],
        annotation_text=f"Mean = {format_number(mean_value)}",
        annotation_position="top right",
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Original Distribution of {column}",
        x_title=column,
        y_title="Frequency",
        height=500,
    )

    return fig


def plot_sampling_distribution(
    sample_means,
    original_mean,
    theoretical_standard_error,
    bins=30
):
    """
    Plotly distribution of sample means with a normal curve overlay.
    """

    sample_means_array = np.asarray(sample_means)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=sample_means_array,
            nbinsx=bins,
            histnorm="probability density",
            name="Sample Means",
            marker={
                "color": PLOT_COLORS["primary"],
                "opacity": 0.55,
                "line": {
                    "color": PLOT_COLORS["grid"],
                    "width": 1,
                },
            },
            hovertemplate="Sample Mean: %{x}<br>Density: %{y}<extra></extra>",
        )
    )

    x_values = np.linspace(
        sample_means_array.min(),
        sample_means_array.max(),
        500
    )

    if theoretical_standard_error > 0:
        normal_curve = stats.norm.pdf(
            x_values,
            loc=original_mean,
            scale=theoretical_standard_error
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=normal_curve,
                mode="lines",
                name="Normal Curve using CLT",
                line={
                    "color": PLOT_COLORS["green"],
                    "width": 3,
                },
                hovertemplate=(
                    "Sample Mean: %{x:.4f}<br>"
                    "Normal Density: %{y:.6f}"
                    "<extra></extra>"
                ),
            )
        )

    sample_means_mean = sample_means_array.mean()

    fig.add_vline(
        x=sample_means_mean,
        line_width=3,
        line_dash="dash",
        line_color=PLOT_COLORS["warning"],
        annotation_text=f"Mean = {format_number(sample_means_mean)}",
        annotation_position="top right",
    )

    fig = apply_plotly_theme(
        fig,
        title="Sampling Distribution of the Mean",
        x_title="Sample Mean",
        y_title="Density",
        height=500,
    )

    return fig


def simulate_multiple_sample_sizes(
    df,
    column,
    sample_sizes,
    number_of_samples=1000,
    random_seed=42
):
    """
    Simulates sample means for multiple sample sizes.
    Useful to show how the sampling distribution changes when sample size increases.
    """

    data = prepare_numeric_data(df, column)

    results = {}

    for sample_size in sample_sizes:
        if sample_size < 2:
            continue

        _, sample_means = simulate_sample_means(
            df,
            column,
            sample_size=sample_size,
            number_of_samples=number_of_samples,
            random_seed=random_seed
        )

        results[sample_size] = sample_means

    if len(results) == 0:
        raise ValueError("Please select at least one valid sample size greater than or equal to 2.")

    return data, results


def plot_sample_size_comparison(sample_size_results):
    """
    Plotly KDE comparison of sampling distributions for different sample sizes.
    """

    fig = go.Figure()

    color_cycle = [
        PLOT_COLORS["primary"],
        PLOT_COLORS["green"],
        PLOT_COLORS["warning"],
        PLOT_COLORS["blue"],
        PLOT_COLORS["secondary"],
        PLOT_COLORS["error"],
    ]

    for index, (sample_size, sample_means) in enumerate(sample_size_results.items()):
        sample_means_array = np.asarray(sample_means)

        if len(sample_means_array) < 2:
            continue

        if np.std(sample_means_array, ddof=1) == 0:
            mean_value = sample_means_array.mean()

            fig.add_vline(
                x=mean_value,
                line_width=2,
                line_dash="dash",
                line_color=color_cycle[index % len(color_cycle)],
                annotation_text=f"n = {sample_size}",
            )

            continue

        kde = stats.gaussian_kde(sample_means_array)

        x_values = np.linspace(
            sample_means_array.min(),
            sample_means_array.max(),
            400
        )

        y_values = kde(x_values)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=f"n = {sample_size}",
                line={
                    "width": 3,
                    "color": color_cycle[index % len(color_cycle)],
                },
                hovertemplate=(
                    "Sample Mean: %{x:.4f}<br>"
                    "Density: %{y:.6f}"
                    "<extra></extra>"
                ),
            )
        )

    fig = apply_plotly_theme(
        fig,
        title="Effect of Sample Size on Sampling Distribution",
        x_title="Sample Mean",
        y_title="Density",
        height=520,
    )

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
        "Note": note,
    }

    return result


def create_sample_means_normality_table(result):
    """
    Converts Shapiro result into a table.
    """

    rows = []

    for key, value in result.items():
        if key == "p-value":
            formatted_value = format_p_value(value)
        else:
            formatted_value = format_number(value)

        rows.append({
            "Metric": key,
            "Value": formatted_value,
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