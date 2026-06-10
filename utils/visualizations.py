import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go


# ------------------------------------------------------------
# SandeepStician chart theme
# ------------------------------------------------------------

CHART_COLORS = ["#56B4E9", "#D55E00", "#009E73", "#E69F00"]

PLOT_COLORS = {
    "primary": "#56B4E9",       # Sky blue
    "secondary": "#D55E00",     # Vermillion
    "green": "#009E73",         # Bluish green
    "warning": "#E69F00",       # Warm orange
    "orange": "#E69F00",
    "blue": "#56B4E9",
    "error": "#C2410C",

    # Soft professional light chart canvas
    "bg": "#F7F9FC",
    "paper": "#FFFFFF",
    "card": "#FFFFFF",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
    "subtle": "#9CA3AF",

    # Transparent fills
    "primary_fill": "rgba(86, 180, 233, 0.22)",
    "secondary_fill": "rgba(213, 94, 0, 0.18)",
    "green_fill": "rgba(0, 158, 115, 0.18)",
    "warning_fill": "rgba(230, 159, 0, 0.20)",
}


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=420):
    """
    Applies the soft professional SandeepStician Plotly chart theme.
    """

    fig.update_layout(
        title={
            "text": title if title else "",
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 18,
                "color": PLOT_COLORS["text"],
                "family": "Arial",
            },
        },
        paper_bgcolor=PLOT_COLORS["paper"],
        plot_bgcolor=PLOT_COLORS["paper"],
        font={
            "color": PLOT_COLORS["text"],
            "family": "Arial",
        },
        height=height,
        margin={
            "l": 55,
            "r": 30,
            "t": 65,
            "b": 55,
        },
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {
                "color": PLOT_COLORS["muted"],
                "size": 12,
            },
        },
    )

    fig.update_xaxes(
        title_text=x_title,
        showgrid=True,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["axis"],
        linecolor=PLOT_COLORS["axis"],
        mirror=False,
        ticks="outside",
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
    )

    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["axis"],
        linecolor=PLOT_COLORS["axis"],
        mirror=False,
        ticks="outside",
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["muted"]},
    )

    return fig


def prepare_numeric_data(df, column):
    """
    Converts a selected column into numeric data and removes missing values.
    """

    numeric_data = pd.to_numeric(df[column], errors="coerce")
    clean_data = numeric_data.dropna()

    if clean_data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    return clean_data


def is_discrete_numeric(clean_data):
    """
    Detects whether a numerical column looks discrete.
    Example: counts, ratings, number of children, etc.
    """

    unique_values = clean_data.nunique()
    total_values = len(clean_data)

    all_integer_like = np.all(clean_data == clean_data.astype(int))

    if all_integer_like and unique_values <= 30:
        return True

    if unique_values / total_values < 0.05 and all_integer_like:
        return True

    return False


def plot_histogram(df, column, bins=20):
    """
    Creates an interactive histogram for a numerical column.
    """

    clean_data = prepare_numeric_data(df, column)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=clean_data,
            nbinsx=bins,
            marker={
                "color": PLOT_COLORS["primary"],
                "line": {
                    "color": "#FFFFFF",
                    "width": 1,
                },
            },
            opacity=0.88,
            name="Frequency",
            hovertemplate=f"{column}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    mean_value = clean_data.mean()
    median_value = clean_data.median()

    fig.add_vline(
        x=mean_value,
        line_width=2.5,
        line_dash="dash",
        line_color=PLOT_COLORS["green"],
        annotation_text="Mean",
        annotation_position="top left",
        annotation_font_color=PLOT_COLORS["green"],
    )

    fig.add_vline(
        x=median_value,
        line_width=2.5,
        line_dash="dot",
        line_color=PLOT_COLORS["warning"],
        annotation_text="Median",
        annotation_position="top right",
        annotation_font_color=PLOT_COLORS["warning"],
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Histogram of {column}",
        x_title=column,
        y_title="Frequency",
        height=420,
    )

    return fig


def plot_boxplot(df, column):
    """
    Creates an interactive boxplot for a numerical column.
    """

    clean_data = prepare_numeric_data(df, column)

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=clean_data,
            name=column,
            boxmean=True,
            marker={
                "color": PLOT_COLORS["secondary"],
                "opacity": 0.78,
            },
            line={
                "color": PLOT_COLORS["primary"],
                "width": 2,
            },
            fillcolor=PLOT_COLORS["primary_fill"],
            hovertemplate=f"{column}: %{{x}}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Boxplot of {column}",
        x_title=column,
        y_title="",
        height=360,
    )

    return fig


def plot_kde_pdf(df, column):
    """
    Creates an interactive KDE curve, which estimates the PDF for continuous data.
    """

    clean_data = prepare_numeric_data(df, column)

    fig = go.Figure()

    if clean_data.nunique() < 2:
        fig.add_annotation(
            text="KDE/PDF cannot be plotted because all values are the same.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={
                "size": 15,
                "color": PLOT_COLORS["muted"],
            },
        )

        fig = apply_plotly_theme(
            fig,
            title=f"Estimated PDF / KDE of {column}",
            x_title=column,
            y_title="Density",
            height=400,
        )

        return fig

    kde = stats.gaussian_kde(clean_data)

    x_values = np.linspace(clean_data.min(), clean_data.max(), 400)
    y_values = kde(x_values)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line={
                "color": PLOT_COLORS["primary"],
                "width": 3,
            },
            fill="tozeroy",
            fillcolor=PLOT_COLORS["primary_fill"],
            name="Estimated PDF / KDE",
            hovertemplate=f"{column}: %{{x:.4f}}<br>Density: %{{y:.6f}}<extra></extra>",
        )
    )

    peak_index = np.argmax(y_values)
    peak_x = x_values[peak_index]
    peak_y = y_values[peak_index]

    fig.add_trace(
        go.Scatter(
            x=[peak_x],
            y=[peak_y],
            mode="markers",
            marker={
                "size": 10,
                "color": PLOT_COLORS["secondary"],
                "line": {
                    "color": "#FFFFFF",
                    "width": 1.5,
                },
            },
            name="Peak density",
            hovertemplate=f"Peak around: %{{x:.4f}}<br>Density: %{{y:.6f}}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Estimated PDF / KDE of {column}",
        x_title=column,
        y_title="Density",
        height=420,
    )

    return fig


def plot_cdf(df, column):
    """
    Creates an interactive empirical CDF plot.
    """

    clean_data = prepare_numeric_data(df, column)

    sorted_data = np.sort(clean_data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=cdf_values,
            mode="lines+markers",
            line={
                "color": PLOT_COLORS["primary"],
                "width": 3,
            },
            marker={
                "size": 5,
                "color": PLOT_COLORS["green"],
                "opacity": 0.75,
            },
            name="Empirical CDF",
            hovertemplate=f"{column}: %{{x:.4f}}<br>CDF: %{{y:.4f}}<extra></extra>",
        )
    )

    q25 = clean_data.quantile(0.25)
    q50 = clean_data.quantile(0.50)
    q75 = clean_data.quantile(0.75)

    for q_value, label, color in [
        (q25, "25%", PLOT_COLORS["primary"]),
        (q50, "50% / Median", PLOT_COLORS["green"]),
        (q75, "75%", PLOT_COLORS["warning"]),
    ]:
        fig.add_vline(
            x=q_value,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
            annotation_font_color=color,
        )

    fig = apply_plotly_theme(
        fig,
        title=f"Empirical CDF of {column}",
        x_title=column,
        y_title="Cumulative Probability",
        height=420,
    )

    fig.update_yaxes(range=[0, 1.02])

    return fig


def plot_qq(df, column):
    """
    Creates an interactive Q-Q plot against the normal distribution.
    This helps check whether data is approximately normal.
    """

    clean_data = prepare_numeric_data(df, column)

    osm, osr = stats.probplot(clean_data, dist="norm", fit=False)
    fit_result = stats.probplot(clean_data, dist="norm", fit=True)
    slope, intercept, r_value = fit_result[1]

    osm = np.array(osm)
    osr = np.array(osr)

    line_x = np.linspace(osm.min(), osm.max(), 200)
    line_y = slope * line_x + intercept

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker={
                "size": 7,
                "color": PLOT_COLORS["primary"],
                "opacity": 0.84,
                "line": {
                    "color": "#FFFFFF",
                    "width": 0.8,
                },
            },
            name="Observed quantiles",
            hovertemplate="Theoretical: %{x:.4f}<br>Observed: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={
                "color": PLOT_COLORS["secondary"],
                "width": 2.5,
                "dash": "dash",
            },
            name=f"Reference line | R²={r_value ** 2:.4f}",
            hovertemplate="Reference line<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Q-Q Plot of {column} Against Normal Distribution",
        x_title="Theoretical Quantiles",
        y_title="Ordered Values",
        height=460,
    )

    return fig


def plot_pmf(df, column):
    """
    Creates an interactive PMF-like bar chart for discrete numerical data.
    """

    clean_data = prepare_numeric_data(df, column)

    value_counts = clean_data.value_counts(normalize=True).sort_index()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker={
                "color": PLOT_COLORS["green"],
                "line": {
                    "color": "#FFFFFF",
                    "width": 1,
                },
            },
            name="Probability",
            hovertemplate=f"{column}: %{{x}}<br>Probability: %{{y:.4f}}<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"PMF of {column}",
        x_title=column,
        y_title="Probability",
        height=420,
    )

    fig.update_yaxes(range=[0, max(value_counts.values) * 1.15])

    return fig


# ------------------------------------------------------------
# General visualization interpretation
# ------------------------------------------------------------

def get_visualization_interpretation(df, column):
    """
    Gives simple interpretation notes for the visualizations.
    """

    clean_data = prepare_numeric_data(df, column)

    mean_value = clean_data.mean()
    median_value = clean_data.median()
    skewness = clean_data.skew()

    interpretations = []

    if mean_value > median_value:
        interpretations.append(
            "The mean is greater than the median, so the distribution may be right-skewed."
        )
    elif mean_value < median_value:
        interpretations.append(
            "The mean is less than the median, so the distribution may be left-skewed."
        )
    else:
        interpretations.append(
            "The mean and median are very close, suggesting the distribution may be fairly balanced."
        )

    if skewness > 1:
        interpretations.append("The skewness value suggests strong right-skewness.")
    elif skewness > 0.5:
        interpretations.append("The skewness value suggests moderate right-skewness.")
    elif skewness >= -0.5:
        interpretations.append("The skewness value suggests the data is approximately symmetric.")
    elif skewness >= -1:
        interpretations.append("The skewness value suggests moderate left-skewness.")
    else:
        interpretations.append("The skewness value suggests strong left-skewness.")

    if is_discrete_numeric(clean_data):
        interpretations.append(
            "This variable looks discrete, so a PMF-style plot is useful."
        )
    else:
        interpretations.append(
            "This variable looks continuous, so histogram, KDE/PDF, CDF, and Q-Q plots are useful."
        )

    return interpretations


# ------------------------------------------------------------
# Plot-specific interpretations
# ------------------------------------------------------------

def format_number(value):
    """
    Formats numbers nicely for interpretation text.
    """

    try:
        return round(float(value), 4)
    except Exception:
        return value


def interpret_histogram(df, column):
    """
    Gives a specific interpretation for the histogram.
    """

    clean_data = prepare_numeric_data(df, column)

    mean_value = clean_data.mean()
    median_value = clean_data.median()
    skewness = clean_data.skew()

    counts, bin_edges = np.histogram(clean_data, bins=10)
    highest_bin_index = np.argmax(counts)

    bin_start = bin_edges[highest_bin_index]
    bin_end = bin_edges[highest_bin_index + 1]

    interpretations = [
        f"The highest frequency appears around the range {format_number(bin_start)} to {format_number(bin_end)}.",
        f"The mean is {format_number(mean_value)} and the median is {format_number(median_value)}.",
    ]

    if skewness > 0.5:
        interpretations.append(
            "The histogram suggests right-skewness, meaning there may be some larger values stretching the right tail."
        )
    elif skewness < -0.5:
        interpretations.append(
            "The histogram suggests left-skewness, meaning there may be some smaller values stretching the left tail."
        )
    else:
        interpretations.append(
            "The histogram looks fairly balanced based on the skewness value."
        )

    return interpretations


def interpret_boxplot(df, column):
    """
    Gives a specific interpretation for the boxplot.
    """

    clean_data = prepare_numeric_data(df, column)

    q1 = clean_data.quantile(0.25)
    median = clean_data.quantile(0.50)
    q3 = clean_data.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = clean_data[(clean_data < lower_fence) | (clean_data > upper_fence)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(clean_data)) * 100

    interpretations = [
        f"The middle 50% of the data lies between Q1 = {format_number(q1)} and Q3 = {format_number(q3)}.",
        f"The median value is {format_number(median)}.",
        f"The IQR is {format_number(iqr)}, which represents the spread of the middle half of the data.",
    ]

    if outlier_count > 0:
        interpretations.append(
            f"The boxplot detects {outlier_count} possible outlier(s), around {format_number(outlier_percentage)}% of the valid values."
        )
    else:
        interpretations.append(
            "The boxplot does not detect clear outliers using the 1.5 × IQR rule."
        )

    return interpretations


def interpret_kde_pdf(df, column):
    """
    Gives a specific interpretation for the KDE / estimated PDF.
    """

    clean_data = prepare_numeric_data(df, column)

    interpretations = []

    if is_discrete_numeric(clean_data):
        interpretations.append(
            "This variable looks discrete, so the KDE/PDF curve can be less reliable. A PMF plot may explain this variable better."
        )

    if clean_data.nunique() < 2:
        return ["KDE/PDF cannot be meaningfully interpreted because all values are the same."]

    kde = stats.gaussian_kde(clean_data)
    x_values = np.linspace(clean_data.min(), clean_data.max(), 300)
    y_values = kde(x_values)

    peak_index = np.argmax(y_values)
    peak_x = x_values[peak_index]

    interpretations.append(
        f"The highest estimated density is around {format_number(peak_x)}."
    )

    interpretations.append(
        "Areas where the curve is higher represent values that appear more commonly in the data."
    )

    interpretations.append(
        "Areas where the curve is lower represent values that appear less commonly."
    )

    return interpretations


def interpret_cdf(df, column):
    """
    Gives a specific interpretation for the CDF.
    """

    clean_data = prepare_numeric_data(df, column)

    q25 = clean_data.quantile(0.25)
    q50 = clean_data.quantile(0.50)
    q75 = clean_data.quantile(0.75)

    interpretations = [
        f"About 25% of the values are less than or equal to {format_number(q25)}.",
        f"About 50% of the values are less than or equal to {format_number(q50)}. This is the median.",
        f"About 75% of the values are less than or equal to {format_number(q75)}.",
        "A steep part of the CDF means many values are concentrated in that range.",
    ]

    return interpretations


def interpret_qq_plot(df, column):
    """
    Gives a specific interpretation for the Q-Q plot against normal distribution.
    """

    clean_data = prepare_numeric_data(df, column)

    probplot_result = stats.probplot(clean_data, dist="norm")
    r_value = probplot_result[1][2]
    r_squared = r_value ** 2

    skewness = clean_data.skew()
    kurtosis = clean_data.kurt()

    interpretations = [
        f"The Q-Q plot R² value is approximately {format_number(r_squared)}."
    ]

    if r_squared >= 0.98:
        interpretations.append(
            "The points are very close to the reference line, so the data looks close to normally distributed."
        )
    elif r_squared >= 0.95:
        interpretations.append(
            "The points follow the reference line reasonably well, but there may be some deviations from normality."
        )
    else:
        interpretations.append(
            "The points deviate noticeably from the reference line, so the data may not be normally distributed."
        )

    if skewness > 0.5:
        interpretations.append(
            "The data has positive skewness, so the right tail may be affecting the Q-Q plot."
        )
    elif skewness < -0.5:
        interpretations.append(
            "The data has negative skewness, so the left tail may be affecting the Q-Q plot."
        )
    else:
        interpretations.append(
            "The skewness is close to 0, so the distribution is not strongly skewed."
        )

    if kurtosis > 1:
        interpretations.append(
            "The excess kurtosis is high, suggesting heavier tails or possible extreme values."
        )
    elif kurtosis < -1:
        interpretations.append(
            "The excess kurtosis is low, suggesting a flatter distribution with lighter tails."
        )
    else:
        interpretations.append(
            "The kurtosis is not extremely different from a normal distribution."
        )

    return interpretations


def interpret_pmf(df, column):
    """
    Gives a specific interpretation for the PMF.
    """

    clean_data = prepare_numeric_data(df, column)

    value_probabilities = clean_data.value_counts(normalize=True).sort_values(ascending=False)

    top_value = value_probabilities.index[0]
    top_probability = value_probabilities.iloc[0]

    interpretations = [
        f"The most likely value is {format_number(top_value)}.",
        f"This value appears with probability approximately {format_number(top_probability)}.",
        f"The variable has {clean_data.nunique()} unique value(s).",
    ]

    if clean_data.nunique() <= 10:
        interpretations.append(
            "Because there are few unique values, the PMF is a good way to show this variable."
        )
    else:
        interpretations.append(
            "There are many unique values, so the PMF may become harder to read."
        )

    return interpretations


def get_plot_interpretation(df, column, plot_type):
    """
    Returns interpretation depending on the selected plot type.
    """

    if plot_type == "Histogram":
        return interpret_histogram(df, column)

    if plot_type == "Boxplot":
        return interpret_boxplot(df, column)

    if plot_type == "Estimated PDF / KDE":
        return interpret_kde_pdf(df, column)

    if plot_type == "CDF":
        return interpret_cdf(df, column)

    if plot_type == "Q-Q Plot":
        return interpret_qq_plot(df, column)

    if plot_type == "PMF":
        return interpret_pmf(df, column)

    return ["No interpretation available for this plot."]
