import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
import plotly.graph_objects as go


PLOT_COLORS = {
    # Main color-blind-friendly palette requested for SandeepStician
    "primary": "#56B4E9",      # sky blue
    "secondary": "#D55E00",    # vermillion
    "green": "#009E73",        # bluish green
    "warning": "#E69F00",      # warm orange

    # Supporting UI/chart colors
    "blue": "#56B4E9",
    "orange": "#E69F00",
    "error": "#C62828",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
    "soft_blue": "rgba(86, 180, 233, 0.28)",
    "soft_green": "rgba(0, 158, 115, 0.18)",
    "soft_orange": "rgba(230, 159, 0, 0.20)",
    "soft_red": "rgba(213, 94, 0, 0.18)",
}

CHART_PALETTE = [
    "#56B4E9",
    "#D55E00",
    "#009E73",
    "#E69F00",
]


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=500):
    """
    Applies a soft professional light theme to Plotly figures.
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
        paper_bgcolor=PLOT_COLORS["bg"],
        plot_bgcolor=PLOT_COLORS["card"],
        font={
            "color": PLOT_COLORS["text"],
            "family": "Arial",
        },
        height=height,
        margin={
            "l": 70,
            "r": 35,
            "t": 70,
            "b": 60,
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
            },
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": PLOT_COLORS["grid"],
            "borderwidth": 1,
        },
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font": {
                "color": PLOT_COLORS["text"],
                "family": "Arial",
            },
            "bordercolor": PLOT_COLORS["grid"],
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

    return fig


def format_number(value):
    """
    Formats numbers nicely for tables.
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
        raise ValueError("Distribution fitting needs at least two different numerical values.")

    return data


def get_histogram_density(data, bins=30):
    """
    Creates histogram density values.
    These act as the observed distribution for least-squares fitting.
    """

    observed_density, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, observed_density


def calculate_errors(observed_density, fitted_density):
    """
    Calculates least-squares error values.
    """

    residuals = observed_density - fitted_density

    sse = np.sum(residuals ** 2)
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)

    return sse, mse, rmse


def calculate_ks_test(data, distribution_name, parameters):
    """
    Calculates Kolmogorov-Smirnov statistic and p-value for the fitted distribution.
    """

    data_array = np.asarray(data)

    try:
        if distribution_name == "Normal":
            ks_statistic, ks_p_value = stats.kstest(
                data_array,
                "norm",
                args=(parameters["mean"], parameters["std"])
            )

        elif distribution_name == "Exponential":
            ks_statistic, ks_p_value = stats.kstest(
                data_array,
                "expon",
                args=(parameters["loc"], parameters["scale"])
            )

        elif distribution_name == "Uniform":
            ks_statistic, ks_p_value = stats.kstest(
                data_array,
                "uniform",
                args=(parameters["loc"], parameters["scale"])
            )

        else:
            return np.nan, np.nan

        return ks_statistic, ks_p_value

    except Exception:
        return np.nan, np.nan


# ------------------------------------------------------------
# Least-squares fitting functions
# ------------------------------------------------------------

def fit_normal_least_squares(bin_centers, observed_density, data):
    """
    Fits normal distribution using least squares.
    Parameters: mean, standard deviation.
    """

    initial_mu = data.mean()
    initial_sigma = data.std(ddof=1)

    if initial_sigma <= 0:
        initial_sigma = 1.0

    def residual_function(params):
        mu, sigma = params

        if sigma <= 0:
            return np.ones_like(observed_density) * 999999

        fitted_density = stats.norm.pdf(bin_centers, loc=mu, scale=sigma)
        return observed_density - fitted_density

    result = least_squares(
        residual_function,
        x0=[initial_mu, initial_sigma],
        bounds=([-np.inf, 0.000001], [np.inf, np.inf])
    )

    mu, sigma = result.x

    fitted_density = stats.norm.pdf(bin_centers, loc=mu, scale=sigma)
    sse, mse, rmse = calculate_errors(observed_density, fitted_density)

    parameters = {
        "mean": mu,
        "std": sigma
    }

    ks_statistic, ks_p_value = calculate_ks_test(
        data,
        "Normal",
        parameters
    )

    return {
        "Distribution": "Normal",
        "Parameters": parameters,
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "KS Statistic": ks_statistic,
        "KS p-value": ks_p_value,
    }


def fit_exponential_least_squares(bin_centers, observed_density, data):
    """
    Fits shifted exponential distribution using least squares.
    Parameters: location, scale.
    """

    data_min = data.min()
    data_range = data.max() - data.min()

    if data_range <= 0:
        data_range = 1.0

    initial_loc = data_min
    initial_scale = max(data.mean() - data_min, 1.0)

    lower_loc = data_min - data_range
    upper_loc = data_min

    def residual_function(params):
        loc, scale = params

        if scale <= 0:
            return np.ones_like(observed_density) * 999999

        fitted_density = stats.expon.pdf(bin_centers, loc=loc, scale=scale)
        return observed_density - fitted_density

    result = least_squares(
        residual_function,
        x0=[initial_loc, initial_scale],
        bounds=([lower_loc, 0.000001], [upper_loc, np.inf])
    )

    loc, scale = result.x

    fitted_density = stats.expon.pdf(bin_centers, loc=loc, scale=scale)
    sse, mse, rmse = calculate_errors(observed_density, fitted_density)

    parameters = {
        "loc": loc,
        "scale": scale
    }

    ks_statistic, ks_p_value = calculate_ks_test(
        data,
        "Exponential",
        parameters
    )

    return {
        "Distribution": "Exponential",
        "Parameters": parameters,
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "KS Statistic": ks_statistic,
        "KS p-value": ks_p_value,
    }


def fit_uniform_least_squares(bin_centers, observed_density, data):
    """
    Fits uniform distribution using a simple least-squares grid search.
    Parameters: location, scale.
    """

    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min

    if data_range <= 0:
        data_range = 1.0

    best_result = None

    loc_candidates = np.linspace(data_min - 0.10 * data_range, data_min, 30)
    max_candidates = np.linspace(data_max, data_max + 0.10 * data_range, 30)

    for loc in loc_candidates:
        for upper in max_candidates:
            scale = upper - loc

            if scale <= 0:
                continue

            fitted_density = stats.uniform.pdf(
                bin_centers,
                loc=loc,
                scale=scale
            )

            sse, mse, rmse = calculate_errors(
                observed_density,
                fitted_density
            )

            if best_result is None or mse < best_result["MSE"]:
                parameters = {
                    "loc": loc,
                    "scale": scale,
                    "min": loc,
                    "max": loc + scale
                }

                best_result = {
                    "Distribution": "Uniform",
                    "Parameters": parameters,
                    "SSE": sse,
                    "MSE": mse,
                    "RMSE": rmse
                }

    if best_result is None:
        raise ValueError("Uniform distribution fitting failed.")

    ks_statistic, ks_p_value = calculate_ks_test(
        data,
        "Uniform",
        best_result["Parameters"]
    )

    best_result["KS Statistic"] = ks_statistic
    best_result["KS p-value"] = ks_p_value

    return best_result


def fit_all_distributions_least_squares(df, column, bins=30):
    """
    Fits Normal, Exponential, and Uniform distributions using least-squares comparison.
    """

    data = prepare_numeric_data(df, column)
    bin_centers, observed_density = get_histogram_density(data, bins=bins)

    results = [
        fit_normal_least_squares(bin_centers, observed_density, data),
        fit_exponential_least_squares(bin_centers, observed_density, data),
        fit_uniform_least_squares(bin_centers, observed_density, data),
    ]

    results = sorted(results, key=lambda x: x["MSE"])

    for rank, result in enumerate(results, start=1):
        result["Rank"] = rank

    return results, data, bin_centers, observed_density


# ------------------------------------------------------------
# Tables and PDF / PPF helpers
# ------------------------------------------------------------

def create_distribution_fit_table(results):
    """
    Converts distribution fitting results into a table.
    """

    rows = []

    for result in results:
        parameter_text = ", ".join([
            f"{key}={format_number(value)}"
            for key, value in result["Parameters"].items()
        ])

        rows.append({
            "Rank": result["Rank"],
            "Distribution": result["Distribution"],
            "Parameters": parameter_text,
            "SSE": format_number(result["SSE"]),
            "MSE": format_number(result["MSE"]),
            "RMSE": format_number(result["RMSE"]),
            "KS Statistic": format_number(result.get("KS Statistic", np.nan)),
            "KS p-value": format_p_value(result.get("KS p-value", np.nan)),
        })

    return pd.DataFrame(rows)


def get_pdf_values(distribution_name, x_values, parameters):
    """
    Returns PDF values for a fitted distribution.
    """

    if distribution_name == "Normal":
        return stats.norm.pdf(
            x_values,
            loc=parameters["mean"],
            scale=parameters["std"]
        )

    if distribution_name == "Exponential":
        return stats.expon.pdf(
            x_values,
            loc=parameters["loc"],
            scale=parameters["scale"]
        )

    if distribution_name == "Uniform":
        return stats.uniform.pdf(
            x_values,
            loc=parameters["loc"],
            scale=parameters["scale"]
        )

    raise ValueError("Unsupported distribution.")


def get_ppf_values(distribution_name, probabilities, parameters):
    """
    Returns theoretical quantiles for Q-Q plots.
    """

    if distribution_name == "Normal":
        return stats.norm.ppf(
            probabilities,
            loc=parameters["mean"],
            scale=parameters["std"]
        )

    if distribution_name == "Exponential":
        return stats.expon.ppf(
            probabilities,
            loc=parameters["loc"],
            scale=parameters["scale"]
        )

    if distribution_name == "Uniform":
        return stats.uniform.ppf(
            probabilities,
            loc=parameters["loc"],
            scale=parameters["scale"]
        )

    raise ValueError("Unsupported distribution.")


# ------------------------------------------------------------
# Plotly plots
# ------------------------------------------------------------

def plot_distribution_fits(data, results, bins=30):
    """
    Plotly histogram with all fitted PDF curves.
    """

    data_array = np.asarray(data)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data_array,
            nbinsx=bins,
            histnorm="probability density",
            name="Observed Histogram",
            marker={
                "color": PLOT_COLORS["primary"],
                "opacity": 0.55,
                "line": {
                    "color": PLOT_COLORS["grid"],
                    "width": 1
                }
            },
            hovertemplate="Value: %{x}<br>Density: %{y}<extra></extra>",
        )
    )

    x_values = np.linspace(data_array.min(), data_array.max(), 500)

    color_cycle = CHART_PALETTE

    for index, result in enumerate(results):
        y_values = get_pdf_values(
            result["Distribution"],
            x_values,
            result["Parameters"]
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=f"{result['Rank']}. {result['Distribution']} PDF",
                line={
                    "width": 3,
                    "color": color_cycle[index % len(color_cycle)]
                },
                hovertemplate=(
                    "Value: %{x:.4f}<br>"
                    "PDF Density: %{y:.6f}"
                    "<extra></extra>"
                ),
            )
        )

    fig = apply_plotly_theme(
        fig,
        title="Distribution Fitting using Least Squares",
        x_title="Value",
        y_title="Density",
        height=520,
    )

    return fig


def plot_single_distribution_fit(data, result, bins=30):
    """
    Plotly histogram with one selected fitted PDF.
    """

    data_array = np.asarray(data)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data_array,
            nbinsx=bins,
            histnorm="probability density",
            name="Observed Histogram",
            marker={
                "color": PLOT_COLORS["primary"],
                "opacity": 0.55,
                "line": {
                    "color": PLOT_COLORS["grid"],
                    "width": 1
                }
            },
            hovertemplate="Value: %{x}<br>Density: %{y}<extra></extra>",
        )
    )

    x_values = np.linspace(data_array.min(), data_array.max(), 500)

    y_values = get_pdf_values(
        result["Distribution"],
        x_values,
        result["Parameters"]
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=f"{result['Distribution']} PDF",
            line={
                "width": 3,
                "color": PLOT_COLORS["green"]
            },
            hovertemplate=(
                "Value: %{x:.4f}<br>"
                "PDF Density: %{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"{result['Distribution']} Fit",
        x_title="Value",
        y_title="Density",
        height=500,
    )

    return fig


def plot_distribution_qq(data, result):
    """
    Creates Plotly Q-Q plot for selected fitted distribution.
    """

    data_array = np.asarray(data)
    sorted_data = np.sort(data_array)

    n = len(sorted_data)

    probabilities = (np.arange(1, n + 1) - 0.5) / n

    distribution = result["Distribution"]
    parameters = result["Parameters"]

    theoretical_quantiles = get_ppf_values(
        distribution,
        probabilities,
        parameters
    )

    valid_mask = np.isfinite(theoretical_quantiles) & np.isfinite(sorted_data)

    theoretical_quantiles = theoretical_quantiles[valid_mask]
    sorted_data = sorted_data[valid_mask]

    if len(sorted_data) < 2:
        raise ValueError("Not enough valid values to create Q-Q plot.")

    combined_values = np.concatenate([theoretical_quantiles, sorted_data])

    line_min = np.min(combined_values)
    line_max = np.max(combined_values)

    correlation = np.corrcoef(theoretical_quantiles, sorted_data)[0, 1]

    if np.isfinite(correlation):
        r_squared = correlation ** 2
    else:
        r_squared = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode="markers",
            name="Observed Quantiles",
            marker={
                "size": 7,
                "color": PLOT_COLORS["primary"],
                "line": {
                    "color": "#FFFFFF",
                    "width": 0.7
                }
            },
            hovertemplate=(
                "Theoretical Quantile: %{x:.4f}<br>"
                "Observed Quantile: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="Reference Line",
            line={
                "color": PLOT_COLORS["green"],
                "width": 3,
                "dash": "dash"
            },
            hoverinfo="skip",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Q-Q Plot against {distribution}",
        x_title=f"Theoretical Quantiles ({distribution})",
        y_title="Observed Quantiles",
        height=520,
    )

    if np.isfinite(r_squared):
        fig.add_annotation(
            text=f"Q-Q R² = {r_squared:.4f}",
            xref="paper",
            yref="paper",
            x=0.03,
            y=0.97,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=PLOT_COLORS["grid"],
            borderwidth=1,
            font={"color": PLOT_COLORS["text"], "size": 13},
        )

    return fig


# ------------------------------------------------------------
# Interpretations
# ------------------------------------------------------------

def get_distribution_fit_interpretation(results, column):
    """
    Creates interpretation for distribution fitting results.
    """

    best = results[0]
    second = results[1] if len(results) > 1 else None

    interpretation = []

    interpretation.append(
        f"The best-fitting distribution for `{column}` is **{best['Distribution']}** based on the lowest MSE."
    )

    interpretation.append(
        f"{best['Distribution']} has MSE = {format_number(best['MSE'])}, "
        f"SSE = {format_number(best['SSE'])}, and RMSE = {format_number(best['RMSE'])}."
    )

    if not pd.isna(best.get("KS p-value", np.nan)):
        interpretation.append(
            f"The KS p-value for the best fit is {format_p_value(best['KS p-value'])}. "
            "A larger KS p-value usually means the fitted distribution is not strongly contradicted by the data."
        )

    if second is not None:
        interpretation.append(
            f"The second-best fit is {second['Distribution']} with MSE = {format_number(second['MSE'])}."
        )

    interpretation.append(
        "This comparison mainly uses least-squares error between the observed histogram density and each fitted PDF curve."
    )

    interpretation.append(
        "Lower MSE/RMSE means the fitted theoretical curve is closer to the observed data distribution."
    )

    interpretation.append(
        "The result can change slightly depending on the number of histogram bins, so visual inspection and the Q-Q plot are also important."
    )

    return interpretation


def get_selected_distribution_interpretation(result):
    """
    Gives explanation for selected distribution.
    """

    distribution = result["Distribution"]

    notes = []

    if distribution == "Normal":
        notes.extend([
            "The normal distribution is useful for data that is roughly symmetric and bell-shaped.",
            "If the Q-Q plot points follow the reference line closely, the normal fit is visually reasonable.",
            "Large tail deviations in the Q-Q plot suggest the data may not be truly normal."
        ])

    elif distribution == "Exponential":
        notes.extend([
            "The exponential distribution is useful for right-skewed data, especially waiting times or time-between-events.",
            "If the histogram has many low values and fewer high values, exponential may fit well.",
            "If the Q-Q plot curves away strongly, the exponential fit may not be suitable."
        ])

    elif distribution == "Uniform":
        notes.extend([
            "The uniform distribution assumes all values in a range are approximately equally likely.",
            "It fits best when the histogram is fairly flat across the range.",
            "If the data has clear peaks or skewness, uniform distribution is usually not suitable."
        ])

    else:
        notes.append("No interpretation available for this distribution.")

    notes.append(
        f"SSE = {format_number(result['SSE'])}, MSE = {format_number(result['MSE'])}, RMSE = {format_number(result['RMSE'])}."
    )

    if "KS Statistic" in result and "KS p-value" in result:
        notes.append(
            f"KS statistic = {format_number(result['KS Statistic'])}, KS p-value = {format_p_value(result['KS p-value'])}."
        )

    return notes