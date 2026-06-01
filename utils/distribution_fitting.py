import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import least_squares


def format_number(value):
    """
    Formats numbers nicely for tables.
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

    return {
        "Distribution": "Normal",
        "Parameters": {
            "mean": mu,
            "std": sigma
        },
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse
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

    return {
        "Distribution": "Exponential",
        "Parameters": {
            "loc": loc,
            "scale": scale
        },
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse
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

            fitted_density = stats.uniform.pdf(bin_centers, loc=loc, scale=scale)
            sse, mse, rmse = calculate_errors(observed_density, fitted_density)

            if best_result is None or mse < best_result["MSE"]:
                best_result = {
                    "Distribution": "Uniform",
                    "Parameters": {
                        "loc": loc,
                        "scale": scale,
                        "min": loc,
                        "max": loc + scale
                    },
                    "SSE": sse,
                    "MSE": mse,
                    "RMSE": rmse
                }

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
        fit_uniform_least_squares(bin_centers, observed_density, data)
    ]

    results = sorted(results, key=lambda x: x["MSE"])

    for rank, result in enumerate(results, start=1):
        result["Rank"] = rank

    return results, data, bin_centers, observed_density


# ------------------------------------------------------------
# Tables and plots
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
            "RMSE": format_number(result["RMSE"])
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


def plot_distribution_fits(data, results, bins=30):
    """
    Plots histogram with fitted PDF curves.
    """

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    ax.hist(
        data,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.55,
        label="Observed Histogram"
    )

    x_values = np.linspace(data.min(), data.max(), 400)

    for result in results:
        y_values = get_pdf_values(
            result["Distribution"],
            x_values,
            result["Parameters"]
        )

        ax.plot(
            x_values,
            y_values,
            linewidth=2,
            label=f"{result['Distribution']} PDF"
        )

    ax.set_title("Distribution Fitting using Least Squares", fontsize=12)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig


def plot_single_distribution_fit(data, result, bins=30):
    """
    Plots histogram with one selected fitted PDF.
    """

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    ax.hist(
        data,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.55,
        label="Observed Histogram"
    )

    x_values = np.linspace(data.min(), data.max(), 400)

    y_values = get_pdf_values(
        result["Distribution"],
        x_values,
        result["Parameters"]
    )

    ax.plot(
        x_values,
        y_values,
        linewidth=2,
        label=f"{result['Distribution']} PDF"
    )

    ax.set_title(f"{result['Distribution']} Fit", fontsize=12)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return fig


def plot_distribution_qq(data, result):
    """
    Creates Q-Q plot for selected fitted distribution.
    """

    fig, ax = plt.subplots(figsize=(5, 5))

    distribution = result["Distribution"]
    params = result["Parameters"]

    if distribution == "Normal":
        stats.probplot(
            data,
            dist=stats.norm,
            sparams=(params["mean"], params["std"]),
            plot=ax
        )

    elif distribution == "Exponential":
        stats.probplot(
            data,
            dist=stats.expon,
            sparams=(params["loc"], params["scale"]),
            plot=ax
        )

    elif distribution == "Uniform":
        stats.probplot(
            data,
            dist=stats.uniform,
            sparams=(params["loc"], params["scale"]),
            plot=ax
        )

    ax.set_title(f"Q-Q Plot against {distribution}", fontsize=12)
    ax.grid(True, alpha=0.3)

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
        f"{best['Distribution']} has MSE = {format_number(best['MSE'])} and SSE = {format_number(best['SSE'])}."
    )

    if second is not None:
        interpretation.append(
            f"The second-best fit is {second['Distribution']} with MSE = {format_number(second['MSE'])}."
        )

    interpretation.append(
        "This comparison uses least-squares error between the observed histogram density and each fitted PDF curve."
    )

    interpretation.append(
        "Lower MSE means the fitted theoretical curve is closer to the observed data distribution."
    )

    interpretation.append(
        "The result can change slightly depending on the number of histogram bins, so visual inspection is also important."
    )

    return interpretation


def get_selected_distribution_interpretation(result):
    """
    Gives explanation for selected distribution.
    """

    distribution = result["Distribution"]

    if distribution == "Normal":
        return [
            "The normal distribution is useful for data that is roughly symmetric and bell-shaped.",
            "If the Q-Q plot points follow the line closely, the normal fit is visually reasonable.",
            "Large tail deviations in the Q-Q plot suggest the data may not be truly normal."
        ]

    if distribution == "Exponential":
        return [
            "The exponential distribution is useful for right-skewed data, especially waiting times or time-between-events.",
            "If the histogram has many low values and fewer high values, exponential may fit well.",
            "If the Q-Q plot curves away strongly, the exponential fit may not be suitable."
        ]

    if distribution == "Uniform":
        return [
            "The uniform distribution assumes all values in a range are approximately equally likely.",
            "It fits best when the histogram is fairly flat across the range.",
            "If the data has clear peaks or skewness, uniform distribution is usually not suitable."
        ]

    return ["No interpretation available."]