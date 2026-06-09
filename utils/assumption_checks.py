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


def format_number(value, decimals=5):
    """
    Formats numbers safely for tables and interpretations.
    """

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if abs(value) < 0.00001 and value != 0:
            return f"{value:.2e}"

        return f"{value:.{decimals}f}"

    except Exception:
        return str(value)


def prepare_numeric_series(df, column):
    """
    Converts selected column into numeric values and removes missing values.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    data = pd.to_numeric(df[column], errors="coerce").dropna()

    if data.empty:
        raise ValueError(f"Column '{column}' does not contain valid numerical data.")

    return data


def run_shapiro_normality(data, alpha=0.05):
    """
    Runs Shapiro-Wilk normality test safely.

    H0: data is normally distributed.
    H1: data is not normally distributed.
    """

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return {
            "statistic": None,
            "p_value": None,
            "status": "Not enough data",
            "status_type": "warning",
            "message": "Shapiro-Wilk requires at least 3 valid values."
        }

    test_data = data

    if len(data) > 5000:
        test_data = data.sample(n=5000, random_state=42)

    statistic, p_value = stats.shapiro(test_data)

    if p_value < alpha:
        return {
            "statistic": statistic,
            "p_value": p_value,
            "status": "Normality concern",
            "status_type": "warning",
            "message": f"Shapiro p-value = {format_number(p_value)}. The data may not be normally distributed."
        }

    return {
        "statistic": statistic,
        "p_value": p_value,
        "status": "Normality acceptable",
        "status_type": "success",
        "message": f"Shapiro p-value = {format_number(p_value)}. There is not enough evidence to say the data is non-normal."
    }


def check_iqr_outliers(data):
    """
    Detects possible outliers using the 1.5 × IQR rule.
    """

    data = pd.Series(data).dropna()

    if len(data) < 4:
        return {
            "outlier_count": 0,
            "outlier_percentage": 0,
            "status": "Not enough data",
            "status_type": "warning",
            "message": "At least 4 values are needed for a useful IQR outlier check."
        }

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = data[(data < lower_fence) | (data > upper_fence)]

    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(data)) * 100

    if outlier_count == 0:
        return {
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "status": "No clear outliers",
            "status_type": "success",
            "message": "No possible outliers were detected using the 1.5 × IQR rule."
        }

    if outlier_percentage <= 5:
        return {
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "status": "Few possible outliers",
            "status_type": "warning",
            "message": f"{outlier_count} possible outlier(s) detected, about {format_number(outlier_percentage, 2)}% of values."
        }

    return {
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
        "status": "Many possible outliers",
        "status_type": "warning",
        "message": f"{outlier_count} possible outlier(s) detected, about {format_number(outlier_percentage, 2)}% of values. This may affect mean-based tests."
    }


def check_sample_size(n, test_name):
    """
    Gives a simple sample size status.
    """

    if n < 2:
        return {
            "status": "Too small",
            "status_type": "error",
            "message": f"{test_name} requires at least 2 valid values."
        }

    if n < 15:
        return {
            "status": "Small sample",
            "status_type": "warning",
            "message": f"Sample size is {n}. Normality matters more with small samples."
        }

    if n < 30:
        return {
            "status": "Moderate sample",
            "status_type": "warning",
            "message": f"Sample size is {n}. Check normality and outliers carefully."
        }

    return {
        "status": "Sample size acceptable",
        "status_type": "success",
        "message": f"Sample size is {n}. Mean-based tests are usually more stable with larger samples."
    }


def create_check(name, status, status_type, message):
    """
    Creates one assumption check item.
    """

    return {
        "Check": name,
        "Status": status,
        "Status Type": status_type,
        "Message": message
    }


def create_assumption_table(assumption_result):
    """
    Converts assumption checks into a dataframe.
    """

    return pd.DataFrame(assumption_result["checks"])


# ------------------------------------------------------------
# One-sample t-test assumptions
# ------------------------------------------------------------

def check_one_sample_ttest_assumptions(df, numeric_column, alpha=0.05):
    """
    Checks assumptions for one-sample t-test.

    Main assumptions:
    - Numerical variable
    - Independent observations
    - Approximately normal data if sample is small
    - No extreme outliers
    """

    data = prepare_numeric_series(df, numeric_column)

    checks = []

    checks.append(
        create_check(
            "Data type",
            "Numerical",
            "success",
            f"`{numeric_column}` contains valid numerical values."
        )
    )

    sample_size_check = check_sample_size(len(data), "One-sample t-test")

    checks.append(
        create_check(
            "Sample size",
            sample_size_check["status"],
            sample_size_check["status_type"],
            sample_size_check["message"]
        )
    )

    normality = run_shapiro_normality(data, alpha=alpha)

    checks.append(
        create_check(
            "Normality",
            normality["status"],
            normality["status_type"],
            normality["message"]
        )
    )

    outliers = check_iqr_outliers(data)

    checks.append(
        create_check(
            "Outliers",
            outliers["status"],
            outliers["status_type"],
            outliers["message"]
        )
    )

    warning_count = sum(check["Status Type"] == "warning" for check in checks)
    error_count = sum(check["Status Type"] == "error" for check in checks)

    if error_count > 0:
        recommendation_title = "Do not run t-test yet"
        recommendation_type = "error"
        recommendation = "The selected data does not satisfy the minimum requirements for a one-sample t-test."

    elif normality["status_type"] == "warning" and len(data) < 30:
        recommendation_title = "Consider non-parametric alternative"
        recommendation_type = "warning"
        recommendation = (
            "Normality may be violated and the sample is small. "
            "Consider a one-sample Wilcoxon/sign-style alternative if your lecturer allows it."
        )

    elif warning_count > 0:
        recommendation_title = "Use t-test carefully"
        recommendation_type = "warning"
        recommendation = (
            "The one-sample t-test can be run, but check the warnings before interpreting the result."
        )

    else:
        recommendation_title = "One-sample t-test is appropriate"
        recommendation_type = "success"
        recommendation = "The main assumptions look acceptable for a one-sample t-test."

    return {
        "test": "One-sample t-test",
        "checks": checks,
        "recommendation_title": recommendation_title,
        "recommendation_type": recommendation_type,
        "recommendation": recommendation,
        "diagnostic_data": {
            "data": data,
            "normality": normality,
            "outliers": outliers,
        }
    }


# ------------------------------------------------------------
# Independent two-sample t-test assumptions
# ------------------------------------------------------------

def check_independent_ttest_assumptions(
    df,
    numeric_column,
    group_column,
    group1,
    group2,
    alpha=0.05
):
    """
    Checks assumptions for independent two-sample t-test.

    Main assumptions:
    - Numerical outcome variable
    - Two independent groups
    - Approximately normal outcome in each group
    - Equal variances for Student's t-test
    - If variances differ, use Welch's t-test
    """

    group1_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group1)][numeric_column],
        errors="coerce"
    ).dropna()

    group2_data = pd.to_numeric(
        df[df[group_column].astype(str) == str(group2)][numeric_column],
        errors="coerce"
    ).dropna()

    checks = []

    checks.append(
        create_check(
            "Data structure",
            "Two independent groups",
            "success",
            f"Comparing `{numeric_column}` between `{group1}` and `{group2}` using `{group_column}`."
        )
    )

    if len(group1_data) < 2 or len(group2_data) < 2:
        checks.append(
            create_check(
                "Group sample sizes",
                "Too small",
                "error",
                "Each group must have at least 2 valid numerical values."
            )
        )

        return {
            "test": "Independent two-sample t-test",
            "checks": checks,
            "recommendation_title": "Do not run t-test yet",
            "recommendation_type": "error",
            "recommendation": "One or both groups do not have enough valid numerical values.",
            "diagnostic_data": {
                "group1_data": group1_data,
                "group2_data": group2_data,
            }
        }

    group1_size = check_sample_size(len(group1_data), f"Group {group1}")
    group2_size = check_sample_size(len(group2_data), f"Group {group2}")

    checks.append(
        create_check(
            f"Sample size: {group1}",
            group1_size["status"],
            group1_size["status_type"],
            group1_size["message"]
        )
    )

    checks.append(
        create_check(
            f"Sample size: {group2}",
            group2_size["status"],
            group2_size["status_type"],
            group2_size["message"]
        )
    )

    normality_group1 = run_shapiro_normality(group1_data, alpha=alpha)
    normality_group2 = run_shapiro_normality(group2_data, alpha=alpha)

    checks.append(
        create_check(
            f"Normality: {group1}",
            normality_group1["status"],
            normality_group1["status_type"],
            normality_group1["message"]
        )
    )

    checks.append(
        create_check(
            f"Normality: {group2}",
            normality_group2["status"],
            normality_group2["status_type"],
            normality_group2["message"]
        )
    )

    outliers_group1 = check_iqr_outliers(group1_data)
    outliers_group2 = check_iqr_outliers(group2_data)

    checks.append(
        create_check(
            f"Outliers: {group1}",
            outliers_group1["status"],
            outliers_group1["status_type"],
            outliers_group1["message"]
        )
    )

    checks.append(
        create_check(
            f"Outliers: {group2}",
            outliers_group2["status"],
            outliers_group2["status_type"],
            outliers_group2["message"]
        )
    )

    levene_statistic, levene_p_value = stats.levene(group1_data, group2_data)

    if levene_p_value < alpha:
        variance_status = "Unequal variances"
        variance_type = "warning"
        variance_message = (
            f"Levene p-value = {format_number(levene_p_value)}. "
            "Equal variance assumption may be violated. Welch's t-test is preferred."
        )
    else:
        variance_status = "Equal variance acceptable"
        variance_type = "success"
        variance_message = (
            f"Levene p-value = {format_number(levene_p_value)}. "
            "There is not enough evidence to say the variances are unequal."
        )

    checks.append(
        create_check(
            "Equal variance",
            variance_status,
            variance_type,
            variance_message
        )
    )

    normality_warning = (
        normality_group1["status_type"] == "warning"
        or normality_group2["status_type"] == "warning"
    )

    small_group = len(group1_data) < 30 or len(group2_data) < 30

    if normality_warning and small_group:
        recommendation_title = "Consider Mann-Whitney U test"
        recommendation_type = "warning"
        recommendation = (
            "At least one group may not be normally distributed and sample size is not large. "
            "Consider Mann-Whitney U as a non-parametric alternative."
        )

    elif levene_p_value < alpha:
        recommendation_title = "Use Welch's t-test"
        recommendation_type = "warning"
        recommendation = (
            "Normality may be acceptable, but Levene's test suggests unequal variances. "
            "Welch's t-test is recommended instead of Student's equal-variance t-test."
        )

    elif normality_warning:
        recommendation_title = "Use t-test carefully"
        recommendation_type = "warning"
        recommendation = (
            "Normality may be violated. If sample sizes are large, the t-test may still be reasonably robust, "
            "but Mann-Whitney U can be considered."
        )

    else:
        recommendation_title = "Independent t-test is appropriate"
        recommendation_type = "success"
        recommendation = (
            "The main assumptions look acceptable. Student's t-test is suitable if equal variance is assumed."
        )

    return {
        "test": "Independent two-sample t-test",
        "checks": checks,
        "recommendation_title": recommendation_title,
        "recommendation_type": recommendation_type,
        "recommendation": recommendation,
        "diagnostic_data": {
            "group1_data": group1_data,
            "group2_data": group2_data,
            "group1": group1,
            "group2": group2,
            "numeric_column": numeric_column,
            "group_column": group_column,
            "levene_statistic": levene_statistic,
            "levene_p_value": levene_p_value,
            "normality_group1": normality_group1,
            "normality_group2": normality_group2,
        }
    }


# ------------------------------------------------------------
# Paired t-test assumptions
# ------------------------------------------------------------

def check_paired_ttest_assumptions(df, before_column, after_column, alpha=0.05):
    """
    Checks assumptions for paired t-test.

    Main assumptions:
    - Two related numerical measurements
    - Complete pairs
    - Differences are approximately normally distributed
    - No extreme outliers in differences
    """

    paired_df = df[[before_column, after_column]].copy()

    paired_df[before_column] = pd.to_numeric(paired_df[before_column], errors="coerce")
    paired_df[after_column] = pd.to_numeric(paired_df[after_column], errors="coerce")

    paired_df = paired_df.dropna()

    checks = []

    if len(paired_df) < 2:
        checks.append(
            create_check(
                "Complete pairs",
                "Too small",
                "error",
                "Paired t-test requires at least 2 complete before-after pairs."
            )
        )

        return {
            "test": "Paired t-test",
            "checks": checks,
            "recommendation_title": "Do not run paired t-test yet",
            "recommendation_type": "error",
            "recommendation": "There are not enough complete pairs.",
            "diagnostic_data": {
                "paired_df": paired_df,
            }
        }

    before_data = paired_df[before_column]
    after_data = paired_df[after_column]
    differences = after_data - before_data

    checks.append(
        create_check(
            "Pairing",
            "Complete pairs detected",
            "success",
            f"{len(paired_df)} complete pair(s) found between `{before_column}` and `{after_column}`."
        )
    )

    pair_size_check = check_sample_size(len(paired_df), "Paired t-test")

    checks.append(
        create_check(
            "Number of pairs",
            pair_size_check["status"],
            pair_size_check["status_type"],
            pair_size_check["message"]
        )
    )

    normality_diff = run_shapiro_normality(differences, alpha=alpha)

    checks.append(
        create_check(
            "Normality of differences",
            normality_diff["status"],
            normality_diff["status_type"],
            normality_diff["message"]
        )
    )

    outliers_diff = check_iqr_outliers(differences)

    checks.append(
        create_check(
            "Outliers in differences",
            outliers_diff["status"],
            outliers_diff["status_type"],
            outliers_diff["message"]
        )
    )

    if normality_diff["status_type"] == "warning" and len(paired_df) < 30:
        recommendation_title = "Consider Wilcoxon signed-rank test"
        recommendation_type = "warning"
        recommendation = (
            "The differences may not be normally distributed and the sample is small. "
            "Consider Wilcoxon signed-rank test as a non-parametric alternative."
        )

    elif normality_diff["status_type"] == "warning":
        recommendation_title = "Use paired t-test carefully"
        recommendation_type = "warning"
        recommendation = (
            "The differences may not be normally distributed. If the sample size is large, "
            "the paired t-test may still be usable, but Wilcoxon signed-rank can be considered."
        )

    else:
        recommendation_title = "Paired t-test is appropriate"
        recommendation_type = "success"
        recommendation = (
            "The differences look acceptable for a paired t-test."
        )

    return {
        "test": "Paired t-test",
        "checks": checks,
        "recommendation_title": recommendation_title,
        "recommendation_type": recommendation_type,
        "recommendation": recommendation,
        "diagnostic_data": {
            "paired_df": paired_df,
            "before_data": before_data,
            "after_data": after_data,
            "differences": differences,
            "normality_diff": normality_diff,
            "outliers_diff": outliers_diff,
        }
    }


# ------------------------------------------------------------
# Plotly diagnostic plots
# ------------------------------------------------------------

def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=380):
    """
    Applies a dark Plotly dashboard theme.
    """

    fig.update_layout(
        title={
            "text": title if title else "",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 17, "color": PLOT_COLORS["text"]},
        },
        paper_bgcolor=PLOT_COLORS["bg"],
        plot_bgcolor=PLOT_COLORS["card"],
        font={"color": PLOT_COLORS["text"], "family": "Arial"},
        height=height,
        margin={"l": 45, "r": 25, "t": 55, "b": 45},
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


def plot_assumption_histogram(data, title, x_label, bins=25):
    """
    Plotly histogram for assumption checking.
    """

    data = pd.Series(data).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bins,
            marker={
                "color": PLOT_COLORS["secondary"],
                "line": {"color": PLOT_COLORS["bg"], "width": 1},
            },
            opacity=0.85,
            name="Values",
            hovertemplate=f"{x_label}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    fig.add_vline(
        x=data.mean(),
        line_width=2,
        line_dash="dash",
        line_color=PLOT_COLORS["green"],
        annotation_text="Mean",
        annotation_position="top left",
    )

    fig.add_vline(
        x=data.median(),
        line_width=2,
        line_dash="dot",
        line_color=PLOT_COLORS["warning"],
        annotation_text="Median",
        annotation_position="top right",
    )

    fig = apply_plotly_theme(
        fig,
        title=title,
        x_title=x_label,
        y_title="Frequency",
        height=360,
    )

    return fig


def plot_assumption_qq(data, title):
    """
    Plotly Q-Q plot against normal distribution.
    """

    data = pd.Series(data).dropna()

    osm, osr = stats.probplot(data, dist="norm", fit=False)
    fit_result = stats.probplot(data, dist="norm", fit=True)

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
                "size": 6,
                "color": PLOT_COLORS["primary"],
                "opacity": 0.85,
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
                "color": PLOT_COLORS["green"],
                "width": 2,
                "dash": "dash",
            },
            name=f"Reference line | R²={r_value ** 2:.4f}",
            hovertemplate="Reference line<extra></extra>",
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=title,
        x_title="Theoretical Quantiles",
        y_title="Ordered Values",
        height=380,
    )

    return fig


def plot_independent_groups_boxplot(group1_data, group2_data, group1, group2, numeric_column):
    """
    Boxplot for two independent groups.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=group1_data,
            name=str(group1),
            marker_color=PLOT_COLORS["primary"],
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Box(
            y=group2_data,
            name=str(group2),
            marker_color=PLOT_COLORS["secondary"],
            boxmean=True,
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"{numeric_column} by group",
        x_title="Group",
        y_title=numeric_column,
        height=380,
    )

    return fig


def plot_paired_differences(differences):
    """
    Histogram of paired differences.
    """

    return plot_assumption_histogram(
        differences,
        title="Distribution of Paired Differences",
        x_label="After - Before",
        bins=25
    )



# ------------------------------------------------------------
# ANOVA assumption checks
# ------------------------------------------------------------

from statsmodels.formula.api import ols


def _clean_anova_data(df, numeric_column, factor_columns):
    """
    Prepares data for ANOVA assumption checks.
    """

    required_columns = [numeric_column] + factor_columns

    clean_df = df[required_columns].copy()
    clean_df[numeric_column] = pd.to_numeric(clean_df[numeric_column], errors="coerce")

    for factor in factor_columns:
        clean_df[factor] = clean_df[factor].astype(str)

    clean_df = clean_df.dropna()

    if clean_df.empty:
        raise ValueError("No valid data available after removing missing values.")

    if clean_df[numeric_column].nunique() < 2:
        raise ValueError("The numerical outcome must contain at least two different values.")

    for factor in factor_columns:
        if clean_df[factor].nunique() < 2:
            raise ValueError(f"The factor column `{factor}` must contain at least two groups.")

    return clean_df


def check_one_way_anova_assumptions(df, numeric_column, factor_column, alpha=0.05):
    """
    Checks assumptions for one-way ANOVA.

    Main assumptions:
    - Numerical outcome variable
    - One categorical factor
    - Independent groups
    - Approximately normal residuals / normality within groups
    - Homogeneity of variances using Levene's test
    - Reasonable group sizes
    """

    clean_df = _clean_anova_data(df, numeric_column, [factor_column])

    checks = []

    checks.append(
        create_check(
            "Data structure",
            "Numerical outcome + categorical factor",
            "success",
            f"`{numeric_column}` is compared across groups of `{factor_column}`."
        )
    )

    group_counts = clean_df[factor_column].value_counts()
    group_count = len(group_counts)
    smallest_group_size = int(group_counts.min())

    if group_count < 2:
        checks.append(
            create_check(
                "Number of groups",
                "Too few groups",
                "error",
                "One-way ANOVA requires at least two groups. Usually it is used for three or more groups."
            )
        )
    elif group_count == 2:
        checks.append(
            create_check(
                "Number of groups",
                "Two groups detected",
                "warning",
                "ANOVA can run with two groups, but an independent t-test is usually simpler."
            )
        )
    else:
        checks.append(
            create_check(
                "Number of groups",
                "Group count acceptable",
                "success",
                f"{group_count} groups detected."
            )
        )

    if smallest_group_size < 2:
        checks.append(
            create_check(
                "Group sizes",
                "Too small",
                "error",
                f"The smallest group has only {smallest_group_size} value(s)."
            )
        )
    elif smallest_group_size < 5:
        checks.append(
            create_check(
                "Group sizes",
                "Small group warning",
                "warning",
                f"The smallest group has {smallest_group_size} values. ANOVA is less stable with very small groups."
            )
        )
    else:
        checks.append(
            create_check(
                "Group sizes",
                "Group sizes acceptable",
                "success",
                f"The smallest group has {smallest_group_size} values."
            )
        )

    # Normality by group
    normality_warnings = 0

    for group_name, group_data in clean_df.groupby(factor_column):
        values = group_data[numeric_column].dropna()
        normality = run_shapiro_normality(values, alpha=alpha)

        if normality["status_type"] == "warning":
            normality_warnings += 1

        checks.append(
            create_check(
                f"Normality: {group_name}",
                normality["status"],
                normality["status_type"],
                normality["message"]
            )
        )

    # Levene's test
    groups = [
        group_data[numeric_column].dropna().values
        for _, group_data in clean_df.groupby(factor_column)
        if len(group_data[numeric_column].dropna()) >= 2
    ]

    if len(groups) >= 2:
        levene_statistic, levene_p_value = stats.levene(*groups)

        if levene_p_value < alpha:
            checks.append(
                create_check(
                    "Equal variance",
                    "Variance concern",
                    "warning",
                    f"Levene p-value = {format_number(levene_p_value)}. Equal variance assumption may be violated."
                )
            )
        else:
            checks.append(
                create_check(
                    "Equal variance",
                    "Equal variance acceptable",
                    "success",
                    f"Levene p-value = {format_number(levene_p_value)}. There is not enough evidence to say variances are unequal."
                )
            )
    else:
        levene_statistic, levene_p_value = np.nan, np.nan

        checks.append(
            create_check(
                "Equal variance",
                "Could not test",
                "warning",
                "Levene's test could not be calculated because too few groups had enough values."
            )
        )

    # Residual normality
    formula = f'Q("{numeric_column}") ~ C(Q("{factor_column}"))'
    model = ols(formula, data=clean_df).fit()
    residuals = model.resid

    residual_normality = run_shapiro_normality(residuals, alpha=alpha)

    checks.append(
        create_check(
            "Residual normality",
            residual_normality["status"],
            residual_normality["status_type"],
            residual_normality["message"]
        )
    )

    warning_count = sum(check["Status Type"] == "warning" for check in checks)
    error_count = sum(check["Status Type"] == "error" for check in checks)

    if error_count > 0:
        recommendation_title = "Do not run ANOVA yet"
        recommendation_type = "error"
        recommendation = "The selected columns do not satisfy the minimum requirements for one-way ANOVA."

    elif normality_warnings > 0 or residual_normality["status_type"] == "warning":
        recommendation_title = "Consider Kruskal-Wallis test"
        recommendation_type = "warning"
        recommendation = (
            "Normality may be weak in one or more groups or in the residuals. "
            "Consider Kruskal-Wallis as a non-parametric alternative."
        )

    elif not pd.isna(levene_p_value) and levene_p_value < alpha:
        recommendation_title = "Use ANOVA carefully"
        recommendation_type = "warning"
        recommendation = (
            "Levene's test suggests unequal variances. ANOVA can be sensitive to this, especially with unequal group sizes. "
            "Consider Kruskal-Wallis or a robust alternative."
        )

    elif warning_count > 0:
        recommendation_title = "ANOVA is usable with caution"
        recommendation_type = "warning"
        recommendation = "One-way ANOVA can be run, but check the warnings before interpreting the result."

    else:
        recommendation_title = "One-way ANOVA is appropriate"
        recommendation_type = "success"
        recommendation = "The main assumptions look acceptable for one-way ANOVA."

    return {
        "test": "One-way ANOVA",
        "checks": checks,
        "recommendation_title": recommendation_title,
        "recommendation_type": recommendation_type,
        "recommendation": recommendation,
        "diagnostic_data": {
            "clean_df": clean_df,
            "numeric_column": numeric_column,
            "factor_column": factor_column,
            "residuals": residuals,
            "levene_statistic": levene_statistic,
            "levene_p_value": levene_p_value,
            "group_counts": group_counts,
        }
    }


def check_two_way_anova_assumptions(df, numeric_column, factor1, factor2, alpha=0.05):
    """
    Checks assumptions for two-way ANOVA.

    Main assumptions:
    - Numerical outcome variable
    - Two categorical factors
    - Reasonable cell sizes
    - Homogeneity of variances across factor combinations
    - Residual normality
    """

    clean_df = _clean_anova_data(df, numeric_column, [factor1, factor2])

    checks = []

    checks.append(
        create_check(
            "Data structure",
            "Numerical outcome + two categorical factors",
            "success",
            f"`{numeric_column}` is explained using `{factor1}`, `{factor2}`, and their interaction."
        )
    )

    factor1_groups = clean_df[factor1].nunique()
    factor2_groups = clean_df[factor2].nunique()

    checks.append(
        create_check(
            "Factor levels",
            "Factor levels detected",
            "success",
            f"`{factor1}` has {factor1_groups} levels and `{factor2}` has {factor2_groups} levels."
        )
    )

    clean_df["_anova_cell"] = clean_df[factor1].astype(str) + " | " + clean_df[factor2].astype(str)

    cell_counts = clean_df["_anova_cell"].value_counts()
    smallest_cell_size = int(cell_counts.min())
    empty_or_small_cells = int((cell_counts < 2).sum())

    if smallest_cell_size < 2:
        checks.append(
            create_check(
                "Cell sizes",
                "Very small cells",
                "warning",
                f"At least one factor combination has only {smallest_cell_size} value(s). Two-way ANOVA may be unstable."
            )
        )
    elif smallest_cell_size < 5:
        checks.append(
            create_check(
                "Cell sizes",
                "Small cell warning",
                "warning",
                f"The smallest factor combination has {smallest_cell_size} values. Interpret interaction effects carefully."
            )
        )
    else:
        checks.append(
            create_check(
                "Cell sizes",
                "Cell sizes acceptable",
                "success",
                f"The smallest factor combination has {smallest_cell_size} values."
            )
        )

    # Levene across factor combinations
    groups = [
        group_data[numeric_column].dropna().values
        for _, group_data in clean_df.groupby("_anova_cell")
        if len(group_data[numeric_column].dropna()) >= 2
    ]

    if len(groups) >= 2:
        levene_statistic, levene_p_value = stats.levene(*groups)

        if levene_p_value < alpha:
            checks.append(
                create_check(
                    "Equal variance across cells",
                    "Variance concern",
                    "warning",
                    f"Levene p-value = {format_number(levene_p_value)}. Variance may differ across factor combinations."
                )
            )
        else:
            checks.append(
                create_check(
                    "Equal variance across cells",
                    "Equal variance acceptable",
                    "success",
                    f"Levene p-value = {format_number(levene_p_value)}. No strong evidence of unequal variances across cells."
                )
            )
    else:
        levene_statistic, levene_p_value = np.nan, np.nan

        checks.append(
            create_check(
                "Equal variance across cells",
                "Could not test",
                "warning",
                "Levene's test could not be calculated because too few factor combinations had enough values."
            )
        )

    # Residual normality
    formula = f'Q("{numeric_column}") ~ C(Q("{factor1}")) * C(Q("{factor2}"))'
    model = ols(formula, data=clean_df).fit()
    residuals = model.resid

    residual_normality = run_shapiro_normality(residuals, alpha=alpha)

    checks.append(
        create_check(
            "Residual normality",
            residual_normality["status"],
            residual_normality["status_type"],
            residual_normality["message"]
        )
    )

    warning_count = sum(check["Status Type"] == "warning" for check in checks)
    error_count = sum(check["Status Type"] == "error" for check in checks)

    if error_count > 0:
        recommendation_title = "Do not run two-way ANOVA yet"
        recommendation_type = "error"
        recommendation = "The selected columns do not satisfy the minimum requirements for two-way ANOVA."

    elif residual_normality["status_type"] == "warning" or warning_count > 0:
        recommendation_title = "Use two-way ANOVA carefully"
        recommendation_type = "warning"
        recommendation = (
            "One or more assumptions may be weak. Two-way ANOVA has no simple direct replacement like Kruskal-Wallis. "
            "Consider transformation, robust methods, or interpret results cautiously."
        )

    else:
        recommendation_title = "Two-way ANOVA is appropriate"
        recommendation_type = "success"
        recommendation = "The main assumptions look acceptable for two-way ANOVA."

    return {
        "test": "Two-way ANOVA",
        "checks": checks,
        "recommendation_title": recommendation_title,
        "recommendation_type": recommendation_type,
        "recommendation": recommendation,
        "diagnostic_data": {
            "clean_df": clean_df,
            "numeric_column": numeric_column,
            "factor1": factor1,
            "factor2": factor2,
            "cell_column": "_anova_cell",
            "cell_counts": cell_counts,
            "residuals": residuals,
            "levene_statistic": levene_statistic,
            "levene_p_value": levene_p_value,
        }
    }


def plot_anova_group_boxplot(clean_df, numeric_column, factor_column):
    """
    Plotly boxplot for one-way ANOVA groups.
    """

    fig = go.Figure()

    for group_name, group_data in clean_df.groupby(factor_column):
        fig.add_trace(
            go.Box(
                y=group_data[numeric_column],
                name=str(group_name),
                boxmean=True,
            )
        )

    fig = apply_plotly_theme(
        fig,
        title=f"{numeric_column} by {factor_column}",
        x_title=factor_column,
        y_title=numeric_column,
        height=420,
    )

    return fig


def plot_anova_cell_boxplot(clean_df, numeric_column, cell_column):
    """
    Plotly boxplot for two-way ANOVA factor combinations.
    """

    fig = go.Figure()

    for cell_name, cell_data in clean_df.groupby(cell_column):
        fig.add_trace(
            go.Box(
                y=cell_data[numeric_column],
                name=str(cell_name),
                boxmean=True,
            )
        )

    fig = apply_plotly_theme(
        fig,
        title=f"{numeric_column} by factor combinations",
        x_title="Factor combination",
        y_title=numeric_column,
        height=460,
    )

    fig.update_xaxes(tickangle=25)

    return fig


def plot_anova_residuals_qq(residuals, title="Q-Q Plot of ANOVA Residuals"):
    """
    Q-Q plot for ANOVA residuals.
    """

    return plot_assumption_qq(
        residuals,
        title=title
    )


def plot_anova_residual_histogram(residuals):
    """
    Histogram of ANOVA residuals.
    """

    return plot_assumption_histogram(
        residuals,
        title="Distribution of ANOVA Residuals",
        x_label="Residuals",
        bins=25
    )