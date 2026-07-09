import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

import plotly.graph_objects as go


# ------------------------------------------------------------
# Plot theme
# ------------------------------------------------------------

PLOT_COLORS = {
    "blue": "#56B4E9",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#7C5CFF",

    "primary": "#56B4E9",
    "secondary": "#D55E00",
    "success": "#009E73",
    "warning": "#E69F00",
    "error": "#D55E00",

    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#64748B",
}


COLOR_SEQUENCE = [
    PLOT_COLORS["blue"],
    PLOT_COLORS["vermillion"],
    PLOT_COLORS["green"],
    PLOT_COLORS["orange"],
    PLOT_COLORS["purple"],
]


def hex_to_rgba(hex_color, opacity=0.18):
    """
    Converts a six-digit hexadecimal colour into an RGBA string.

    Plotly may reject eight-digit hexadecimal colours such as
    #56B4E933. Therefore, transparency is added through rgba().
    """

    clean_hex = hex_color.lstrip("#")

    if len(clean_hex) != 6:
        raise ValueError(
            "hex_color must be a valid six-digit hexadecimal colour."
        )

    red = int(clean_hex[0:2], 16)
    green = int(clean_hex[2:4], 16)
    blue = int(clean_hex[4:6], 16)

    return f"rgba({red}, {green}, {blue}, {opacity})"


def apply_plotly_theme(
    fig,
    title=None,
    x_title=None,
    y_title=None,
    height=460
):
    """
    Applies the soft professional Plotly theme used by the app.
    """

    fig.update_layout(
        title={
            "text": title or "",
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
            "l": 55,
            "r": 30,
            "t": 70,
            "b": 55,
        },

        hovermode="closest",
        colorway=COLOR_SEQUENCE,

        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {
                "color": PLOT_COLORS["muted"],
            },
            "bgcolor": "rgba(255, 255, 255, 0)",
        },
    )

    fig.update_xaxes(
        title_text=x_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["grid"],
        linecolor=PLOT_COLORS["axis"],
        tickfont={
            "color": PLOT_COLORS["muted"],
        },
        title_font={
            "color": PLOT_COLORS["muted"],
        },
        showline=True,
        linewidth=1,
    )

    fig.update_yaxes(
        title_text=y_title,
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor=PLOT_COLORS["grid"],
        linecolor=PLOT_COLORS["axis"],
        tickfont={
            "color": PLOT_COLORS["muted"],
        },
        title_font={
            "color": PLOT_COLORS["muted"],
        },
        showline=True,
        linewidth=1,
    )

    return fig


# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------

def format_number(value):
    """
    Formats numerical values for result tables and interpretations.
    """

    try:
        numeric_value = float(value)

        if np.isnan(numeric_value):
            return "N/A"

        if np.isinf(numeric_value):
            return "N/A"

        if numeric_value == 0:
            return 0.0

        if abs(numeric_value) < 0.00001:
            return f"{numeric_value:.2e}"

        return round(numeric_value, 5)

    except (TypeError, ValueError):
        return value


def interpret_p_value(p_value, alpha=0.05):
    """
    Interprets a p-value using the selected significance level.
    """

    if pd.isna(p_value):
        return (
            "Not available",
            "The statistical decision could not be calculated."
        )

    if p_value < alpha:
        return (
            "Reject H0",
            "There is a statistically significant difference."
        )

    return (
        "Fail to Reject H0",
        "There is not enough evidence for a statistically significant difference."
    )


def prepare_anova_data(df, numeric_column, factor_columns):
    """
    Prepares data for ANOVA.

    The outcome is converted to numeric, missing rows are removed,
    and factor columns are converted to strings after missing-value
    removal.
    """

    required_columns = [numeric_column] + list(factor_columns)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required column(s) are missing: "
            + ", ".join(missing_columns)
        )

    anova_df = df[required_columns].copy()

    anova_df[numeric_column] = pd.to_numeric(
        anova_df[numeric_column],
        errors="coerce"
    )

    # Remove missing data before converting categories to strings.
    # Otherwise, NaN could become the category text "nan".
    anova_df = anova_df.dropna(subset=required_columns)

    for column in factor_columns:
        anova_df[column] = anova_df[column].astype(str)

    if anova_df.empty:
        raise ValueError(
            "No valid data is available after removing missing values."
        )

    if anova_df[numeric_column].nunique() < 2:
        raise ValueError(
            "The numerical outcome must contain at least two different values."
        )

    for column in factor_columns:
        if anova_df[column].nunique() < 2:
            raise ValueError(
                f"The factor column '{column}' must contain at least two groups."
            )

    return anova_df


def create_anova_table(model, typ=2):
    """
    Creates a clean ANOVA table from a fitted statsmodels model.
    """

    raw_table = sm.stats.anova_lm(
        model,
        typ=typ
    )

    table = (
        raw_table
        .reset_index()
        .rename(columns={"index": "Source"})
    )

    rows = []

    for _, row in table.iterrows():

        rows.append({
            "Source": row.get("Source"),
            "Sum of Squares": format_number(
                row.get("sum_sq")
            ),
            "df": format_number(
                row.get("df")
            ),
            "F-statistic": format_number(
                row.get("F")
            ),
            "p-value": format_number(
                row.get("PR(>F)")
            ),
        })

    return pd.DataFrame(rows)


def _run_levene_test(groups):
    """
    Runs Levene's test when at least two usable groups exist.
    """

    usable_groups = [
        np.asarray(group, dtype=float)
        for group in groups
        if len(group) >= 2
    ]

    if len(usable_groups) < 2:
        return np.nan, np.nan

    try:
        statistic, p_value = stats.levene(
            *usable_groups
        )

        return float(statistic), float(p_value)

    except Exception:
        return np.nan, np.nan


# ------------------------------------------------------------
# One-way ANOVA
# ------------------------------------------------------------

def run_one_way_anova(
    df,
    numeric_column,
    factor_column,
    alpha=0.05
):
    """
    Runs one-way ANOVA.

    H0: All group means are equal.
    H1: At least one group mean is different.
    """

    anova_df = prepare_anova_data(
        df,
        numeric_column,
        [factor_column]
    )

    formula = (
        f'Q("{numeric_column}") '
        f'~ C(Q("{factor_column}"))'
    )

    model = ols(
        formula,
        data=anova_df
    ).fit()

    raw_anova_table = sm.stats.anova_lm(
        model,
        typ=2
    )

    f_statistic = float(
        raw_anova_table["F"].iloc[0]
    )

    p_value = float(
        raw_anova_table["PR(>F)"].iloc[0]
    )

    decision, conclusion = interpret_p_value(
        p_value,
        alpha
    )

    groups = [
        group_data[numeric_column]
        .dropna()
        .to_numpy()

        for _, group_data
        in anova_df.groupby(
            factor_column,
            sort=False
        )
    ]

    levene_statistic, levene_p_value = _run_levene_test(
        groups
    )

    group_summary = (
        anova_df
        .groupby(
            factor_column,
            sort=False
        )[numeric_column]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max"
        ])
        .reset_index()
    )

    return {
        "Test": "One-way ANOVA",

        "H0": (
            f"The mean of {numeric_column} is equal across "
            f"all groups of {factor_column}."
        ),

        "H1": (
            f"At least one group mean of {numeric_column} "
            f"is different across {factor_column}."
        ),

        "Numeric Column": numeric_column,
        "Factor Column": factor_column,

        "Number of Groups": int(
            anova_df[factor_column].nunique()
        ),

        "F-statistic": f_statistic,
        "p-value": p_value,
        "Alpha": alpha,

        "Decision": decision,
        "Conclusion": conclusion,

        "Levene Statistic": levene_statistic,
        "Levene p-value": levene_p_value,

        "Model": model,

        "ANOVA Table": create_anova_table(
            model,
            typ=2
        ),

        "Group Summary": group_summary,
    }


def run_tukey_hsd(
    df,
    numeric_column,
    factor_column,
    alpha=0.05
):
    """
    Runs Tukey HSD post-hoc comparisons after one-way ANOVA.
    """

    anova_df = prepare_anova_data(
        df,
        numeric_column,
        [factor_column]
    )

    if anova_df[factor_column].nunique() < 2:
        raise ValueError(
            "Tukey HSD requires at least two groups."
        )

    tukey_result = pairwise_tukeyhsd(
        endog=anova_df[numeric_column],
        groups=anova_df[factor_column],
        alpha=alpha
    )

    tukey_table = tukey_result.summary()

    tukey_df = pd.DataFrame(
        data=tukey_table.data[1:],
        columns=tukey_table.data[0]
    )

    return tukey_df


def plot_one_way_anova(
    df,
    numeric_column,
    factor_column
):
    """
    Creates a Plotly boxplot for one-way ANOVA groups.
    """

    anova_df = prepare_anova_data(
        df,
        numeric_column,
        [factor_column]
    )

    fig = go.Figure()

    group_names = (
        anova_df[factor_column]
        .drop_duplicates()
        .tolist()
    )

    for index, group_name in enumerate(group_names):

        group_values = anova_df.loc[
            anova_df[factor_column] == group_name,
            numeric_column
        ]

        colour = COLOR_SEQUENCE[
            index % len(COLOR_SEQUENCE)
        ]

        fig.add_trace(
            go.Box(
                y=group_values,
                name=str(group_name),

                boxmean=True,

                marker_color=colour,
                line_color=colour,

                # Valid Plotly transparency format
                fillcolor=hex_to_rgba(
                    colour,
                    0.18
                ),

                hovertemplate=(
                    f"{factor_column}: {group_name}<br>"
                    f"{numeric_column}: %{{y:.4f}}"
                    "<extra></extra>"
                ),
            )
        )

    fig = apply_plotly_theme(
        fig,
        title=f"{numeric_column} by {factor_column}",
        x_title=factor_column,
        y_title=numeric_column,
        height=440
    )

    fig.update_xaxes(
        type="category",
        tickangle=25
    )

    fig.update_layout(
        showlegend=False
    )

    return fig


# ------------------------------------------------------------
# Two-way ANOVA
# ------------------------------------------------------------

def run_two_way_anova(
    df,
    numeric_column,
    factor1,
    factor2,
    alpha=0.05
):
    """
    Runs two-way ANOVA with an interaction term.

    H0 for Factor 1:
        Factor 1 has no effect on the outcome.

    H0 for Factor 2:
        Factor 2 has no effect on the outcome.

    H0 for interaction:
        There is no interaction between the two factors.
    """

    if factor1 == factor2:
        raise ValueError(
            "Factor 1 and Factor 2 must be different columns."
        )

    anova_df = prepare_anova_data(
        df,
        numeric_column,
        [factor1, factor2]
    )

    formula = (
        f'Q("{numeric_column}") ~ '
        f'C(Q("{factor1}")) * C(Q("{factor2}"))'
    )

    model = ols(
        formula,
        data=anova_df
    ).fit()

    raw_anova_table = sm.stats.anova_lm(
        model,
        typ=2
    )

    formatted_anova_table = create_anova_table(
        model,
        typ=2
    )

    group_summary = (
        anova_df
        .groupby(
            [factor1, factor2],
            sort=False
        )[numeric_column]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max"
        ])
        .reset_index()
    )

    combination_column = "_anova_group_combination"

    anova_df[combination_column] = (
        anova_df[factor1].astype(str)
        + " | "
        + anova_df[factor2].astype(str)
    )

    groups = [
        group_data[numeric_column]
        .dropna()
        .to_numpy()

        for _, group_data
        in anova_df.groupby(
            combination_column,
            sort=False
        )
    ]

    levene_statistic, levene_p_value = _run_levene_test(
        groups
    )

    effects = []

    for source, row in raw_anova_table.iterrows():

        if source == "Residual":
            continue

        f_statistic = row.get(
            "F",
            np.nan
        )

        p_value = row.get(
            "PR(>F)",
            np.nan
        )

        decision, conclusion = interpret_p_value(
            p_value,
            alpha
        )

        effects.append({
            "Effect": source,
            "F-statistic": format_number(
                f_statistic
            ),
            "p-value": format_number(
                p_value
            ),
            "Decision": decision,
            "Conclusion": conclusion,
        })

    effects_table = pd.DataFrame(
        effects
    )

    return {
        "Test": "Two-way ANOVA",

        "H0 Factor 1": (
            f"{factor1} has no effect on the mean "
            f"of {numeric_column}."
        ),

        "H0 Factor 2": (
            f"{factor2} has no effect on the mean "
            f"of {numeric_column}."
        ),

        "H0 Interaction": (
            f"There is no interaction effect between "
            f"{factor1} and {factor2}."
        ),

        "Numeric Column": numeric_column,
        "Factor 1": factor1,
        "Factor 2": factor2,

        "Alpha": alpha,

        "Model": model,

        "ANOVA Table": formatted_anova_table,
        "Effects Table": effects_table,
        "Group Summary": group_summary,

        "Levene Statistic": levene_statistic,
        "Levene p-value": levene_p_value,
    }


def plot_two_way_interaction(
    df,
    numeric_column,
    factor1,
    factor2
):
    """
    Creates a Plotly interaction plot for two-way ANOVA.

    Roughly parallel lines suggest weak interaction.

    Crossing or strongly diverging lines may indicate an
    interaction between the factors.
    """

    if factor1 == factor2:
        raise ValueError(
            "Factor 1 and Factor 2 must be different columns."
        )

    clean_df = prepare_anova_data(
        df,
        numeric_column,
        [factor1, factor2]
    )

    summary_df = (
        clean_df
        .groupby(
            [factor1, factor2],
            sort=False
        )[numeric_column]
        .mean()
        .reset_index()
    )

    fig = go.Figure()

    factor2_levels = (
        summary_df[factor2]
        .drop_duplicates()
        .tolist()
    )

    for index, factor2_level in enumerate(factor2_levels):

        subset = summary_df[
            summary_df[factor2] == factor2_level
        ]

        colour = COLOR_SEQUENCE[
            index % len(COLOR_SEQUENCE)
        ]

        fig.add_trace(
            go.Scatter(
                x=subset[factor1],
                y=subset[numeric_column],

                mode="lines+markers",
                name=str(factor2_level),

                line={
                    "width": 3,
                    "color": colour,
                },

                marker={
                    "size": 9,
                    "color": colour,
                    "line": {
                        "width": 1,
                        "color": PLOT_COLORS["card"],
                    },
                },

                hovertemplate=(
                    f"{factor1}: %{{x}}<br>"
                    f"{factor2}: {factor2_level}<br>"
                    f"Mean {numeric_column}: %{{y:.4f}}"
                    "<extra></extra>"
                ),
            )
        )

    fig = apply_plotly_theme(
        fig,
        title=f"Interaction Plot: {factor1} × {factor2}",
        x_title=factor1,
        y_title=f"Mean {numeric_column}",
        height=470
    )

    fig.update_xaxes(
        type="category"
    )

    fig.update_layout(
        legend_title_text=factor2
    )

    return fig


# ------------------------------------------------------------
# Interpretation helpers
# ------------------------------------------------------------

def get_one_way_anova_interpretation(result):
    """
    Creates a plain-English interpretation of one-way ANOVA.
    """

    interpretation = [
        f"**H0:** {result['H0']}",
        f"**H1:** {result['H1']}",

        (
            f"The F-statistic is "
            f"{format_number(result['F-statistic'])}, "
            f"and the p-value is "
            f"{format_number(result['p-value'])}."
        ),
    ]

    if result["Decision"] == "Reject H0":

        interpretation.extend([
            "Because the p-value is less than alpha, we reject H0.",

            (
                "This means at least one group mean is "
                "statistically different."
            ),

            (
                "ANOVA shows that a difference exists, but it does not "
                "identify the specific groups. Use Tukey HSD for "
                "pairwise comparisons."
            ),
        ])

    else:

        interpretation.extend([
            (
                "Because the p-value is greater than or equal to alpha, "
                "we fail to reject H0."
            ),

            (
                "There is not enough evidence to conclude that the "
                "group means are different."
            ),
        ])

    levene_p_value = result.get(
        "Levene p-value",
        np.nan
    )

    if pd.isna(levene_p_value):

        interpretation.append(
            "Levene's test could not be calculated because too few "
            "usable groups were available."
        )

    elif levene_p_value < result["Alpha"]:

        interpretation.append(
            "Levene's test suggests that the equal-variance "
            "assumption may be violated."
        )

    else:

        interpretation.append(
            "Levene's test does not indicate a serious "
            "equal-variance problem."
        )

    return interpretation


def get_two_way_anova_interpretation(result):
    """
    Creates a plain-English interpretation of two-way ANOVA.
    """

    interpretation = [
        f"**H0 Factor 1:** {result['H0 Factor 1']}",
        f"**H0 Factor 2:** {result['H0 Factor 2']}",
        f"**H0 Interaction:** {result['H0 Interaction']}",
    ]

    effects_table = result["Effects Table"]

    for _, row in effects_table.iterrows():

        effect = row["Effect"]
        p_value = row["p-value"]
        decision = row["Decision"]

        if decision == "Reject H0":

            interpretation.append(
                f"For `{effect}`, p-value = {p_value}. "
                "This effect is statistically significant."
            )

        elif decision == "Fail to Reject H0":

            interpretation.append(
                f"For `{effect}`, p-value = {p_value}. "
                "This effect is not statistically significant."
            )

        else:

            interpretation.append(
                f"For `{effect}`, the statistical decision "
                "could not be calculated."
            )

    levene_p_value = result.get(
        "Levene p-value",
        np.nan
    )

    if pd.isna(levene_p_value):

        interpretation.append(
            "Levene's test could not be calculated because some "
            "factor combinations had too few observations."
        )

    elif levene_p_value < result["Alpha"]:

        interpretation.append(
            "Levene's test suggests unequal variances across "
            "factor combinations."
        )

    else:

        interpretation.append(
            "Levene's test does not indicate a serious equal-variance "
            "problem across factor combinations."
        )

    interpretation.append(
        "The interaction effect is especially important. A significant "
        "interaction means the effect of one factor changes depending "
        "on the level of the other factor."
    )

    return interpretation