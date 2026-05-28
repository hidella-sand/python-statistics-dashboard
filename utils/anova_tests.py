import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats


def format_number(value):
    """
    Formats numbers nicely for result tables.
    """
    try:
        return round(float(value), 5)
    except Exception:
        return value


def interpret_p_value(p_value, alpha=0.05):
    """
    Interprets p-value for ANOVA.
    """
    if p_value < alpha:
        return "Reject H0", "There is a statistically significant difference."
    else:
        return "Fail to Reject H0", "There is not enough evidence for a statistically significant difference."


def prepare_anova_data(df, numeric_column, factor_columns):
    """
    Prepares data for ANOVA by keeping only required columns,
    converting the outcome to numeric, and removing missing values.
    """

    required_columns = [numeric_column] + factor_columns

    anova_df = df[required_columns].copy()

    anova_df[numeric_column] = pd.to_numeric(anova_df[numeric_column], errors="coerce")

    for col in factor_columns:
        anova_df[col] = anova_df[col].astype(str)

    anova_df = anova_df.dropna()

    if anova_df.empty:
        raise ValueError("No valid data available after removing missing values.")

    if anova_df[numeric_column].nunique() < 2:
        raise ValueError("The numerical column must contain at least two different values.")

    for col in factor_columns:
        if anova_df[col].nunique() < 2:
            raise ValueError(f"The factor column '{col}' must contain at least two groups.")

    return anova_df


def create_anova_table(model, typ=2):
    """
    Creates ANOVA table from fitted statsmodels model.
    typ=2 is commonly used for balanced or reasonably balanced designs.
    """

    table = sm.stats.anova_lm(model, typ=typ)
    table = table.reset_index()
    table = table.rename(columns={"index": "Source"})

    formatted_rows = []

    for _, row in table.iterrows():
        formatted_rows.append({
            "Source": row.get("Source"),
            "Sum of Squares": format_number(row.get("sum_sq")),
            "df": format_number(row.get("df")),
            "F-statistic": format_number(row.get("F")),
            "p-value": format_number(row.get("PR(>F)"))
        })

    return pd.DataFrame(formatted_rows)


# ------------------------------------------------------------
# One-way ANOVA
# ------------------------------------------------------------

def run_one_way_anova(df, numeric_column, factor_column, alpha=0.05):
    """
    Runs one-way ANOVA.

    H0: All group means are equal.
    H1: At least one group mean is different.
    """

    anova_df = prepare_anova_data(df, numeric_column, [factor_column])

    formula = f'Q("{numeric_column}") ~ C(Q("{factor_column}"))'
    model = ols(formula, data=anova_df).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    p_value = anova_table["PR(>F)"].iloc[0]
    f_statistic = anova_table["F"].iloc[0]

    decision, conclusion = interpret_p_value(p_value, alpha)

    groups = [
        group[numeric_column].values
        for _, group in anova_df.groupby(factor_column)
    ]

    levene_stat, levene_p = stats.levene(*groups)

    group_summary = anova_df.groupby(factor_column)[numeric_column].agg(
        ["count", "mean", "std", "min", "max"]
    ).reset_index()

    result = {
        "Test": "One-way ANOVA",
        "H0": f"The mean of {numeric_column} is equal across all groups of {factor_column}.",
        "H1": f"At least one group mean of {numeric_column} is different across {factor_column}.",
        "Numeric Column": numeric_column,
        "Factor Column": factor_column,
        "Number of Groups": anova_df[factor_column].nunique(),
        "F-statistic": f_statistic,
        "p-value": p_value,
        "Alpha": alpha,
        "Decision": decision,
        "Conclusion": conclusion,
        "Levene Statistic": levene_stat,
        "Levene p-value": levene_p,
        "Model": model,
        "ANOVA Table": create_anova_table(model),
        "Group Summary": group_summary
    }

    return result


def run_tukey_hsd(df, numeric_column, factor_column, alpha=0.05):
    """
    Runs Tukey HSD post-hoc test for one-way ANOVA.
    Useful when ANOVA is significant.
    """

    anova_df = prepare_anova_data(df, numeric_column, [factor_column])

    tukey = pairwise_tukeyhsd(
        endog=anova_df[numeric_column],
        groups=anova_df[factor_column],
        alpha=alpha
    )

    tukey_df = pd.DataFrame(
        data=tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    return tukey_df


def plot_one_way_anova(df, numeric_column, factor_column):
    """
    Boxplot for one-way ANOVA.
    """

    anova_df = prepare_anova_data(df, numeric_column, [factor_column])

    group_names = sorted(anova_df[factor_column].unique())
    group_data = [
        anova_df[anova_df[factor_column] == group][numeric_column]
        for group in group_names
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.boxplot(group_data, tick_labels=group_names)

    ax.set_title(f"{numeric_column} by {factor_column}", fontsize=12)
    ax.set_xlabel(factor_column)
    ax.set_ylabel(numeric_column)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=25)

    return fig


# ------------------------------------------------------------
# Two-way ANOVA
# ------------------------------------------------------------

def run_two_way_anova(df, numeric_column, factor1, factor2, alpha=0.05):
    """
    Runs two-way ANOVA with interaction.

    H0 for factor 1: Factor 1 has no effect on the mean.
    H0 for factor 2: Factor 2 has no effect on the mean.
    H0 for interaction: There is no interaction between factor 1 and factor 2.
    """

    anova_df = prepare_anova_data(df, numeric_column, [factor1, factor2])

    formula = f'Q("{numeric_column}") ~ C(Q("{factor1}")) * C(Q("{factor2}"))'
    model = ols(formula, data=anova_df).fit()

    anova_table_raw = sm.stats.anova_lm(model, typ=2)
    anova_table = create_anova_table(model)

    group_summary = anova_df.groupby([factor1, factor2])[numeric_column].agg(
        ["count", "mean", "std", "min", "max"]
    ).reset_index()

    # Levene test across factor combinations
    combination_column = "_anova_group_combination"
    anova_df[combination_column] = anova_df[factor1].astype(str) + " | " + anova_df[factor2].astype(str)

    groups = [
        group[numeric_column].values
        for _, group in anova_df.groupby(combination_column)
        if len(group) >= 2
    ]

    if len(groups) >= 2:
        levene_stat, levene_p = stats.levene(*groups)
    else:
        levene_stat, levene_p = np.nan, np.nan

    effects = []

    for source, row in anova_table_raw.iterrows():
        if source == "Residual":
            continue

        p_value = row["PR(>F)"]
        decision, conclusion = interpret_p_value(p_value, alpha)

        effects.append({
            "Effect": source,
            "F-statistic": format_number(row["F"]),
            "p-value": format_number(p_value),
            "Decision": decision,
            "Conclusion": conclusion
        })

    effects_df = pd.DataFrame(effects)

    result = {
        "Test": "Two-way ANOVA",
        "H0 Factor 1": f"{factor1} has no effect on the mean of {numeric_column}.",
        "H0 Factor 2": f"{factor2} has no effect on the mean of {numeric_column}.",
        "H0 Interaction": f"There is no interaction effect between {factor1} and {factor2}.",
        "Numeric Column": numeric_column,
        "Factor 1": factor1,
        "Factor 2": factor2,
        "Alpha": alpha,
        "Model": model,
        "ANOVA Table": anova_table,
        "Effects Table": effects_df,
        "Group Summary": group_summary,
        "Levene Statistic": levene_stat,
        "Levene p-value": levene_p
    }

    return result


def plot_two_way_interaction(df, numeric_column, factor1, factor2):
    """
    Interaction plot for two-way ANOVA.
    Shows how mean of numeric column changes across factor1 for each level of factor2.
    """

    anova_df = prepare_anova_data(df, numeric_column, [factor1, factor2])

    mean_df = anova_df.groupby([factor1, factor2])[numeric_column].mean().reset_index()

    factor1_levels = sorted(mean_df[factor1].unique())
    factor2_levels = sorted(mean_df[factor2].unique())

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for level in factor2_levels:
        subset = mean_df[mean_df[factor2] == level]

        y_values = []

        for f1_level in factor1_levels:
            value = subset[subset[factor1] == f1_level][numeric_column]

            if len(value) > 0:
                y_values.append(value.iloc[0])
            else:
                y_values.append(np.nan)

        ax.plot(factor1_levels, y_values, marker="o", label=str(level))

    ax.set_title(f"Interaction Plot: {factor1} × {factor2}", fontsize=12)
    ax.set_xlabel(factor1)
    ax.set_ylabel(f"Mean {numeric_column}")
    ax.grid(True, alpha=0.3)
    ax.legend(title=factor2)

    plt.xticks(rotation=25)

    return fig


# ------------------------------------------------------------
# Interpretation helpers
# ------------------------------------------------------------

def get_one_way_anova_interpretation(result):
    """
    Plain-English interpretation for one-way ANOVA.
    """

    interpretation = []

    interpretation.append(f"**H0:** {result['H0']}")
    interpretation.append(f"**H1:** {result['H1']}")
    interpretation.append(
        f"The F-statistic is {format_number(result['F-statistic'])}, and the p-value is {format_number(result['p-value'])}."
    )

    if result["Decision"] == "Reject H0":
        interpretation.append(
            "Because the p-value is less than alpha, we reject H0."
        )
        interpretation.append(
            "This means at least one group mean is significantly different."
        )
        interpretation.append(
            "ANOVA tells us that a difference exists, but it does not directly tell which groups are different. Use Tukey HSD for that."
        )
    else:
        interpretation.append(
            "Because the p-value is greater than or equal to alpha, we fail to reject H0."
        )
        interpretation.append(
            "This means there is not enough evidence to say the group means are different."
        )

    if result["Levene p-value"] < result["Alpha"]:
        interpretation.append(
            "Levene's test suggests the equal variance assumption may be violated."
        )
    else:
        interpretation.append(
            "Levene's test does not suggest a serious equal variance problem."
        )

    return interpretation


def get_two_way_anova_interpretation(result):
    """
    Plain-English interpretation for two-way ANOVA.
    """

    interpretation = []

    interpretation.append(f"**H0 Factor 1:** {result['H0 Factor 1']}")
    interpretation.append(f"**H0 Factor 2:** {result['H0 Factor 2']}")
    interpretation.append(f"**H0 Interaction:** {result['H0 Interaction']}")

    effects_df = result["Effects Table"]

    for _, row in effects_df.iterrows():
        effect = row["Effect"]
        p_value = row["p-value"]
        decision = row["Decision"]

        if decision == "Reject H0":
            interpretation.append(
                f"For `{effect}`, p-value = {p_value}. This effect is statistically significant."
            )
        else:
            interpretation.append(
                f"For `{effect}`, p-value = {p_value}. This effect is not statistically significant."
            )

    if not pd.isna(result["Levene p-value"]):
        if result["Levene p-value"] < result["Alpha"]:
            interpretation.append(
                "Levene's test suggests the equal variance assumption may be violated across factor combinations."
            )
        else:
            interpretation.append(
                "Levene's test does not suggest a serious equal variance problem across factor combinations."
            )
    else:
        interpretation.append(
            "Levene's test could not be calculated properly because some factor combinations had too few values."
        )

    interpretation.append(
        "In two-way ANOVA, the interaction effect is very important. If the interaction is significant, the effect of one factor depends on the level of the other factor."
    )

    return interpretation