import pandas as pd
import numpy as np
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


def format_number(value, decimals=4):
    """
    Formats values safely for display.
    """

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if value.is_integer():
            return str(int(value))

        return f"{value:.{decimals}f}"

    except Exception:
        return str(value)


def is_integer_like(series):
    """
    Checks whether numeric values are integer-like.
    Example: 0.0, 1.0, 2.0 are integer-like.
    """

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    if numeric_series.empty:
        return False

    return np.all(np.isclose(numeric_series, numeric_series.astype(int)))


def detect_variable_type(df, column):
    """
    Detects the best statistical interpretation of a column.
    """

    series = df[column]
    clean_series = series.dropna()

    total_count = len(series)
    unique_count = clean_series.nunique()

    if total_count == 0 or clean_series.empty:
        return {
            "type": "Empty",
            "badge": "Empty",
            "badge_type": "warning",
            "reason": "This column has no usable values."
        }

    column_lower = column.lower().strip()
    unique_ratio = unique_count / total_count

    identifier_keywords = [
        "id", "uuid", "name", "email", "phone", "mobile",
        "passport", "nic", "card", "address", "index"
    ]

    if any(keyword in column_lower for keyword in identifier_keywords):
        return {
            "type": "Identifier",
            "badge": "Identifier",
            "badge_type": "warning",
            "reason": "This column looks like an identifier and may not be useful for statistical summaries."
        }

    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "type": "Datetime",
            "badge": "Datetime",
            "badge_type": "info",
            "reason": "This column contains dates or times."
        }

    if pd.api.types.is_bool_dtype(series):
        return {
            "type": "Binary categorical",
            "badge": "Binary",
            "badge_type": "secondary",
            "reason": "This column has two logical categories."
        }

    if pd.api.types.is_numeric_dtype(series):
        if unique_count == 2:
            return {
                "type": "Binary categorical",
                "badge": "Binary",
                "badge_type": "secondary",
                "reason": "This numeric column has only two unique values, so it is better treated as binary categorical."
            }

        if unique_count <= 12 and is_integer_like(series):
            return {
                "type": "Discrete numerical",
                "badge": "Discrete",
                "badge_type": "primary",
                "reason": "This looks like count-like or integer-coded numerical data."
            }

        return {
            "type": "Continuous numerical",
            "badge": "Continuous",
            "badge_type": "success",
            "reason": "This column is suitable for numerical descriptive statistics."
        }

    if unique_count == 2:
        return {
            "type": "Binary categorical",
            "badge": "Binary",
            "badge_type": "secondary",
            "reason": "This column has two categories."
        }

    if unique_count <= 30:
        return {
            "type": "Categorical",
            "badge": "Categorical",
            "badge_type": "secondary",
            "reason": "This column has a limited number of categories."
        }

    if unique_ratio > 0.80:
        return {
            "type": "Text / Identifier",
            "badge": "Text",
            "badge_type": "warning",
            "reason": "This text column has many unique values, so it may not be useful for basic statistical summaries."
        }

    return {
        "type": "Categorical",
        "badge": "Categorical",
        "badge_type": "secondary",
        "reason": "This text column can be summarized as categorical data."
    }


def get_descriptive_candidate_columns(df):
    """
    Returns columns suitable for descriptive summary.
    Excludes columns that are very likely identifiers.
    """

    candidate_columns = []

    for column in df.columns:
        detected = detect_variable_type(df, column)

        if detected["type"] not in ["Identifier", "Text / Identifier", "Empty"]:
            candidate_columns.append(column)

    return candidate_columns


def create_categorical_summary(df, column):
    """
    Creates categorical summary dictionary and frequency table.
    """

    series = df[column]
    clean_series = series.dropna().astype(str)

    total_rows = len(series)
    valid_count = len(clean_series)
    missing_count = total_rows - valid_count
    missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0

    value_counts = clean_series.value_counts()
    percentages = clean_series.value_counts(normalize=True) * 100

    frequency_table = pd.DataFrame({
        "Category": value_counts.index,
        "Count": value_counts.values,
        "Percentage": percentages.values
    })

    frequency_table["Percentage"] = frequency_table["Percentage"].round(2)

    if len(value_counts) > 0:
        mode_value = value_counts.index[0]
        mode_count = int(value_counts.iloc[0])
        mode_percentage = float(percentages.iloc[0])
    else:
        mode_value = "N/A"
        mode_count = 0
        mode_percentage = 0

    summary = {
        "Column": column,
        "Total Rows": total_rows,
        "Valid Values": valid_count,
        "Missing Values": missing_count,
        "Missing %": missing_percentage,
        "Unique Categories": clean_series.nunique(),
        "Mode": mode_value,
        "Mode Count": mode_count,
        "Mode %": mode_percentage,
    }

    return summary, frequency_table


def apply_plotly_theme(fig, title=None, x_title=None, y_title=None, height=420):
    """
    Applies the app dark theme to Plotly charts.
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
        margin={"l": 45, "r": 25, "t": 60, "b": 45},
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


def plot_categorical_bar(frequency_table, column):
    """
    Creates an interactive bar chart for categorical data.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=frequency_table["Category"].astype(str),
            y=frequency_table["Count"],
            marker={
                "color": PLOT_COLORS["secondary"],
                "line": {"color": PLOT_COLORS["bg"], "width": 1},
            },
            hovertemplate=(
                "Category: %{x}<br>"
                "Count: %{y}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Category Counts for {column}",
        x_title=column,
        y_title="Count",
        height=420,
        
    )
    fig.update_xaxes(type="category")
    return fig


def plot_categorical_percentage_bar(frequency_table, column):
    """
    Creates an interactive percentage bar chart.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=frequency_table["Category"].astype(str),
            y=frequency_table["Percentage"],
            marker={
                "color": PLOT_COLORS["primary"],
                "line": {"color": PLOT_COLORS["bg"], "width": 1},
            },
            hovertemplate=(
                "Category: %{x}<br>"
                "Percentage: %{y:.2f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Category Percentages for {column}",
        x_title=column,
        y_title="Percentage",
        height=420,
        
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0, max(frequency_table["Percentage"]) * 1.20])

    return fig


def plot_categorical_donut(frequency_table, column):
    """
    Creates a donut chart for categorical data.
    Best for columns with a small number of categories.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=frequency_table["Category"].astype(str),
            values=frequency_table["Count"],
            hole=0.55,
            marker={
                "colors": [
                    PLOT_COLORS["primary"],
                    PLOT_COLORS["secondary"],
                    PLOT_COLORS["blue"],
                    PLOT_COLORS["green"],
                    PLOT_COLORS["warning"],
                    PLOT_COLORS["error"],
                ]
            },
            textinfo="label+percent",
            hovertemplate=(
                "Category: %{label}<br>"
                "Count: %{value}<br>"
                "Percentage: %{percent}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig = apply_plotly_theme(
        fig,
        title=f"Category Share for {column}",
        height=420,
    )

    return fig


def get_categorical_interpretation(summary, frequency_table, detected_type):
    """
    Creates plain-English interpretation for categorical/binary data.
    """

    column = summary["Column"]
    mode_value = summary["Mode"]
    mode_count = summary["Mode Count"]
    mode_percentage = summary["Mode %"]
    unique_categories = summary["Unique Categories"]
    missing_percentage = summary["Missing %"]

    interpretation = []

    if detected_type == "Binary categorical":
        interpretation.append(
            f"`{column}` is being treated as a binary categorical variable."
        )
    else:
        interpretation.append(
            f"`{column}` is being treated as a categorical variable."
        )

    interpretation.append(
        f"The most common category is `{mode_value}`, appearing {mode_count} time(s), which is {format_number(mode_percentage, 2)}% of valid values."
    )

    interpretation.append(
        f"The column has {unique_categories} unique category/categories."
    )

    if missing_percentage > 0:
        interpretation.append(
            f"{format_number(missing_percentage, 2)}% of values are missing, so missing data should be considered."
        )
    else:
        interpretation.append(
            "There are no missing values in this selected column."
        )

    if detected_type == "Binary categorical":
        interpretation.append(
            "For a 0/1 variable, the mean can be interpreted as the proportion of 1s, but category percentages are usually clearer."
        )

    if unique_categories > 10:
        interpretation.append(
            "This variable has many categories, so a bar chart may be easier to read than a donut chart."
        )

    return interpretation