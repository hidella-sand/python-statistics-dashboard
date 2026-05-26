import pandas as pd
import numpy as np
from pathlib import Path


def make_unique_columns(columns):
    """
    Makes sure column names are unique.
    Example:
    ['Age', 'Age'] becomes ['Age', 'Age_2']
    """
    seen = {}
    new_columns = []

    for col in columns:
        col = str(col).strip()

        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    return new_columns


def load_dataset(uploaded_file):
    """
    Loads CSV or Excel dataset uploaded from Streamlit.
    Returns a pandas DataFrame.
    """

    file_name = uploaded_file.name.lower()
    file_extension = Path(file_name).suffix

    try:
        if file_extension == ".csv":
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

        elif file_extension in [".xlsx"]:
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError("Unsupported file type. Please upload a CSV or XLSX file.")

        # Clean column names slightly
        df.columns = make_unique_columns(df.columns)

        return df

    except Exception as error:
        raise Exception(f"Error loading dataset: {error}")


def get_basic_dataset_info(df):
    """
    Returns basic information about the uploaded dataset.
    """

    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

    return info


def get_column_type(series):
    """
    Detects whether a column is numerical, categorical, datetime, or text.
    """

    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    unique_count = series.nunique(dropna=True)
    total_count = len(series)

    if total_count == 0:
        return "Unknown"

    unique_ratio = unique_count / total_count

    if unique_count <= 20:
        return "Categorical"

    if unique_ratio > 0.8:
        return "Text / Identifier"

    return "Text"


def suggest_column_action(column_name, series):
    """
    Suggests whether a column should be kept or reviewed/removed.
    This is only a suggestion. User can still choose manually.
    """

    name = column_name.lower().strip()
    total_count = len(series)
    unique_count = series.nunique(dropna=True)

    if total_count == 0:
        return "Review", "Empty or unknown column"

    unique_ratio = unique_count / total_count

    identifier_keywords = [
        "id", "student_id", "user_id", "customer_id", "transaction_id",
        "name", "email", "phone", "mobile", "address", "card", "passport",
        "nic", "uuid", "index"
    ]

    # Strong name-based signals
    for keyword in identifier_keywords:
        if keyword in name:
            return "Remove", "Looks like an identifier or personal information column"

    # Almost every value is unique
    if unique_ratio > 0.95 and unique_count > 20:
        return "Review", "Too many unique values; may be an ID/name-like column"

    # Useful numerical column
    if pd.api.types.is_numeric_dtype(series) and unique_count > 2:
        return "Keep", "Useful numerical column for statistics and plots"

    # Useful categorical column
    if unique_count <= 20:
        return "Keep", "Useful categorical column for grouping or tests"

    return "Review", "May be useful, but should be checked manually"


def get_column_summary(df):
    """
    Creates a summary table for all columns.
    This will be shown before the checkbox selection.
    """

    summary_rows = []

    for column in df.columns:
        series = df[column]

        missing_count = int(series.isna().sum())
        missing_percentage = round((missing_count / len(df)) * 100, 2) if len(df) > 0 else 0
        unique_count = int(series.nunique(dropna=True))
        column_type = get_column_type(series)
        suggestion, reason = suggest_column_action(column, series)

        sample_values = series.dropna().unique()[:3]
        sample_values = ", ".join([str(value) for value in sample_values])

        summary_rows.append({
            "Column": column,
            "Detected Type": column_type,
            "Missing Values": missing_count,
            "Missing %": missing_percentage,
            "Unique Values": unique_count,
            "Suggestion": suggestion,
            "Reason": reason,
            "Sample Values": sample_values
        })

    return pd.DataFrame(summary_rows)


def get_selected_dataframe(df, selected_columns):
    """
    Returns a new dataframe with only selected columns.
    Original dataframe is not changed.
    """

    return df[selected_columns].copy()


def get_numerical_columns(df):
    """
    Returns list of numerical columns.
    """

    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """
    Returns list of categorical/text columns.
    """

    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()