import pandas as pd
import numpy as np


def minute_to_daily(df):
    """Stacks observations of time points in a day as columns,
    such that every observation/row represents a day.
    If a column 'electricity' is available and observations follow a 15min interval,
    create new columns 'electricity_00_00', 'electricity_00_15', 'electricity_00_30'
    The widening is repeated with any additional columns as well.
    Inverse to daily_to_minute.

    Args:
        df (pd.DataFrame): Long dataframe where multiple observations make up an entire day.

    Returns:
        pd.DataFrame: Wide dataframe where every observation is a day.
    """
    df = df.copy()
    original_shape = df.shape
    original_frequency = int((df.index[1] - df.index[0]).seconds / 60)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    columns = df.columns
    df.index.name = ""
    df.index = pd.to_datetime(df.index)

    # Extract the date and formatted time
    df["date"] = df.index.date ##creates a new column date, containing just the date (without the time) from the index
    df["time"] = df.index.strftime("%H_%M") ##addsa new column time to represent the time of day in HH_MM format

    # Perform the pivot (reshape)
    reshaped_df = df.pivot(index="date", columns="time", values=columns) #reshapes the DataFrame: each unique date --> row index, each unique time --> a column, values in each cell are the original data values

    # Flatten multi-level columns if necessary
    reshaped_df.columns = [f"{col[0]}_{col[1]}" for col in reshaped_df.columns]
    if reshaped_df.index.dtype == "object":
        reshaped_df.index = pd.to_datetime(reshaped_df.index)

    return reshaped_df

def daily_to_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Goes from wide format where the columns of the dataframe df are all x-min intervals of a day to long format where every row is a x-min interval.
    Assumes that the columns are of the form electricity_00_00, electricity_00_15, electricity_00_30
    If multiple column prefixes are available, e.g. additionally: wind_00_00, wind_00_15,
    create two columns in return frame, 'electricity', 'wind'.
    Inverse to minute_to_daily.


    Args:
        df (pd.DataFrame): Wide dataframe where all columns represent an entire day.

    Returns:
        pd.DataFrame: Long dataframe where two observations differ by the original column frequency difference.
    """
    df = df.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.index = pd.to_datetime(df.index)
    original_frequency = int((df.index[1] - df.index[0]).total_seconds() / 60 / 60 / 24)
    original_shape = df.shape
    df.index.name = "date"
    df.columns = df.columns.str.rsplit("_", n=2, expand=True)
    df.columns.names = [None, "hour", "minute"] + [None] * (len(df.columns.names) - 3)
    df = df.stack(["hour", "minute"], future_stack=True)
    df = df.reset_index()
    index = pd.to_datetime(df.date.astype(str) + " " + df.hour + ":" + df.minute)
    df = df.set_index(index)
    df = df.drop(columns=["date", "hour", "minute"])
    new_frequency = int((df.index[1] - df.index[0]).total_seconds() / 60)
    new_shape = df.shape
    return df

def scale_power_data(df, target_column='power'):
    """Scales the power data using log transformation."""
    max_power_value = df[target_column].max()
    max_power_value_rounded = np.ceil(max_power_value / 1000) * 1000
    #epsilon = 1e-9
    epsilon = 1e-3
    df[target_column] = np.log(df[target_column] / max_power_value_rounded + epsilon)
    return df

def add_lagged_features(df, target_column='power', lag=96):
    """Adds lagged power feature."""
    df[f'{target_column}_t-{lag}'] = df[target_column].shift(lag)
    return df

def add_interval_index(df):
    """Creates an interval index feature based on time."""
    df['interval_index'] = ((df.index.hour * 60 + df.index.minute) // 15) + 1
    return df
