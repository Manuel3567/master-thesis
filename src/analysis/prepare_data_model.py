import numpy as np
import pandas as pd


def transform_power(df, target_column='power'):
    """Scales the power data using log transformation."""
    max_power_value = df[target_column].max()
    max_power_value_rounded = np.ceil(max_power_value / 1000) * 1000
    #epsilon = 1e-9
    epsilon = 1e-3
    df[target_column] = np.log(df[target_column] / max_power_value_rounded + epsilon)
    return df

def add_interval_index(df):
    """Creates an interval index feature based on time."""
    df['interval_index'] = ((df.index.hour * 60 + df.index.minute) // 15) + 1
    return df

def add_lagged_features(df, target_column='power', lag=96):
    """Adds lagged power feature."""
    df[f'{target_column}_t-{lag}'] = df[target_column].shift(lag)
    return df

#old does not keep p_t-96 and power column
def prepare_features(df: pd.DataFrame, features: list):
    """
    Selects only the specified features from the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    features (list): A list of column names to be selected.

    Returns:
    pd.DataFrame: A DataFrame containing only the selected features.
    """
    return df[features] if all(feature in df.columns for feature in features) else df[[feature for feature in features if feature in df.columns]]
