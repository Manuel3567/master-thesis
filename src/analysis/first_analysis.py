import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis.datasets import *





def add_previous_day_and_calculate_differences(df):
    """
    Adds columns for the previous day's values and calculates day-to-day differences 
    for each time interval in the input DataFrame.
    
    This function operates on a DataFrame containing power data for both 'offshore' and 'onshore' columns 
    indexed by time. It computes the day-to-day differences (ΔP) for each time interval, by first adding 
    columns for the previous day's power values, and then calculating the differences between consecutive days.
    The resulting DataFrame includes the original power values, previous day's values, and the day-to-day differences.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing power data for 'offshore' and 'onshore' columns. The DataFrame should be indexed by 
        time, with a `DatetimeIndex` at a 15-minute frequency.
    
    Example Input:
    --------------
                                offshore  onshore
    time                                        
    2016-01-01 00:00:00            21.0    1428.0
    2016-01-01 00:15:00            27.0    1379.0
    2016-01-01 00:30:00            24.0    1399.0
    2016-01-01 00:45:00            20.0    1448.0
    2016-01-01 01:00:00            16.0    1517.0
    ...
    
    Returns:
    --------
    pandas.DataFrame
        A modified version of the input DataFrame, which includes the original values, 
        previous day's values, and the day-to-day differences for each time interval.
        The columns are prefixed as follows:
        - 'P_' for the original values
        - 'P_t-1_' for the previous day's values
        - 'delta_P_' for the day-to-day differences.

    Notes:
    ------
    - The DataFrame must have a `DatetimeIndex` with 15-minute frequency.
    - The function handles `NaN` values, which can arise when there is missing data for a specific time interval.
    """

    df = minute_to_daily_50Hertz(df)
    df = df.copy()
    df = df.drop([col for col in df.columns if "offshore" in col], axis=1)

    # Step 1: Add columns for previous day's power values for each interval
    df_prev = df.shift(1)  # Shifts all rows down by one to get the previous day's values

    # Step 2: Calculate day-to-day differences for each interval
    df_diff = df - df_prev  # Subtract shifted DataFrame to get the differences

    # Combine original DataFrame, previous day, and difference DataFrames
    # Prefix 'P_' for original, 'P_t-1_' for previous day, 'ΔP_' for differences
    df_combined = pd.concat(
        [df.add_prefix('P_'), df_prev.add_prefix('P_t-1_'), df_diff.add_prefix('delta_P_')],
        axis=1
    )
    return df_combined



    

def get_weibull_mle(data):
    """
    Estimates the maximum likelihood estimation (MLE) parameters for a Weibull distribution from the given data.

    Parameters:
    -----------
    data : pd.Series or np.array
        A series or array of values representing a sample of data to fit the Weibull distribution.

    Returns:
    --------
    tuple
        A tuple containing the shape parameter and scale parameter of the Weibull distribution.
    """
    data = data[np.isfinite(data)]
    shape_param, loc, scale_param = weibull_min.fit(data, floc=0)  # floc=0 fixes the location parameter
    return shape_param, scale_param

def get_weibull_mles(data):
    """
    Computes Weibull MLE parameters for each column in the input DataFrame and calculates the theoretical mean and error for each.

    Parameters:
    -----------
    data : pd.DataFrame
        A DataFrame with columns representing time intervals, where each column contains data for a particular time interval.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the Weibull shape, scale parameters, theoretical mean, actual mean, and relative error for each time interval.
    """
    mles = []
    for column in data.columns:
        shape, scale = get_weibull_mle(data[column])
        th_mean = scale*gamma(1+1/shape)
        mean = data[column].mean()
        error = (th_mean - mean) / mean
        mles.append({"time_interval": column, "shape": shape, "scale": scale, "th_mean": th_mean, "mean": mean, "error": error})
    return pd.DataFrame(mles)



def load_and_merge_wind_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads and merges MERRA2 and turbine data with 1-hour resampled intervals.
    
    Args:
        start_date (str): The start date for the analysis (format: 'YYYY-MM-DD').
        end_date (str): The end date for the analysis (format: 'YYYY-MM-DD').
    
    Returns:
        pd.DataFrame: Combined DataFrame with columns `ws_merra`, `ws_penn`, and `delta`.
    """
    # Load and preprocess MERRA2 data
    merra = load_merra2(start_date=start_date, end_date=end_date)
    merra = merra.drop(columns=[col for col in merra.columns if col != "wind_speed (m/s)"])
    
    # Load and preprocess turbine electricity data
    penn = load_turbine_electricity_data_dynamic(2016)
    penn = penn.drop(columns=[col for col in penn.columns if col != "Wind speed (m/s)"])
    penn = penn.resample("H").mean()  # Resampling to 1-hour intervals
    penn = penn.loc[start_date:end_date]
    
    # Merge datasets
    combined = pd.merge(merra, penn, left_index=True, right_index=True, how="inner")
    
    # Rename columns for clarity
    combined.rename(columns={"wind_speed (m/s)": "ws_merra", "Wind speed (m/s)": "ws_penn"}, inplace=True)
    
    # Calculate delta
    combined['delta'] = combined['ws_merra'] - combined['ws_penn']
    
    return combined


def analyze_wind_speed(combined: pd.DataFrame) -> None:
    """
    Analyzes the combined wind speed data by generating a scatter plot of the two sources' wind speeds.
    
    Args:
        combined (pd.DataFrame): Combined DataFrame with columns `ws_merra`, `ws_penn`, and `delta`.
    
    Returns:
        None
    """
    # Print index information
    print("Index minimum:", combined.index.min())
    print("Index maximum:", combined.index.max())
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(combined["ws_merra"], combined["ws_penn"], alpha=0.7)
    plt.title(f"Wind Speed Comparison")
    plt.xlabel("MERRA2 Wind Speed (m/s)")
    plt.ylabel("Turbine Wind Speed (m/s)")
    plt.grid(True)
    plt.show()



import pandas as pd

def process_and_merge_dataframes(wind_park_data, electricity_data):
    """
    Processes and merges two dataframes: interpolates wind_park_data, applies minute_to_daily_50Hertz, 
    filters for "mean" columns, and merges with electricity_data on time index.

    Args:
        wind_park_data (pd.DataFrame): Wind park data loaded using `load_wind_park_data()`.
        electricity_data (pd.DataFrame): ENTSO-E data loaded using `load_entsoe()`.
        minute_to_daily_50Hertz (function): Function to apply to the interpolated wind_park_data.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    # Step 1: Filter columns containing "mean"
    wind_park_data_mean = wind_park_data.filter(like="mean")
    
    # Step 2: Interpolate wind_park_data_mean from 1-hour intervals to 15-minute intervals
    wind_park_data_mean.index = pd.to_datetime(wind_park_data_mean.index)  # Ensure index is datetime
    wind_park_data_interpolated = wind_park_data_mean.resample("15T").interpolate(method="linear")
    
    # Step 3: Apply minute_to_daily_50Hertz function
    wind_park_data_processed = minute_to_daily_50Hertz(wind_park_data_interpolated)
    
    # Step 4: Load and process electricity_data (keep only the "onshore" column)
    electricity_data_processed = electricity_data[["onshore"]].copy()
    electricity_data_processed = minute_to_daily_50Hertz(electricity_data_processed)
    electricity_data_processed.index = pd.to_datetime(electricity_data_processed.index)  # Ensure index is datetime
    
    # Step 5: Merge the processed dataframes on time index
    merged_df = pd.merge(
        wind_park_data_processed,
        electricity_data_processed,
        left_index=True,
        right_index=True,
        how="inner"
    )
    
    return merged_df


def get_columns_by_time(df, time: str):
    time = time.replace(":", "_")
    columns = [c for c in df.columns if c.endswith(time)]
    df_filtered = df.loc[:, columns]

    return df_filtered
