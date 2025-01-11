import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import weibull_min
from scipy.special import gamma
import seaborn as sns
from analysis.datasets import *


def minute_to_daily_50Hertz(df):
    """
    Reshapes a DataFrame containing time series data recorded at 15-minute intervals into a daily format, 
    where each row corresponds to a single day, and each column represents a specific time of day 
    (in 15-minute intervals) for a given measurement.

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame indexed by time (`DatetimeIndex`) with measurements taken at 15-minute intervals.
        The DataFrame must contain columns with time-series data (e.g., `offshore`, `onshore`) with a 
        DatetimeIndex of 15-minute frequency.
        
        Example Input:
        
        offshore        onshore
        time
        2016-01-01 00:00:00    21.0    1428.0
        2016-01-01 00:15:00    27.0    1379.0
        2016-01-01 00:30:00    24.0    1399.0
        2016-01-01 00:45:00    20.0    1448.0
        2016-01-01 01:00:00    16.0    1517.0
        ...                    ...      ...
        2024-12-31 22:45:00    NaN     NaN
        2024-12-31 23:00:00    NaN     NaN
        2024-12-31 23:15:00    NaN     NaN
        2024-12-31 23:30:00    NaN     NaN
        2024-12-31 23:45:00    NaN     NaN

    Returns:
    --------
    pd.DataFrame
        A reshaped DataFrame where each row represents a single day, and each column represents a 
        specific 15-minute time slot of that day (e.g., `offshore_00_00`, `offshore_00_15`, `onshore_23_45`).
        The measurements from the original DataFrame are aligned with their respective time of day (formatted as HH_MM).
        
        Example Output (after transformation):

                            offshore_00_00  offshore_00_15  offshore_00_30  offshore_00_45  offshore_01_00  ...  onshore_23_30  onshore_23_45
        date                                                                                                      
        2016-01-01       21.0             27.0            24.0            20.0            16.0            ...      509.0        528.0
        2016-01-02       208.0            226.0           235.0           260.0           292.0           ...      6672.0       6721.0
        2016-01-03       320.0            320.0           321.0           320.0           321.0           ...      5234.0       5220.0
        2016-01-04       319.0            320.0           320.0           320.0           320.0           ...      2100.0       2050.0
        2016-01-05       320.0            319.0           319.0           320.0           319.0           ...      1218.0       1121.0
        ...              ...              ...             ...             ...             ...             ...      ...          ...
        2024-12-27       NaN              NaN             NaN             NaN             NaN             ...      NaN          NaN

    Notes:
    ------
    - The DataFrame's index is expected to be of `DatetimeIndex` with 15-minute intervals.
    - The columns represent different time-series data (e.g., `offshore`, `onshore`) and will be split into 
      multiple columns based on the specific 15-minute time intervals of the day (e.g., `offshore_00_00`, 
      `onshore_23_45`).
    - The function extracts the date from the `DatetimeIndex` and reformats the time as `HH_MM` to create 
      the new column names for each time slot of the day.
    - Missing values (`NaN`) will be preserved in the reshaped DataFrame.
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





def to_train_validation_test_data(df: pd.DataFrame, train_end_date: str, validation_end_date: str):
    """
    Splits the dataset into train, validation, and test sets based on specified end dates.

    Parameters:
        df (pd.DataFrame): The full dataset with a DateTime index.
        train_end_date (str): The end date for the training set in 'YYYY-MM-DD' format.
        validation_end_date (str): The end date for the validation set in 'YYYY-MM-DD' format.

    Returns:
        tuple: (train_data, validation_data, test_data)
    """
    # Split the data
    train = df.loc[df.index < train_end_date, :].copy()
    validation = df.loc[
        (train_end_date <= df.index) & (df.index < validation_end_date), :
    ].copy()
    test = df.loc[validation_end_date <= df.index, :]

    # Calculate sizes
    n = len(df)
    n_train, n_val, n_test = len(train), len(validation), len(test)

    # Logging the results
    print(f"# of training observations: {n_train} | {(n_train / n * 100):.2f}%")
    print(f"# of validation observations: {n_val} | {(n_val / n * 100):.2f}%")
    print(f"# of test observations: {n_test} | {(n_test / n * 100):.2f}%")

    return train, validation, test



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
    wind_park_data_interpolated = wind_park_data_mean.resample("15min").interpolate(method="linear")
    
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

