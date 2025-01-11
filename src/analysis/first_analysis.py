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


def plot_difference_histograms(df_combined, max_plots=10):

    """
    Plots histograms of day-to-day differences (ΔP) for each time interval in the input DataFrame.
    Each column represents a time interval (e.g., '00_00', '00_15', etc.) and the day-to-day 
    differences (ΔP) are calculated by subtracting the previous day's value from the current day's value.
    The histograms of these differences are overlaid with a normal distribution curve fitted to the data.
    The function also prints the mean and standard deviation for each difference column, 
    as well as the overall mean of all the difference columns.
    
    Parameters:
    -----------
    df_combined : pandas.DataFrame
        A DataFrame where each column represents a specific time interval (e.g., '00_00', '00_15', etc.),
        and contains day-to-day differences (ΔP) in the values. The columns should be prefixed with 'delta_P_' 
        (e.g., 'delta_P_onshore_00_00'). These differences are computed by subtracting the previous day's value 
        from the current day's value for each time interval.
    
    max_plots : int, optional, default=10
        The maximum number of histograms to plot. If there are more than 'max_plots' columns with 'delta_P_',
        only the first 'max_plots' columns will be plotted.
    
    Returns:
    --------
    None
        This function displays histograms for each column with day-to-day differences (ΔP), and prints 
        statistical information (mean and standard deviation) for each difference column. Additionally, 
        it prints the overall mean of all the difference columns.
    """
    
    plot_counter = 0

    for col in df_combined.columns:
        if 'delta_P_' in col:  # Only plot histograms for the difference columns
            data = df_combined[col].dropna()
            # Calculate the mean and standard deviation of the data
            mean, std = np.mean(data), np.std(data)
            print(mean)
            
            # Create the range of x values (the bin edges or a finer range)
            x = np.linspace(data.min(), data.max(), 100)
            
            # Get the PDF of the normal distribution
            pdf = stats.norm.pdf(x, mean, std)
            
            # Plot the histogram
            plt.figure(figsize=(8, 6))
            plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.6)
            
            # Plot the normal distribution curve
            plt.plot(x, pdf, 'r-', label=f'Normal Fit\n$\mu={mean:.2f}$, $\sigma={std:.2f}$')
            
            # Add labels and title
            plt.title(f'Frequency Distribution of {col}', fontsize=12)
            plt.xlabel('Difference (ΔP)', fontsize=10)
            plt.ylabel('Density', fontsize=10)
            plt.legend(loc='upper right')
            plt.show()  # Display the plot for each column

            # Increment the plot counter
            plot_counter += 1
            
            # Stop after 10 plots
            if plot_counter >= max_plots:
                break  # Exit the loop after plotting 10 histograms
    overall_mean = df_combined.loc[:, [c for c in df_combined.columns if 'delta_P_' in c]].mean().mean()
    print(f'Overall Mean of all column means: {overall_mean}')
    

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


def plot_weibull_parameters_distribution(mles, parameter, kde=True):
    """
    Plots the distribution (using Kernel Density Estimation, KDE) of a specified Weibull parameter (shape or scale) from MLE results.

    Parameters:
    -----------
    mles : pd.DataFrame
        A DataFrame containing the MLE parameters for the Weibull distribution, including shape and scale parameters.
    parameter : str
        The Weibull parameter to plot ('shape' or 'scale').
    kde : bool, optional, default=True
        Whether to plot the Kernel Density Estimate (KDE) of the parameter distribution.

    Returns:
    --------
    None
    """
    if parameter not in mles.columns:
        print(f"Parameter '{parameter}' not found in MLE results. Available columns: {mles.columns}")
        return
    
    plt.figure(figsize=(8, 6))
    sns.kdeplot(mles[parameter], fill=kde, label=parameter.capitalize(), color='blue')
    plt.title(f'Distribution of Weibull {parameter.capitalize()} Parameter')
    plt.xlabel(f'{parameter.capitalize()}')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()

def plot_weibull_for_one_interval(dataframe, time_interval):
    """
    Plots a histogram and the fitted Weibull distribution for a specific time interval from the input DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        A DataFrame with time-series data for different time intervals.
    time_interval : str
        The specific time interval (column) for which to plot the histogram and fitted Weibull distribution.

    Returns:
    --------
    None
    """
    if time_interval not in dataframe.columns:
        print(f"Time interval '{time_interval}' not found in the DataFrame.")
        return

    data = dataframe[time_interval].dropna()  # Clean data
    if data.empty:
        print(f"No data available for the time interval '{time_interval}'.")
        return
    
    plt.figure(figsize=(10, 6))
    # Plot histogram
    plt.hist(data, bins=30, density=True, alpha=0.6, color='blue', label='Data Histogram')

    # Fit Weibull distribution
    shape, loc, scale = weibull_min.fit(data, floc=0)
    x = np.linspace(0, data.max(), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, weibull_pdf, 'r-', lw=2, label=f'Weibull Fit (shape={shape:.2f}, scale={scale:.2f})')

    # Finalize the plot
    plt.title(f'Histogram and Weibull Fit for Time Interval {time_interval}')
    plt.xlabel('Power (kW)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()


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
    train = df.loc[df.index <= train_end_date, :].copy()
    validation = df.loc[
        (train_end_date < df.index) & (df.index <= validation_end_date), :
    ].copy()
    test = df.loc[validation_end_date < df.index, :]

    # Calculate sizes
    n = len(df)
    n_train, n_val, n_test = len(train), len(validation), len(test)

    # Logging the results
    print(f"# of training observations: {n_train} | {(n_train / n * 100):.2f}%")
    print(f"# of validation observations: {n_val} | {(n_val / n * 100):.2f}%")
    print(f"# of test observations: {n_test} | {(n_test / n * 100):.2f}%")

    return train, validation, test


def load_installed_capacity(start_date="2017-01-01", end_date="2024-01-01", method="linear"):
    """
    Loads the installed capacity data of all onshore wind turbines of the North German supplier 50Hertz in MW
    from 2017-06-01 - 2023-06-01 and interpolates missing values at 15-minute intervals.

    Parameters:
    -----------
    start_date : str, optional, default="2017-01-01"
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str, optional, default="2024-01-01"
        The end date for the data range in 'YYYY-MM-DD' format.
    method : str, optional, default="linear"
        The interpolation method to use (e.g., "linear", "polynomial").

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the interpolated installed capacity data at 15-minute intervals for the given date range.
    """

    data = {
        'date': ['2017-06-01', '2018-06-01', '2019-06-01', '2020-06-01', '2021-06-01', '2022-06-01', '2023-06-01'],
        'installed_capacity (MW)': [17866, 18346, 18711, 19138, 19748, 20414, 21078]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Convert 'year' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Set 'year' as the index
    df.set_index('date', inplace=True)

    # Resample to 15-minute intervals and interpolate using the specified method
    df = df.resample("15min").interpolate(method=method)

    dt = pd.date_range(start_date, end_date, inclusive="left", freq="15min")
    df = df.reindex(dt).bfill().ffill()

    return df



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



def plot_wind_parks_cumulative_capacity(aggregated_df):
    """
    Plots the cumulative percentage of installed capacity for wind parks, sorted by size.

    Parameters:
        aggregated_df (pd.DataFrame): A DataFrame containing 'installed_capacity_sum' and 
                                      'cumulative_percentage' columns.

    Returns:
        None
    """
    # Sort the DataFrame by installed_capacity_sum
    sorted_df = aggregated_df.sort_values(by='installed_capacity_sum', ascending=False).reset_index()

    # Create a plot for cumulative percentage
    plt.figure(figsize=(10, 6))

    # Plot cumulative percentage
    plt.plot(sorted_df.index, sorted_df['cumulative_percentage'], 
             label='Cumulative Percentage', color='blue', marker='o')

    # Add labels and title
    plt.xlabel('Index (Sorted by Installed Capacity)')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Cumulative Percentage of Installed Capacity for Wind Parks')
    plt.axhline(100, color='red', linestyle='--', label='100% Threshold')  # Optional line for 100% mark

    # Add grid for better readability
    plt.grid(visible=True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_installed_capacity_scatter(aggregated_df):
    """
    Creates a scatter plot of installed capacity with points sized based on capacity
    and highlights the weighted average longitude and latitude.

    Parameters:
        aggregated_df (pd.DataFrame): A DataFrame containing 'installed_capacity_sum', 
                                      'longitude', and 'latitude' columns.

    Returns:
        None
    """
    # Scale the size for better visualization (adjust the scaling factor as needed)
    sizes = aggregated_df['installed_capacity_sum'] * 0.001

    # Calculate the weighted average longitude and latitude
    weighted_longitude = (
        (aggregated_df['longitude'] * aggregated_df['installed_capacity_sum']).sum() /
        aggregated_df['installed_capacity_sum'].sum()
    )
    weighted_latitude = (
        (aggregated_df['latitude'] * aggregated_df['installed_capacity_sum']).sum() /
        aggregated_df['installed_capacity_sum'].sum()
    )

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        aggregated_df['longitude'],
        aggregated_df['latitude'],
        s=sizes,  # Set the size of the points based on 'installed_capacity_sum'
        alpha=0.5
    )

    # Add lines for weighted averages
    plt.axhline(weighted_latitude, color='red', linestyle='--', label='Total Weighted Average Latitude')
    plt.axvline(weighted_longitude, color='blue', linestyle='--', label='Total Weighted Average Longitude')

    # Highlight the current mean
    plt.scatter(13.125, 53.00, color='red', s=100, label='Current Mean (13.125, 53.00)', zorder=5, marker='x')

    # Add labels, title, and legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Installed Capacity')
    plt.legend()

    # Show the plot
    plt.show()



def plot_top_10_wind_parks_by_installed_capacity(aggregated_df):
    """
    Creates a scatter plot of wind park locations, highlighting the 10 largest wind parks 
    in terms of installed capacity.

    Parameters:
        aggregated_df (pd.DataFrame): A DataFrame containing 'installed_capacity_sum', 
                                      'longitude', and 'latitude' columns.

    Returns:
        None
    """
    # Scale the size for better visualization (adjust the scaling factor as needed)
    sizes = aggregated_df['installed_capacity_sum'] * 0.001

    # Calculate the weighted average longitude and latitude
    weighted_longitude = (
        (aggregated_df['longitude'] * aggregated_df['installed_capacity_sum']).sum() /
        aggregated_df['installed_capacity_sum'].sum()
    )
    weighted_latitude = (
        (aggregated_df['latitude'] * aggregated_df['installed_capacity_sum']).sum() /
        aggregated_df['installed_capacity_sum'].sum()
    )

    # Identify the indices of the 10 largest wind parks
    largest_indices = aggregated_df['installed_capacity_sum'].nlargest(10).index

    # Create scatter plot
    plt.figure(figsize=(10, 6))

    # Plot all wind park locations
    plt.scatter(
        aggregated_df['longitude'],
        aggregated_df['latitude'],
        s=sizes,  # Set the size of the points based on 'installed_capacity_sum'
        alpha=0.5,
        label='Other Points',
        color='gray'
    )

    # Highlight the 10 largest wind parks
    plt.scatter(
        aggregated_df.loc[largest_indices, 'longitude'],
        aggregated_df.loc[largest_indices, 'latitude'],
        s=sizes[largest_indices],  # Use the same sizes
        alpha=0.8,
        color='orange',
        label='Top 10 Largest'
    )

    # Plot weighted averages and markers
    plt.axhline(weighted_latitude, color='red', linestyle='--', label='Total Weighted Average Latitude')
    plt.axvline(weighted_longitude, color='blue', linestyle='--', label='Total Weighted Average Longitude')
    plt.scatter(13.125, 53.00, color='red', s=100, label='Current Mean (13.125, 53.00)', zorder=5, marker='x')

    # Add labels, title, and legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Installed Capacity with Top 10 Wind Parks Highlighted')
    plt.legend()

    # Show the plot
    plt.show()
