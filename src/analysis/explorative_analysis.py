import pandas as pd
import matplotlib.pyplot as plt

def calculate_expected_entries(dataframe):
    """
    Calculate and print the expected and actual number of entries in the DataFrame based on its time index.

    This function calculates the expected number of rows (entries) in a time-series DataFrame by considering 
    the time difference between consecutive rows (assumed to be uniform) and the range of the time index. 
    It then compares the expected number of entries with the actual number of rows in the DataFrame and prints 
    both values.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        A pandas DataFrame with a `DatetimeIndex` that represents a time series. The function assumes that 
        the DataFrame's index is a sequence of datetime values at consistent intervals.

    Returns:
    --------
    None
        The function prints the expected and actual number of entries, but does not return any value.

    Notes:
    ------
    - The function assumes the time intervals between consecutive rows are consistent.
    - The `min_date` and `max_date` are used to determine the range of the time series, and the expected 
      number of entries is computed based on this range and the time interval between entries.

    Example:
    --------
    Given a DataFrame `df` with a `DatetimeIndex` from '2022-01-01' to '2022-01-10' and 15-minute intervals:
    
    calculate_expected_entries(df)
    # Output:
    # Expected number of entries: 576 (2022-01-01 00:00:00 - 2022-01-10 23:45:00)
    # Actual number of entries: 576
    """
    min_date = dataframe.index.min()
    max_date = dataframe.index.max()
    time_interval = dataframe.index[1] - dataframe.index[0]

    expected_entries = (max_date - min_date) // time_interval + 1
    print(f"Expected number of entries: {expected_entries} ({min_date} - {max_date}) ")
    print(f"Actual number of entries: {dataframe.shape[0]}")


def get_power_columns(dataframe):
    """Return a list of columns related to power data."""
    power_columns = [col for col in dataframe.columns if 'Power (kW)' in col or col == 'onshore']

    return power_columns

def get_wind_speed_columns(dataframe):
    """Return a list of columns related to wind speed data."""
    wind_speed_columns = [col for col in dataframe.columns if col == 'Wind speed (m/s)' or col.startswith('wind_speed_')]
    print(wind_speed_columns)
    return wind_speed_columns


def plot_power_histogram_and_monthly_mean_timeseries(dataframe, start, end):
    """Plot histograms and monthly time series for power-related columns."""

    power_columns = get_power_columns(dataframe)
    
    if not power_columns:
        print("No power-related columns found. Skipping power plots.")
        return

    for col in power_columns:
        print(f"Plotting histogram for: {col} for {start} - {end}")
        
        plt.figure()
        dataframe[col].hist()
        plt.title(f"Histogram of {col} for {start} - {end}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
        print(f"\nPlotting monthly time series of {col} data for the time period {start} - {end}:")

        
        # Plot monthly time series
        monthly_data = dataframe.resample('ME').mean()  # Resample by month, taking the mean
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_data.index, monthly_data[col], marker='o', linestyle='-', color='b', label=col)
        plt.title(f'Monthly Time Series of {col} Data for {start} - {end}')
        plt.xlabel('Month')
        plt.ylabel(f'{col}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_wind_speed_histogram(dataframe, start, end):
    """Plot histograms for wind speed-related columns."""
    wind_speed_columns = get_wind_speed_columns(dataframe)
    
    if not wind_speed_columns:
        print("No wind speed columns found. Skipping wind speed plots.")
        return
    if wind_speed_columns:
        print("\nColumns containing 'Wind Speed':")
        for col in wind_speed_columns:
            print(f"Plotting histogram for: {col}")
            plt.figure()
            dataframe[col].hist(bins=30)
            plt.title(f"Histogram of {col} for {start} - {end}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

def plot_wind_speed_time_series(dataframe, start, end):
    """Plot time series of wind speed-related columns."""
    wind_speed_columns = get_wind_speed_columns(dataframe)
    
    if not wind_speed_columns:
        print("No wind speed columns found. Skipping wind speed time series plot.")
        return

    # Ensure the index is datetime
    dataframe.index = pd.to_datetime(dataframe.index)

    for col in wind_speed_columns:
        print(f"\nPlotting time series for: {col} for {start} - {end}")
        
        # Plot the time series
        plt.figure(figsize=(10, 6))
        plt.plot(dataframe.index, dataframe[col], marker='o', linestyle='-', label=col)
        plt.title(f'Time Series of {col} for {start} - {end}')
        plt.xlabel('Date')
        plt.ylabel(f'{col}')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_relevant_columns(dataframe, start, end):
    """coordinates the plotting of power, wind speed, and specific-month data."""
   
    # Plot power histograms and monthly time series
    plot_power_histogram_and_monthly_mean_timeseries(dataframe, start, end)
    
    # Plot wind speed histograms
    plot_wind_speed_histogram(dataframe, start, end)


    plot_wind_speed_time_series(dataframe, start, end)

    
    # Plot specific month data based on user input
    plot_power_data_for_specific_month(dataframe)

def get_specific_month_year(dataframe):
    year = input("Which year would you like to have a time series? ")
    month = input("And for which month? Please indicate using 2 digits, e.g., '01' for January: ")
    return year, month


def plot_power_data_for_specific_month(dataframe):
    # Get year and month from user input
    year, month = get_specific_month_year(dataframe)

    power_columns = get_power_columns(dataframe)

    if not power_columns:
        print("No power-related columns found. Skipping specific month plot for power data.")
        return

    for col in power_columns:
        try:
            # Ensure the index is datetime
            dataframe.index = pd.to_datetime(dataframe.index)

            # Filter data for the specified year and month
            filtered_data = dataframe[(dataframe.index.year == int(year)) & (dataframe.index.month == int(month))]
            print(f"Filtered data for {col} in {month}/{year}: {filtered_data.shape[0]} records found.")

            # Check if filtered data is empty
            if filtered_data.empty:
                print(f"No data available for {col} in {month}/{year}.")
                continue

            # Plotting the filtered data
            plt.figure(figsize=(10, 6))
            plt.scatter(filtered_data.index, filtered_data[col], color='b', label=col)
            plt.title(f'Time Series of {col} Data for {month} {year}')
            plt.xlabel('Date')
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()

            input("Press Enter to continue...")  # This keeps the console open and allows you to read print statements.

        except KeyError:
            print(f"No data available for {col} in {month}/{year}")




def find_problematic_power_ranges(dataframe):

    if get_power_columns(dataframe) != None:
        return
    
    else:
        right_range = dataframe[(dataframe['Power (kW)'] > -20) & (dataframe['Power (kW)'] < 2050)][['Power (kW)', 'Wind speed (m/s)']]
        wrong_range = dataframe[~((dataframe['Power (kW)'] > -20) & (dataframe['Power (kW)'] < 2050))][['Power (kW)', 'Wind speed (m/s)']]
        
        return wrong_range


def explorative_analysis(dataframe):
    """Perform a full exploratory analysis of the dataframe."""

    start = dataframe.index.min().date()
    end = dataframe.index.max().date()
    print(f"Summary statistics for time frame: {start} - {end}")
    
    # Display the first few rows
    print("First 5 rows of the dataframe:")
    display(dataframe.head(5))

    # Display the last few rows
    print("\nLast 5 rows of the dataframe:")
    display(dataframe.tail())

    # Summary statistics
    print(f"\nSummary statistics of the dataframe for time frame: {start} - {end}")
    display(dataframe.describe())

    calculate_expected_entries(dataframe)
    
    # Shape of the dataframe
    print(f"\nShape of dataframe (rows, columns) for time frame: {start} - {end}")
    display(dataframe.shape)
    
    # Check for duplicates in the index
    check_duplicates(dataframe)
    
    # Find rows with NaN values
    find_nan_powers(dataframe)
    
    # Find problematic power ranges
    find_problematic_power_ranges(dataframe)
    
    # Plot relevant columns (power, wind speed, etc.)
    plot_relevant_columns(dataframe, start, end)


def find_nan_powers(dataframe):
    start = dataframe.index.min()
    end = dataframe.index.max()
  # Find rows with NaN values
    columns_to_check = ["Power (kW)", "onshore", "Wind speed (m/s)", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m"]

    for col in columns_to_check:
        if col in dataframe.columns:  # Check if the column exists in the dataframe
            nan_count = dataframe[col].isna().sum()  # Count NaN values in the column
            print(f"\nRows with NaN values for {col}:")
            nan_rows = dataframe[col].isna()  # Get rows where NaN is present
            print(f"In total there are: {nan_count} NaN values for time frame {start} - {end}")

            nan_entries = dataframe[nan_rows]  # Filter the rows where NaN is present
                
            # Minimum index where NaN is found for 'onshore'
            if col == "onshore" and nan_count > 0:
                first_nan_index = dataframe[nan_rows].index.min()  # Get the first index with NaN
                print(f"\nFirst NaN value in 'onshore' found at index: {first_nan_index}")
    display(nan_entries)


def check_duplicates(dataframe):
    """Check and display duplicate indices."""
    duplicate_indices = dataframe.index.duplicated()
    if duplicate_indices.any():
        print(f"Total duplicates: {duplicate_indices.sum()}")
        display(dataframe[duplicate_indices])
    else:
        print("No duplicates found in the index.")