import pandas as pd
import matplotlib.pyplot as plt

def expected_no_entries(dataframe):
    min_date = dataframe.index.min()
    max_date = dataframe.index.max()
    time_interval = dataframe.index[1] - dataframe.index[0]

    expected_entries = (max_date - min_date) // time_interval + 1
    print(f"Expected number of entries: {expected_entries} ({min_date} - {max_date}) ")
    print(f"Actual number of entries: {dataframe.shape[0]}")


def get_power_columns(dataframe):
    power_columns = [col for col in dataframe.columns if 'Power (kW)' in col or col == 'onshore']

    return power_columns

def get_wind_speed_columns(dataframe):
    wind_speed_columns = [col for col in dataframe.columns if 'wind_speed (m/s)' in col.lower()]

    return wind_speed_columns



def plot_power_histogram(dataframe, start, end):
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
    wind_speed_columns = get_wind_speed_columns(dataframe)
    
    if not wind_speed_columns:
        print("No wind speed columns found. Skipping wind speed plots.")
        return
    if wind_speed_columns:
        print("\nColumns containing 'Wind Speed':")
        for col in wind_speed_columns:
            print(f"Plotting histogram for: {col}")
            plt.figure()
            dataframe[col].hist()
            plt.title(f"Histogram of {col} for {start} - {end}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()


def plot_relevant_columns(dataframe, start, end):
    """Main function to coordinate the plotting of power, wind speed, and specific-month data."""
   
    # Plot power histograms and monthly time series
    plot_power_histogram(dataframe, start, end)
    
    # Plot wind speed histograms
    plot_wind_speed_histogram(dataframe, start, end)
    
    # Plot specific month data based on user input
    plot_specific_power_month(dataframe)

def get_specific_month_year(dataframe):
    year = input("Which year would you like to have a time series? ")
    month = input("And for which month? Please indicate using 2 digits, e.g., '01' for January: ")
    return year, month


def plot_specific_power_month(dataframe):
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




def explorative_analysis(dataframe):
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

    expected_no_entries(dataframe)
    
    # Shape of the dataframe
    print(f"\nShape of dataframe (rows, columns) for time frame: {start} - {end}")
    display(dataframe.shape)
    
    # Check for duplicates in the index
    print(f"\nChecking for duplicates in the index for time frame: {start} - {end}")
    duplicate_indices = dataframe.index.duplicated()
    if duplicate_indices.any():
        display(dataframe[duplicate_indices])
        print(f"Total duplicates in index for time frame {start} - {end}: {duplicate_indices.sum()}")
    else:
        print("Index column has no duplicates.")
    
    # Find rows with NaN values
    nan_count = dataframe.isna().any(axis=1).sum()
    print("\nRows with NaN values:")
    nan_rows = dataframe[dataframe.isna().any(axis=1)]
    print(f"In total there are: {nan_count} for time frame {start} - {end}")
    display(nan_rows)
    
    # Minimum index where NaN is found
    if not nan_rows.empty:
        print("\nFirst index with NaN values:")
        first_nan_index = nan_rows.index.min()
        print(first_nan_index)
        print("\nLast index with NaN values:")
        last_nan_index = nan_rows.index.max()
        print(last_nan_index)
    else:
        print("\nNo NaN values found.")
        first_nan_index = None
    
    check_problematic_power_ranges(dataframe)
    
    plot_relevant_columns(dataframe, start, end)



def check_problematic_power_ranges(dataframe):
    right_range = dataframe[(dataframe['Power (kW)'] > -20) & (dataframe['Power (kW)'] < 2050)][['Power (kW)', 'Wind speed (m/s)']]
    wrong_range = dataframe[~((dataframe['Power (kW)'] > -20) & (dataframe['Power (kW)'] < 2050))][['Power (kW)', 'Wind speed (m/s)']]
    
    return wrong_range