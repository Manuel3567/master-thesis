import pandas as pd
import matplotlib.pyplot as plt


def expected_no_entries(dataframe):
        
    time_interval = dataframe.index[1] - dataframe.index[0]

    time_interval_minutes = time_interval.total_seconds() / 60

    
    # Print the time interval in minutes
    print(f"Calculated time interval: {time_interval_minutes} minutes")
    
    # Get the start and end dates from the dataframe
    start_date = dataframe.index.min()
    end_date = dataframe.index.max()
    
    total_duration = end_date - start_date
    total_days = total_duration.days + 1  # Include both start and end day
    
    # Calculate the expected number of entries
    expected_entries = total_days * (24 * 60 // time_interval_minutes)
    
    # Get the actual number of entries
    actual_entries = dataframe.shape[0]
    
    print(f"Start date: {start_date}, End date: {end_date}, Total days (inclusive): {total_days}")
    print(f"Time interval: {time_interval_minutes} minutes, Expected entries per day: {24 * 60 // time_interval_minutes}")
    print(f"Expected entries based on actual date range: {expected_entries}")
    print(f"Actual entries: {actual_entries}")
    
    # Output the result
    match = actual_entries == expected_entries
    print(f"Do actual entries match expected entries? {match}")

def explorative_analysis(dataframe):
    # Display the first few rows
    print("First 5 rows of the dataframe:")
    display(dataframe.head(5))
    
    # Display the last few rows
    print("\nLast 5 rows of the dataframe:")
    display(dataframe.tail())
    
    # Summary statistics
    print("\nSummary statistics of the dataframe:")
    display(dataframe.describe())
    
    # Shape of the dataframe
    print("\nShape of the dataframe (rows, columns):")
    display(dataframe.shape)
    
    # Check for duplicates in the index
    print("\nChecking for duplicates in the index:")
    duplicate_indices = dataframe.index.duplicated()
    if duplicate_indices.any():
        display(dataframe[duplicate_indices])
        print(f"Total duplicates in index: {duplicate_indices.sum()}")
    else:
        print("Index column has no duplicates.")
    
    # Find rows with NaN values
    print("\nRows with NaN values:")
    nan_rows = dataframe[dataframe.isna().any(axis=1)]
    display(nan_rows)
    
    # Minimum index where NaN is found
    if not nan_rows.empty:
        print("\nFirst index with NaN values:")
        first_nan_index = nan_rows.index.min()
        print(first_nan_index)
    else:
        print("\nNo NaN values found.")
        first_nan_index = None
    
    # Check if any column contains the string "Power" or is named 'onshore'
    relevant_columns = [col for col in dataframe.columns if 'Power (kW)' in col or col == 'onshore']
    
    if relevant_columns:
        print("\nColumns containing 'Power (kW)' or named 'onshore':")
        for col in relevant_columns:
            print(f"Plotting histogram for: {col}")
            plt.figure()
            dataframe[col].hist()
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
            print(f"\nPlotting monthly time series of {col} data:")
            monthly_data = dataframe.resample('M').mean()  # Resample by month, taking the mean
            plt.figure(figsize=(10, 6))
            plt.plot(monthly_data.index, monthly_data[col], marker='o', linestyle='-', color='b', label=col)
            plt.title(f'Monthly Time Series of {col} Data')
            plt.xlabel('Month')
            plt.ylabel(f'{col}')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()
            # Filter the data for January 2024 (or any month you want)
            filtered_data = dataframe.loc['2019-01'] # warning hard coded month

            # Plot the filtered data
            plt.figure(figsize=(10, 6))
            plt.scatter(filtered_data.index, filtered_data[col], color='b', label=col)
            plt.title(f'Time Series of {col} Data for January 2019')
            plt.xlabel('Date')
            plt.ylabel(f'{col}')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        print("\n'Onshore' column not found. Skipping monthly plot.")
    
    # Resample the data by month and plot the 'onshore' column if available
    
