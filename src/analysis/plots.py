import numpy as np
import matplotlib as plt
from scipy.stats import weibull_min
from scipy.special import gamma
import scipy.stats as stats
import seaborn as sns

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