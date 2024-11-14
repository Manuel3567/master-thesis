import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def minute_to_daily_50Hertz(df):
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

def plot_weibull_for_one_interval(Dataframe, time_interval):


    # Step 1: Get the data from the DataFrame

    ## interessanter Wert 07_30
    #data = minute_to_daily_50Hertz(reshaped)['onshore_21_30']
    data = Dataframe[f'onshore_{time_interval}']

    # Step 2: Clean the data by removing NaN and infinite values
    data_clean = data[np.isfinite(data)]

    # Step 3: Plot the histogram of the cleaned data
    plt.figure(figsize=(10, 6))
    plt.hist(data_clean, bins=30, density=True, alpha=0.6, color='blue', label='Data Histogram')

    # Step 4: Fit the Weibull distribution to the cleaned data
    shape_param, loc, scale_param = weibull_min.fit(data_clean, floc=0)  # floc=0 fixes the location parameter

    # Step 5: Generate x values for the Weibull PDF within the specified range
    x = np.linspace(0, 16000, 100)  # Specify range from 0 to 16000

    # Step 6: Calculate the Weibull PDF
    weibull_pdf = weibull_min.pdf(x, shape_param, loc=loc, scale=scale_param)

    # Step 7: Plot the Weibull distribution
    plt.plot(x, weibull_pdf, 'r-', lw=2, label='Weibull Fit (shape={:.2f}, scale={:.2f})'.format(shape_param, scale_param))

    # Finalize the plot
    plt.title('Histogram and Weibull Distribution Fit')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.xlim(0, 16000)  # Set x-axis limits from 0 to 16000
    plt.legend()
    plt.grid()
    plt.show()