import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis.datasets import load_entsoe
from analysis.transformations import minute_to_daily
from analysis.splits import to_train_validation_test_data
from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.scores import LogScore, CRPScore
from ngboost.distns.normal import NormalCRPScore


def get_columns_by_time(df, time: str):
    time = time.replace(':', '_')
    #columns = [c.removesuffix(f"_{time}") for c in df.columns if c.endswith(time)]
    columns = [c for c in df.columns if c.endswith(time)]

    selection = df.loc[:, columns]
    selection.columns = [c.removesuffix(f"_{time}") for c in selection.columns]
    return selection

def delay(df, delays: int | list[int], columns: None | str | list[str] = None):

    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
        
    if isinstance(df, pd.Series):
        df = df.to_frame()

    dfs = [df]

    if isinstance(delays, int):
        delays = range(1, delays + 1)
    for t in delays:
        delayed_df = df.loc[:, columns].shift(t)
        delayed_df.columns = [f"{c}_t-{t}" for c in delayed_df.columns]
        dfs.append(delayed_df)
    vstacked_df = pd.concat(reversed(dfs), axis=1).dropna()
    return vstacked_df


def prepare_and_select_features(time_interval, columns_to_include):

    df = load_entsoe()  # Load the dataframe containing aggregated power and wind speed at 10 locations and at geographic mean

    # Rename and drop "offshore" column
    df = df.rename(columns={"onshore": "power"})
    df = df.drop(columns=["offshore"])

    # Resample to daily data (each column represents a 15 minute time interval)
    df_daily = minute_to_daily(df)

    # Select by time interval
    df_daily_time = get_columns_by_time(df_daily, time_interval)

    # Delay the power column for lagged features
    df_daily_time_delay = delay(df_daily_time, delays=1, columns="power")  # delay the "power" column

    selected_columns = []


    if 'only_power' in columns_to_include:
        selected_columns = ['power_t-1']  # Always include power_t-1 as a baseline
    
    # Add additional columns based on the input argument 'columns_to_include'
    if 'all_wind_speeds' in columns_to_include:
        selected_columns = ['power_t-1']  # Always include power_t-1 as a baseline
        selected_columns.extend([col for col in df_daily_time_delay.columns if col.startswith('ws')])
        

    if 'only_sin_cos' in columns_to_include:
        selected_columns = ['power_t-1']  # Always include power_t-1 as a baseline
        # Add sine and cosine transformation for annual periodicity
        selected_columns.extend(['sin_day', 'cos_day'])
        
        df_daily_time_delay['day_of_year'] = df_daily_time_delay.index.dayofyear
        df_daily_time_delay['sin_day'] = np.sin(2 * np.pi * df_daily_time_delay['day_of_year'] / 365)
        df_daily_time_delay['cos_day'] = np.cos(2 * np.pi * df_daily_time_delay['day_of_year'] / 365)
        df_daily_time_delay = df_daily_time_delay.drop(columns="day_of_year")
    
    if 'only_mean_wind' in columns_to_include:
        selected_columns = ['power_t-1']  # Always include power_t-1 as a baseline
        # Add mean wind speed columns
        selected_columns.extend(['ws_100m_loc_mean', 'ws_10m_loc_mean'])

    # If no specific selection criteria are met, use the full dataframe (excluding 'power' column)


    if "max" in columns_to_include:

        selected_columns.extend([col for col in df_daily_time_delay.columns if col != "power"])
        #df_daily_time_delay = df_daily_time_delay.drop(columns=['power'])  # Exclude 'power' column
        df_daily_time_delay['day_of_year'] = df_daily_time_delay.index.dayofyear
        df_daily_time_delay['sin_day'] = np.sin(2 * np.pi * df_daily_time_delay['day_of_year'] / 365)
        df_daily_time_delay['cos_day'] = np.cos(2 * np.pi * df_daily_time_delay['day_of_year'] / 365)

        df_daily_time_delay = df_daily_time_delay.drop(columns="day_of_year")
        selected_columns.extend(['sin_day', 'cos_day'])



    
    # Ensure that only the columns that exist in the dataframe are included
    selected_columns = [col for col in selected_columns if col in df_daily_time_delay.columns]

    # Split the data into training and validation sets
    train, validation, test = to_train_validation_test_data(df_daily_time_delay, "2022-12-31", "2023-12-31")
    
    # Filter the training and validation sets to only the selected columns
    X_train = train[selected_columns]
    X_validation = validation[selected_columns]

    # Separate target variable 'power' for training and validation
    y_train = train['power']
    y_validation = validation['power']

    return X_train, y_train, X_validation, y_validation

def model_NGBoost(time, columns_to_include, epochs, lr, distribution, natural_gradient=True):

    random_seed = 42
    X_train, y_train, X_validation, y_validation = prepare_and_select_features(time, columns_to_include)
    
    model = NGBRegressor(Dist=distribution, Score=CRPScore, n_estimators=epochs, learning_rate=lr, verbose_eval=True, natural_gradient=natural_gradient, random_state=random_seed)

    model.fit(X_train, y_train, X_val=X_validation, Y_val=y_validation)

    # Predict on training data
    y_train_pred = model.predict(X_train)
    y_train_dists = model.pred_dist(X_train)


    y_val_pred = model.predict(X_validation)
    y_val_dists = model.pred_dist(X_validation)

    # Calculate Mean Squared Error (MSE) for training and validation
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_validation, y_val_pred)

    # Print the results
    print(f"Training features: {[c for c in X_train.columns]} -> {y_train.name}")
    print(f"Train MSE:\t {train_mse}")
    print(f"Validation MSE:\t {val_mse}")

    return y_train_pred, y_train_dists, y_val_pred, y_val_dists, X_train, y_train, X_validation, y_validation


def visualize_predictions_with_uncertainty(y_validation, y_val_pred, y_val_dists):
    start = 0
    end = 90
    x = y_validation.index[start:end]
    y = y_validation[start:end]
    mu = y_val_pred[start:end]
    print(mu)
    sigma = [y.scale for y in y_val_dists][start:end]  # Normal sigma
    lower_bound = mu - sigma  # Normal uncertainty band
    upper_bound = mu + sigma  # Normal uncertainty band

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, label="True Values (y)", color="green", alpha=0.5)
    plt.plot(x, mu, label="Predicted Values (mu)", color="red", alpha=0.5)
    plt.fill_between(x, lower_bound, upper_bound, color="orange", alpha=0.3, label="Uncertainty Band (mu ± sigma)")

    # Customize the plot
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"True Values, Predictions, and Uncertainty Band")
    plt.legend()
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()
