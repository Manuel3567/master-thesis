import pandas as pd

def to_train_validation_test_data(df: pd.DataFrame, train_start_date: str, train_end_date: str, validation_start_date: str, validation_end_date: str):
    """
    Splits the dataset into train, validation, and test sets based on specified start and end dates,
    ensuring that all time intervals within the end dates are included.

    Parameters:
        df (pd.DataFrame): The full dataset with a DateTime index.
        train_start_date (str): The start date for the training set in 'YYYY-MM-DD' format.
        train_end_date (str): The end date for the training set in 'YYYY-MM-DD' format.
        validation_start_date (str): The start date for the validation set in 'YYYY-MM-DD' format.
        validation_end_date (str): The end date for the validation set in 'YYYY-MM-DD' format.

    Returns:
        tuple: (train_data, validation_data, test_data)
    """
    # Ensure date columns are in datetime format
    df = df.sort_index()
    
    # Adjust end dates to include all time intervals of that day
    train_end_datetime = pd.to_datetime(train_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) #sets train_end_datetime to last day 23:59:59
    validation_end_datetime = pd.to_datetime(validation_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) #sets validation_end_datetime to last day 23:59:59
    
    # Split the data
    train = df.loc[(df.index >= train_start_date) & (df.index <= train_end_datetime), :].copy()
    validation = df.loc[(df.index >= validation_start_date) & (df.index <= validation_end_datetime), :].copy()
    test = df.loc[df.index > validation_end_datetime, :].copy()

    # Split the data into X (features) and y (target variable)
    train_X = train.drop(columns=['power'])
    train_y = train['power']
    
    validation_X = validation.drop(columns=['power'])
    validation_y = validation['power']
    
    test_X = test.drop(columns=['power'])
    test_y = test['power']


    # Calculate sizes
    n = len(df)
    n_train, n_val, n_test = len(train), len(validation), len(test)

    # Logging the results
    print(f"# of training observations: {n_train} | {(n_train / n * 100):.2f}%")
    print(f"# of validation observations: {n_val} | {(n_val / n * 100):.2f}%")
    print(f"# of test observations: {n_test} | {(n_test / n * 100):.2f}%")

    return train_X, train_y, validation_X, validation_y, test_X, test_y