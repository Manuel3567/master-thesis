import pandas as pd


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