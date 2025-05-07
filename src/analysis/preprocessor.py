import numpy as np

from analysis.datasets import load_entsoe
from analysis.splits import to_train_validation_test_data


class DataPreprocessor:
    def __init__(self, target_column="power"):
        self.target_column = target_column
        self.max_power_value_rounded = None

    def load_data(self):
        """Loads the dataset."""
        self.df = load_entsoe()
        return self

    def transform_power(self, epsilon=1e-3):
        """Scales the power data using log transformation."""
        max_power_value = self.df[self.target_column].max()
        self.max_power_value_rounded = np.ceil(max_power_value / 1000) * 1000
        self.df[self.target_column] = np.log(self.df[self.target_column] / self.max_power_value_rounded + epsilon)
        return self

    def add_interval_index(self):
        """Creates an interval index feature based on time."""
        self.df['interval_index'] = ((self.df.index.hour * 60 + self.df.index.minute) // 15) + 1
        return self

    def add_lagged_features(self, lag=96):
        """Adds lagged power feature."""
        self.df[f'{self.target_column}_t-{lag}'] = self.df[self.target_column].shift(lag)
        self.df.dropna(inplace=True)
        return self

    def prepare_features(self, selected_features):
        """Selects only the specified features from the DataFrame."""
        selected_features.append(self.target_column)
        self.df = self.df[[feature for feature in selected_features if feature in self.df.columns]]
        return self

    def split_data(self, train_start, train_end, val_start, val_end):
        """Splits dataset into train, validation, and test sets."""
        self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y = to_train_validation_test_data(
            self.df, train_start, train_end, val_start, val_end
        )
        return self

    def get_processed_data(self):
        """Returns processed train, validation, and test sets."""
        return self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y 