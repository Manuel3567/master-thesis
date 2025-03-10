import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis.datasets import load_entsoe
from analysis.splits import to_train_validation_test_data
from analysis.transformations import scale_power_data
from tabpfn import TabPFNRegressor
from analysis.transformations import add_interval_index, add_lagged_features
from torchinfo import summary

entsoe = load_entsoe("data/")
entsoe = scale_power_data(entsoe)
entsoe = add_lagged_features(entsoe)
entsoe = add_interval_index(entsoe)
entsoe.dropna(inplace=True)
train, validation, test = to_train_validation_test_data(entsoe, "2016-06-30 23:45:00", "2016-09-30 23:45:00")

feature_columns = ['power_t-96']
#feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean']
#feature_columns = ['power_t-96', 'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
#                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
#                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
#                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
target_column='power'

X_train, y_train = train[feature_columns], train[target_column]
X_validation, y_validation = validation[feature_columns], validation[target_column]

n_train = 10
n_pred = 10
#model = TabPFNRegressor(device='auto', ignore_pretraining_limits=True, fit_mode='low_memory', random_state=42)
model = TabPFNRegressor(device='auto', fit_mode='low_memory', random_state=42, n_jobs=1, ignore_pretraining_limits=True)
X_train, X_validation, y_train = X_train.head(n_train), X_validation.head(n_pred), y_train.head(n_train)
model.fit(X_train, y_train)
quantiles_custom = np.arange(0.1, 1, 0.1)

probs_val = model.predict(X_validation, output_type="full", quantiles=quantiles_custom)
all_quantiles = np.array(probs_val["quantiles"])