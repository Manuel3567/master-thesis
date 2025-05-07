from dataclasses import dataclass
from datetime import datetime
import os
import pickle
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

from analysis.datasets import load_entsoe
from analysis.splits import to_train_validation_test_data
from analysis.preprocessor import *
from analysis.experiment_mapper import *
    
@dataclass
class Experiment_Baseline:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    beta_0: None | float = None
    beta_1: None | np.ndarray = None
    sigma_sq: None | float = None
    intercept: bool = True

    def perform(self):
        model = LinearRegression(fit_intercept=self.intercept)
        model.fit(self.X_train, self.y_train)

        if self.intercept:
            self.beta_0 = model.intercept_  # Intercept (β₀) when fit_intercept=True
        else:
            self.beta_0 = 0.0  # Set beta_0 to 0 manually when fit_intercept=False

        self.beta_1 = model.coef_    # Coefficient for P_t-96 (β₁)

        # Calculate sigma^2 (variance of residuals)
        y_pred = model.predict(self.X_train)
        residuals = self.y_train - y_pred
        self.sigma_sq = (residuals ** 2).sum() / (len(self.X_train) - 2)

        return self
    
    def calculate_crps(self):
        """Calculates the Continuous Ranked Probability Score (CRPS)."""
        start_time = time.time()
        
        # Initialize CRPS statistics
        crps_values = []
        
        crps_mean = 0
        crps_min = 10000
        crps_max = 0
        counter = 0

        for i, y in enumerate(self.y_validation):  # Iterate over the validation set
            # Select the row corresponding to the i-th observation from X_validation
            mu = self.beta_0 + np.dot(self.X_validation.iloc[i, :], self.beta_1)  # Use iloc for row selection
            
            sigma = np.sqrt(self.sigma_sq)  # Predicted standard deviation (using variance)
            
            # CDF and PDF of standard normal distribution
            z = (y - mu) / sigma
            
            # Calculate PDF and CDF values for z
            pdf_z = norm.pdf(z)  # Standard normal PDF at z
            cdf_z = norm.cdf(z)  # Standard normal CDF at z

            # CRPS formula for normal distribution
            crps = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
            crps_values.append(crps)
            # Update CRPS statistics
            #crps_mean += crps
            #crps_min = min(crps_min, crps)
            #crps_max = max(crps_max, crps)

            counter += 1

            # Print progress every 5000 iterations
            if counter % 5000 == 0:
                end_time = time.time()
                elapsed = end_time - start_time
                print("Elapsed time:", elapsed)
                print("Counter:", counter)
                start_time = time.time()
        
        # Calculate the average CRPS value
        #crps_mean /= len(self.y_validation)
        crps_mean = np.mean(crps_values)
        #crps_min = np.min(crps_values)
        #crps_max = np.max(crps_values)
        #crps_median = np.median(crps_values)
        crps_quantiles = np.percentile(crps_values, np.array([5, 25, 75, 95]))
        # Return the CRPS statistics
        print("CRPS calculation finished")
        #return crps_mean, crps_median, crps_min, crps_max
        return crps_quantiles, crps_mean, crps_values

    def calculate_nll(self):
        """Calculates the Negative Log-Likelihood (NLL)."""
        start_time = time.time()
        nll_values = []
        counter = 0
        sigma = np.sqrt(self.sigma_sq)  # Predicted standard deviation (using variance)
        for i, y in enumerate(self.y_validation):  # Iterate over the validation set
            # Select the row corresponding to the i-th observation from X_validation
            mu = self.beta_0 + np.dot(self.X_validation.iloc[i, :], self.beta_1)  # Use iloc for row selection

            if counter == 0 or counter == 500 or counter == 6325:
                print(f"i = {i}, mu = {mu}, sigma = {sigma}")

            counter += 1
            # NLL formula for normal distribution
            nll = 0.5 * np.log(2 * np.pi * sigma**2) + ((y - mu)**2) / (2 * sigma**2)
            nll_values.append(nll)

            if counter % 5000 == 0:
                end_time = time.time()
                elapsed = end_time - start_time
                print("elapsed time", elapsed)
                start_time = time.time()
             
        nll_mean = np.mean(nll_values)
        nll_quantiles = np.percentile(nll_values, np.array([5, 25, 75, 95]))
        #nll_min = np.min(nll_values)
        #nll_max = np.max(nll_values)
        #nll_median = np.median(nll_values)
        #return nll_mean, nll_median, nll_min, nll_max
        return nll_quantiles, nll_mean, nll_values
        
class ExperimentStorage:
    def __init__(self, file_path):
        self.file_path = file_path

    def save(self, experiment: Experiment_Baseline):
        """Save the ExperimentTracker object to a file."""

        directory = os.path.dirname(self.file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.file_path, "wb") as f:
            pickle.dump(experiment, f)

    def load(self):
        """Load the Experiment object from a file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                return pickle.load(f)


from datetime import datetime
import time
import warnings

def run_baseline_model(experiment_ids, storage_path="C:/Users/Minu/Documents", fit_intercept=False):
    """
    Run multiple experiments with different configurations.
    
    Parameters:
    - experiment_configs (list of dict): Each dictionary should contain:
        - selected_features
        - train_start
        - train_end
        - val_start
        - val_end
        - random_state (optional)
    """

    baseline_dir = os.path.join(storage_path, "baseline")
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)


    if isinstance(experiment_ids, int):
        experiment_ids = [experiment_ids]

    experiment_mapper = ExperimentMapper()

    for experiment_id in experiment_ids:

        config_list = experiment_mapper.map_id_to_config(experiment_id)

        for config in config_list:

            print(f"Running experiment {experiment_id}...")
            start_time = time.time()

            # Extract experiment-specific parameters
            selected_features = config["selected_features"]
            train_start = config["train_start"]
            train_end = config["train_end"]
            val_start = config["val_start"]
            val_end = config["val_end"]
            random_state = config.get("random_state", 42)  # Default random state if not provided

            # Preprocess the data
            print("- Preprocessing data...")
            preprocessor = DataPreprocessor()
            preprocessor.load_data()
            preprocessor.transform_power()
            preprocessor.add_interval_index()
            preprocessor.add_lagged_features()
            preprocessor.prepare_features(selected_features)
            print("- Splitting data into train, validation, test...")
            preprocessor.split_data(train_start, train_end, val_start, val_end)
            train_X, train_y, validation_X, validation_y, test_X, test_y = preprocessor.get_processed_data()
            display(train_X.head(3))


            print("- Running model training")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)

                experiment = Experiment_Baseline(X_train=train_X, y_train=train_y, X_validation=validation_X, y_validation=validation_y, intercept=fit_intercept)
                experiment = experiment.perform()

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"⏱️ Experiment {experiment_id} completed in {elapsed:.2f} seconds")

            print("- Saving experiment results...")
            experiment_filename = os.path.join(baseline_dir, f"experiment_{experiment_id}.pkl")
            storage = ExperimentStorage(experiment_filename)
            storage.save(experiment)
            print(f"Experiment saved to: {experiment_filename}")

    print("All experiments completed and saved.")


import pandas as pd


def calculate_scores_baseline(experiment_id, storage_path="experiments/", all_scores=False):
    
    storage_path = os.path.join(storage_path, "baseline")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    experiment_filename = f"{storage_path}/experiment_{experiment_id}.pkl"
    storage = ExperimentStorage(experiment_filename)
    experiment = storage.load()

    #crps_mean, crps_median, crps_min, crps_max = experiment.calculate_crps()
    #mean_nll, median_nll, min_nll, max_nll  = experiment.calculate_nll()
    #mean_nll, nll_quantiles  = experiment.calculate_nll()
    #crps_mean, crps_quantiles = experiment.calculate_crps()

    nll_quantiles, mean_nll, nll_scores = experiment.calculate_nll()
    crps_quantiles, mean_crps, crps_scores = experiment.calculate_crps()

    #scores = {
    #    'Metric': ['nll', 'crps'],
    #    'Mean': [mean_nll, crps_mean],
    #    '5%': [nll_quantiles[0], crps_quantiles[0]],
    #    '25%': [nll_quantiles[1], crps_quantiles[1]],
    #    '75%': [nll_quantiles[2], crps_quantiles[2]],
    #    '95%': [nll_quantiles[3], crps_quantiles[3]]
    #}

    #df = pd.DataFrame(scores)

    scores = {
        'Metric': ['nll', 'crps'],
        'q5': [nll_quantiles[0], crps_quantiles[0]],
        'q25': [nll_quantiles[1], crps_quantiles[1]],
        'q75': [nll_quantiles[2], crps_quantiles[2]],
        'q95': [nll_quantiles[3], crps_quantiles[3]],
        'Mean': [mean_nll, mean_crps],
    }

    # Stack the scores first
    score_columns = {f'score_{i}': [nll_scores[i], crps_scores[i]] for i in range(len(nll_scores))}

    # Merge with original scores
    scores.update(score_columns)

    df = pd.DataFrame(scores)

    file_name = f"{storage_path}/quantiles/experiment_results_{experiment_id}.pkl"
    storage = ExperimentStorage(file_name)
    storage.save(df)

    return df