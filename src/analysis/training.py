import numpy as np
import torch
from tabpfn import TabPFNRegressor  # Ensure this is correctly installed

import numpy as np
import torch
import pandas as pd
from tabpfn import TabPFNRegressor

def train_tabpfn(X_train, y_train, X_validation, y_validation, 
                 device: str = "auto", fit_mode: str = "low_memory", random_state: int = 42, ignore_pretraining_limits=False):
    """
    Trains a TabPFNRegressor model and makes predictions.
    
    Parameters:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target values.
        X_validation (pd.DataFrame or np.ndarray): Validation features.
        y_validation (pd.Series or np.ndarray): Validation target values.
        device (str): Device to use for training ('auto', 'cpu', 'cuda').
        fit_mode (str): Fit mode for TabPFNRegressor.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing:
            - DataFrame with logits, borders, quantiles, y_validation_torch
            - DataFrame with meta_info (splits, device, fit_mode, random_state)
    """

    # Train model
    model = TabPFNRegressor(device=device, fit_mode=fit_mode, random_state=random_state, ignore_pretraining_limits=ignore_pretraining_limits)
    model.fit(X_train, y_train)

    # Define custom quantiles (from 0.1 to 0.9)
    quantiles_custom = np.arange(0.1, 1, 0.1)

    # Predict with custom quantiles
    probs_val_q = model.predict(X_validation, output_type="full", quantiles=quantiles_custom)

    # Extract relevant data from the prediction
    logits_q = probs_val_q["logits"]  # logits (N, 5000)
    borders_q = probs_val_q["criterion"].borders  # borders (5001,)
    all_quantiles_q = np.array(probs_val_q["quantiles"])  # quantiles (N, 9)

    # Convert y_validation to tensor
    y_validation_q_torch = torch.tensor(y_validation.values, dtype=torch.float32)

    # Create a DataFrame containing logits, borders, quantiles, and y_validation_torch
    logits_df = pd.DataFrame(logits_q)  # (N, 5000)
    borders_df = pd.DataFrame(borders_q).T  # (1, 5001) - Transpose to make it a single row DataFrame
    quantiles_df = pd.DataFrame(all_quantiles_q)  # (N, 9)
    y_validation_df = pd.DataFrame(y_validation_q_torch.numpy(), columns=['y_validation_torch'])  # (N, 1)

    # Concatenate logits, borders, quantiles, and y_validation_torch into one DataFrame
    results_df = pd.concat([logits_df, borders_df, quantiles_df, y_validation_df], axis=1)

    # Prepare meta info DataFrame
    meta_info = pd.DataFrame({
        "splits": [f"Train: {len(X_train)} samples, Validation: {len(X_validation)} samples"],
        "device": [device],
        "fit_mode": [fit_mode],
        "random_state": [random_state],
        "ignore_pretraining_limits": ignore_pretraining_limits
    })

    # Return the DataFrame containing results and the DataFrame containing meta info
    return results_df, meta_info


