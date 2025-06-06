import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from analysis.TabPFN import *
from analysis.baseline_model import *
from analysis.ngboost import *
from analysis.experiment_mapper import *
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


    
def analysis_2_tabpfn(file_path="C:/Users/Minu/Documents"):
    """
    Creates performance matrices for each of the four training quarters (2022 Q1–Q4)
    across four validation quarters (2023 Q1–Q4), for the same feature model (Power_t-96, two mean wind speeds).

    For each predefined train-validation quarter combination, loads the corresponding
    pickle file, extracts mean CRPS and NLL metrics, and populates two matrices
    (CRPS and NLL). Each matrix has training quarters as rows and validation quarters
    as columns, with an additional column showing the row-wise mean across validation quarters.

    Returns:
        crps_matrix (pd.DataFrame): 4x5 DataFrame showing CRPS values for each train-val pair.
        nll_matrix (pd.DataFrame): 4x5 DataFrame showing NLL values for each train-val pair.
    """

    # Map from ID to (train, val) quarters
    id_to_split = {
        1: ('22Q1', '23Q1'),
        2: ('22Q2', '23Q2'),
        3: ('22Q3', '23Q3'),
        4: ('22Q4', '23Q4'),
        5: ('22Q4', '23Q1'),
        6: ('22Q4', '23Q2'),
        7: ('22Q4', '23Q3'),
        8: ('22Q4', '23H1'),
        9: ('22Q4', '23H2'),
        10: ('22Q4', '23H1'),
        11: ('22Q4', '23H2'),
        12: ('22Q4', '23H1'),
        13: ('22Q4', '23H2'),
        14: ('22Q1', '23Q2'),
        15: ('22Q3', '23Q2'),
        16: ('22Q1', '23Q4'),
        17: ('22Q1', '23Q3'),
        18: ('22Q2', '23Q1'),
        19: ('22Q2', '23Q3'),
        20: ('22Q2', '23Q4'),
        21: ('22Q3', '23Q1'),
        22: ('22Q3', '23Q4'),
    }

    file_path = os.path.join(file_path, "tabpfn")
    file_path = file_path.replace("\\", "/")

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Target quarters
    train_quarters = ['22Q1', '22Q2', '22Q3', '22Q4']
    val_quarters = ['23Q1', '23Q2', '23Q3', '23Q4']

    # Initialize matrices with NaNs
    crps_matrix = pd.DataFrame(np.nan, index=train_quarters, columns=val_quarters)
    nll_matrix = pd.DataFrame(np.nan, index=train_quarters, columns=val_quarters)

    # Load each pickle and populate matrices
    for i in range(1, 23):
        train_time, val_time = id_to_split[i]
        
        # Only consider entries where both are Q-quarters
        if train_time in train_quarters and val_time in val_quarters:
            pkl_file_path = os.path.join(file_path, f"experiment_results_{i}.pkl")
            pkl_file_path = pkl_file_path.replace("\\", "/")  # Ensure correct path format

            #with open(f'C:/Users/Minu/Documents/TabPFN/experiments/experiment_results_{i}.pkl', 'rb') as f:
            with open(pkl_file_path, 'rb') as f:
                df = pickle.load(f)

            # Assume row 0 is NLL and row 1 is CRPS
            nll_mean = df.loc[df['Metric'].str.startswith('nll'), 'Mean'].values[0]
            crps_mean = df.loc[df['Metric'].str.startswith('crps'), 'Mean'].values[0]

            # Fill the matrix
            crps_matrix.loc[train_time, val_time] = crps_mean
            nll_matrix.loc[train_time, val_time] = nll_mean

    # Add mean column
    crps_matrix['mean'] = crps_matrix.mean(axis=1)
    nll_matrix['mean'] = nll_matrix.mean(axis=1)

    # Print results
    print("CRPS Matrix: CRPS values obtained by TabPFN for different combinations of training and validation date ranges. \nThe feature model is always Power_t-96, and the two mean wind speeds")
    display(crps_matrix.round(3))
    print("NLL Matrix:NLL values obtained by TabPFN for different combinations of training and validation date ranges. \nThe feature model is always Power_t-96, and the two mean wind speeds")
    display(nll_matrix.round(3))

    return crps_matrix, nll_matrix



def final_table_q4(file_path="C:/Users/Minu/Documents"):
    # Load all results
    ngboost_matrix_q4 = analyze_3_ngboost(f"{file_path}", split="quarter")
    ngboost_matrix_ft = analyze_3_ngboost(f"{file_path}", split="full")

    tabpfn_result_q4 = analyze_3("tabpfn", file_path)
    baseline_result_q4 = analyze_3("baseline", file_path)

    # Prepare CRPS
    ngboost_crps_q4 = ngboost_matrix_q4.set_index('feature_abbr')[['CRPS_gaussian_mean']].rename(columns={'CRPS_gaussian_mean': 'NGBoost (Q4 training)'})
    ngboost_crps_ft = ngboost_matrix_ft.set_index('feature_abbr')[['CRPS_gaussian_mean']].rename(columns={'CRPS_gaussian_mean': 'NGBoost (full training)'})
    
    tabpfn_crps_q4 = tabpfn_result_q4[['CRPS']].rename(columns={'CRPS': 'TabPFN (Q4 training)'})
    baseline_crps_q4 = baseline_result_q4[['CRPS']].rename(columns={'CRPS': 'Baseline (Q4 training)'})
    baseline_crps_ft = baseline_result_q4[['CRPS_2016']].rename(columns={'CRPS_2016': 'Baseline (full training)'})

    crps_matrix = pd.concat([baseline_crps_q4, baseline_crps_ft, ngboost_crps_q4, ngboost_crps_ft, tabpfn_crps_q4], axis=1)

    # Prepare NLL
    ngboost_nll_q4 = ngboost_matrix_q4.set_index('feature_abbr')[['NLL_mean']].rename(columns={'NLL_mean': 'NGBoost (Q4 training)'})
    ngboost_nll_ft = ngboost_matrix_ft.set_index('feature_abbr')[['NLL_mean']].rename(columns={'NLL_mean': 'NGBoost (full training)'})

    tabpfn_nll_q4 = tabpfn_result_q4[['NLL']].rename(columns={'NLL': 'TabPFN (Q4 training)'})
    baseline_nll_q4 = baseline_result_q4[['NLL']].rename(columns={'NLL': 'Baseline (Q4 training)'})
    baseline_nll_ft = baseline_result_q4[['NLL_2016']].rename(columns={'NLL_2016': 'Baseline (full training)'})

    nll_matrix = pd.concat([baseline_nll_q4, baseline_nll_ft, ngboost_nll_q4, ngboost_nll_ft, tabpfn_nll_q4], axis=1)

    print("Comparative Continuous Ranked Probability Score (CRPS) across different forecasting models and feature representations. \nResults are shown for models trained on Q4 data and full-year (2016) data.")
    display(crps_matrix.round(3))

    print("Comparative Negative Log-Likelihood (NLL) performance across different forecasting models and feature representations. \nColumns reflect models trained on Q4 data versus full-year (2016) training data.")
    display(nll_matrix.round(3))

    return crps_matrix.round(3), nll_matrix.round(3)

import pandas as pd

def analyze_3_ngboost(filepath="C:/Users/Minu/Documents", split="quarter"):
    
    if split == "quarter":
        filepath = f"{filepath}/ngboost/q4_train/Merged_sheet.xlsx"
    else:
        filepath = f"{filepath}/ngboost/full_year/Merged_sheet.xlsx"

    #filepath="C:/Users/Minu/Documents/NGboost/q4_train/Merged_sheet.xlsx
    
    # Read the Excel file
    df = pd.read_excel(filepath)

    # Ensure 'loss_function' is treated as string
    df['loss_function'] = df['loss_function'].astype(str)

    # List of feature_abbr values we're interested in
    target_features = {
        "p, ws_mean, ws_10_loc",
        "p, ws_mean, ws_10_loc, t_index",
        "p, ws_mean",
        "p, ws_10_loc"
    }

    # Filter rows
    filtered = df[
        df['loss_function'].str.contains('LogScore', case=False, na=False) &
        df['feature_abbr'].isin(target_features)
    ]

    # Rename the feature_abbr values
    rename_map = {
        "p, ws_10_loc": "Power, 10 wind speeds",
        "p, ws_mean": "Power, mean wind speed",
        "p, ws_mean, ws_10_loc, t_index": "Power, all wind speeds, time",
        "p, ws_mean, ws_10_loc": "Power, all wind speeds"
    }
    filtered['feature_abbr'] = filtered['feature_abbr'].map(rename_map)

    # Extract only the required columns and include renamed feature_abbr
    result = filtered[['feature_abbr', 'CRPS_gaussian_mean', 'NLL_mean']]

     # Custom sort
    feature_order = [
        "Power, mean wind speed",
        "Power, 10 wind speeds",
        "Power, all wind speeds",
        "Power, all wind speeds, time"
    ]
    result['feature_abbr'] = pd.Categorical(result['feature_abbr'], categories=feature_order, ordered=True)
    result = result.sort_values('feature_abbr').reset_index(drop=True)
    result = result.round(3)

    return result

import os
import glob
import re
import pickle
import pandas as pd
from analysis.TabPFN import ExperimentMapper

def analyze_3(method="tabpfn", file_path="C:/Users/Minu/Documents"):

    if method == "tabpfn":
        dir = f"{file_path}/tabpfn/"
        nll_name = "nll_5000"
        crps_name = "crps_5000"
        compute_2016 = False
    else:
        "C:/Users/Minu/Documents/results"
        dir = f"{file_path}/baseline/"
        nll_name = "nll"
        crps_name = "crps"
        compute_2016 = True

    # Step 1: Load experiment files

    def categorize_ID_to_feature_group():
        files = glob.glob(os.path.join(dir, 'experiment_results_*'))
        ids = [int(re.search(r'\d+', s).group()) for s in files if re.search(r'\d+', s)]

        feature_groups = {
            "power, all ws": [],
            "power, all ws, time bin": [],
            'power, mean ws': [],
            "power, ws at 10 loc": []
        }

        feature_groups_2016 = {
            "power, all ws": [],
            "power, all ws, time bin": [],
            'power, mean ws': [],
            "power, ws at 10 loc": []
        } if compute_2016 else None

        for id in ids:
            config = ExperimentMapper.map_id_to_config(id)
            dates = ExperimentMapper.extract_date_abbreviations_from_config(config)
            dates_split = dates.strip().split(" / ")
            if not dates_split or len(dates_split) < 2:
                continue

            feature = ExperimentMapper.get_feature_string_from_selected_features(config).strip()
            if feature not in feature_groups:
                continue

            # Add to overall group if it's Q4 (original behavior)
            if dates.startswith("Q4"):
                feature_groups[feature].append(id)

            # Add to 2016 group if any date starts with 2016
            if dates.startswith("2016"):
                feature_groups_2016[feature].append(id)

        return feature_groups, feature_groups_2016


        
    def load_metrics_per_feature_group(feature_groups, feature_groups_2016):
        results_summary = {}

        for feature_name in feature_groups:
            id_list_all = feature_groups[feature_name]
            id_list_2016 = feature_groups_2016[feature_name] if feature_groups_2016 else []

            def compute_metrics(id_list):
                nlls = []
                crps = []
                for id in id_list:
                    file_path = os.path.join(dir, f"experiment_results_{id}.pkl")
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        try:
                            nll = data.loc[data['Metric'] == nll_name, 'Mean'].values[0]
                            crps_val = data.loc[data['Metric'] == crps_name, 'Mean'].values[0]
                            nlls.append(nll)
                            crps.append(crps_val)
                        except IndexError:
                            continue
                return nlls, crps

            nlls_all, crps_all = compute_metrics(id_list_all)
            nlls_2016, crps_2016 = compute_metrics(id_list_2016) if compute_2016 else ([], [])

            if nlls_all and crps_all:
                results_summary[feature_name] = {
                    "average_nll": round(sum(nlls_all) / len(nlls_all), 5),
                    "average_crps": round(sum(crps_all) / len(crps_all), 5),
                    "average_nll_2016": round(sum(nlls_2016) / len(nlls_2016), 5) if nlls_2016 else None if nlls_2016 else None,
                    "average_crps_2016": round(sum(crps_2016) / len(crps_2016), 5) if crps_2016 else None if nlls_2016 else None,
                    "count": len(nlls_all),
                    "count_2016": len(nlls_2016) if nlls_2016 else None
                }
        return results_summary


    def load_result_df(results_summary):
        pretty_names = {
            "power, mean ws": "Power, mean wind speed",
            "power, ws at 10 loc": "Power, 10 wind speeds",
            "power, all ws": "Power, all wind speeds",
            "power, all ws, time bin": "Power, all wind speeds, time"
        }

        df_data = []
        for feature, stats in results_summary.items():
            row = {
                "Feature model": pretty_names.get(feature, feature),
                "CRPS": stats["average_crps"],
                "NLL": stats["average_nll"],
            }

            if compute_2016:
                row["CRPS_2016"] = stats["average_crps_2016"]
                row["NLL_2016"] = stats["average_nll_2016"]

            df_data.append(row)

        df = pd.DataFrame(df_data)
        columns = ["Feature model", "CRPS", "NLL"]
        if compute_2016:
            columns += ["CRPS_2016", "NLL_2016"]

        df = df[columns]
        df = df.set_index("Feature model")
        df = df.round(3)

        feature_order = [
            "Power, mean wind speed",
            "Power, 10 wind speeds",
            "Power, all wind speeds",
            "Power, all wind speeds, time"
        ]
        df.index = pd.CategoricalIndex(df.index, categories=feature_order, ordered=True)
        df = df.sort_index()
        return df
    
    feature_groups, feature_groups_2016 = categorize_ID_to_feature_group()
    results_summary = load_metrics_per_feature_group(feature_groups, feature_groups_2016)
    df = load_result_df(results_summary)
    return df


import pickle
import os
import pandas as pd

# analyse impact of different training periods
def analyze_impact_of_training_period(directory="experiments/"):

    id_pairs = [(1, 5), (2, 6), (3, 7), (4,4), (6,14)]  # (1, 5), (2, 6), (3, 7): same validation quarter, but different training quarter

    # Function to load a pickle file given an ID
    def load_experiment_by_id(exp_id):
        filepath = os.path.join(directory, f"experiment_results_{exp_id}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    final_dataframes = []

    print("🔄 Loading experiment pairs:")
    
    for id1, id2 in id_pairs:
        print(f" - Pair: ({id1}, {id2})")
        exp1 = load_experiment_by_id(id1)
        exp2 = load_experiment_by_id(id2)

        # Extract the rows for nll_5000 and crps_5000 from both experiments
        exp1_nll_5000 = exp1[exp1["Metric"] == "nll_5000"]
        exp1_crps_5000 = exp1[exp1["Metric"] == "crps_5000"]
        exp2_nll_5000 = exp2[exp2["Metric"] == "nll_5000"]
        exp2_crps_5000 = exp2[exp2["Metric"] == "crps_5000"]

        # Create a dataframe for the pair with the required structure
        pair_data = [
            [f"nll_5000 (id: {id1})", exp1_nll_5000["Mean"].values[0], exp1_nll_5000["Median"].values[0], exp1_nll_5000["Min"].values[0], exp1_nll_5000["Max"].values[0]],
            [f"nll_5000 (id: {id2})", exp2_nll_5000["Mean"].values[0], exp2_nll_5000["Median"].values[0], exp2_nll_5000["Min"].values[0], exp2_nll_5000["Max"].values[0]],
            [f"crps_5000 (id: {id1})", exp1_crps_5000["Mean"].values[0], exp1_crps_5000["Median"].values[0], exp1_crps_5000["Min"].values[0], exp1_crps_5000["Max"].values[0]],
            [f"crps_5000 (id: {id2})", exp2_crps_5000["Mean"].values[0], exp2_crps_5000["Median"].values[0], exp2_crps_5000["Min"].values[0], exp2_crps_5000["Max"].values[0]]
        ]

        # Convert the pair data into a DataFrame
        pair_df = pd.DataFrame(pair_data, columns=["Metric", "Mean", "Median", "Min", "Max"])

        # Calculate the relative difference between the means of nll_5000 and crps_5000 across the two id pairs
        nll_5000_mean_id1 = exp1_nll_5000["Mean"].values[0]
        nll_5000_mean_id2 = exp2_nll_5000["Mean"].values[0]
        crps_5000_mean_id1 = exp1_crps_5000["Mean"].values[0]
        crps_5000_mean_id2 = exp2_crps_5000["Mean"].values[0]

        nll_5000_rel_diff = ((nll_5000_mean_id2 - nll_5000_mean_id1) / nll_5000_mean_id1) * 100
        crps_5000_rel_diff = ((crps_5000_mean_id2 - crps_5000_mean_id1) / crps_5000_mean_id1) * 100

        # Append the relative difference as a new column to the dataframe
        pair_df["Relative Difference (%)"] = [nll_5000_rel_diff, nll_5000_rel_diff, crps_5000_rel_diff, crps_5000_rel_diff]

        pair_df = pair_df.round(3)

        pair_df["Relative Difference (%)"] = pair_df["Relative Difference (%)"].round(0).astype(int)

        # Append the resulting dataframe to the final list
        final_dataframes.append(pair_df)

    print("✅ All pairs processed successfully.")
    
    # Return the list of dataframes for all pairs
    return final_dataframes


def analyze_baseline_for_different_feature_models(output_file = "C:/Users/Minu/Documents/results", score="NLL"):

    filepath = f"{output_file}/baseline"
    files = glob.glob(os.path.join(filepath, 'experiment_results_*'))

    ids = [int(re.search(r'\d+', s).group()) for s in files if re.search(r'\d+', s)]


    configs = [
        'power',
        'power, mean ws',
        'power, ws at 10 loc',
        'power, all ws',
        'power, all ws, time bin'
    ]

    dfs = []
    for id in ids:
            config = ExperimentMapper.map_id_to_config(id)
            date = ExperimentMapper.extract_date_abbreviations_from_config(config)

            if date == "Q4 2022 / FY 2023":
                feature = ExperimentMapper.get_feature_string_from_selected_features(config)

                if feature in configs:
                    file_path = os.path.join(filepath, f"experiment_results_{id}.pkl")
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        data = data.loc[data['Metric'] == score]
                        data = data.copy()
                        data.index = [feature]
                        data = data.drop(columns=['Metric'])  # 🚫 Drop the 'Metric' column
                        dfs.append(data)

    result_df = pd.concat(dfs)


    dfs_ft = []

    for id in ids:
            config = ExperimentMapper.map_id_to_config(id)
            date = ExperimentMapper.extract_date_abbreviations_from_config(config)

            if date == "2016-2022 / FY 2023":
                feature = ExperimentMapper.get_feature_string_from_selected_features(config)

                if feature in configs:
                    file_path = os.path.join(filepath, f"experiment_results_{id}.pkl")
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        data = data.loc[data['Metric'] == score]
                        data = data.copy()
                        data.index = [feature]
                        data = data.drop(columns=['Metric'])  # 🚫 Drop the 'Metric' column
                        dfs_ft.append(data)

    result_df_ft = pd.concat(dfs_ft)

    # Desired row order
    feature_order = [
        'power',
        'power, mean ws',
        'power, ws at 10 loc',
        'power, all ws',
        'power, all ws, time bin'
    ]

    # Reindex both DataFrames to match desired order
    result_df = result_df.reindex(feature_order)
    result_df_ft = result_df_ft.reindex(feature_order)

    result_df = result_df.round(3)
    result_df["Max"] = result_df["Max"].round(1)

    result_df_ft = result_df_ft.round(3)
    result_df_ft["Max"] = result_df_ft["Max"].round(1)

    display(result_df)
    print(f"{score} scores for the baseline model for different feature models (Q4 training)")

    display(result_df_ft)
    print(f"{score} scores for the baseline model for different feature models (2016-2022 training)")

    return result_df, result_df_ft

def baseline_parameters_for_different_feature_models(output_file = "C:/Users/Minu/Documents/results"):

    filepath = f"{output_file}/baseline"
    files = glob.glob(os.path.join(filepath, 'experiment_[0-9]*.pkl'))

    ids = [int(re.search(r'\d+', s).group()) for s in files if re.search(r'\d+', s)]


    configs = [
        'power',
        'power, mean ws',
        'power, ws at 10 loc',
        'power, all ws',
        'power, all ws, time bin'
    ]
    beta_1s_q4 = []
    beta_0s_q4 = []
    sigma_sqs_q4 = []
    feature_labels_q4 = []


    beta_1s_ft = []
    beta_0s_ft = []
    sigma_sqs_ft = []
    feature_labels_ft = []

    for id in ids:
            config = ExperimentMapper.map_id_to_config(id)
            date = ExperimentMapper.extract_date_abbreviations_from_config(config)

            if date == "Q4 2022 / FY 2023":
                feature = ExperimentMapper.get_feature_string_from_selected_features(config)

                if feature in configs:
                    file_path = os.path.join(filepath, f"experiment_{id}.pkl")
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    beta_0 = data.beta_0
                    beta_1 = data.beta_1
                    sigma_sq = data.sigma_sq

                    # Reorder beta_1 so that power_t-96 is first
                    col_idx = data.X_train.columns.get_loc("power_t-96")
                    print("X columns:", data.X_train.columns)
                    print("beta0:", beta_0)
                    print("beta1s:", beta_1.round(3))
                    beta_power = beta_1[col_idx]
                    # Get remaining 4 coefficients (excluding power_t-96)
                    beta_others = [b for i, b in enumerate(beta_1) if i != col_idx][:4]
                    beta_1_reordered = [beta_power] + beta_others

                    # Append values
                    beta_0s_q4.append(beta_0)
                    beta_1s_q4.append(beta_1_reordered)
                    feature_labels_q4.append(feature)
                    sigma_sqs_q4.append(sigma_sq)

            if date == "2016-2022 / FY 2023":
                feature = ExperimentMapper.get_feature_string_from_selected_features(config)

                if feature in configs:
                    file_path = os.path.join(filepath, f"experiment_{id}.pkl")
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    beta_0 = data.beta_0
                    beta_1 = data.beta_1
                    sigma_sq = data.sigma_sq

                    # Reorder beta_1 so that power_t-96 is first
                    col_idx = data.X_train.columns.get_loc("power_t-96")
                    #print("X columns:", data.X_train.columns)
                    #print("beta0:", beta_0)
                    #print("beta1s:", beta_1.round(3))
                    beta_power = beta_1[col_idx]
                    # Get remaining 4 coefficients (excluding power_t-96)
                    beta_others = [b for i, b in enumerate(beta_1) if i != col_idx][:4]
                    beta_1_reordered = [beta_power] + beta_others

                    # Append values
                    beta_0s_ft.append(beta_0)
                    beta_1s_ft.append(beta_1_reordered)
                    feature_labels_ft.append(feature)
                    sigma_sqs_ft.append(sigma_sq)      
    
    #rounded_numbers = [round(num, 2) if isinstance(num, (int, float)) else num for num in beta_1s_ft]
    #print(rounded_numbers)

    # Create DataFrame
    column_names = ['beta_0'] + [f'beta_1_{i+1}' for i in range(5)]
    result_df = pd.DataFrame(beta_1s_q4, columns=column_names[1:], index=feature_labels_q4)
    result_df['beta_0'] = beta_0s_q4
    result_df['sigma_sq'] = sigma_sqs_q4
    result_df = result_df[['beta_0'] + column_names[1:] + ['sigma_sq']]  # Ensure correct order
    
    # Create DataFrame
    column_names_2 = ['beta_0'] + [f'beta_1_{i+1}' for i in range(5)]
    result_df_2 = pd.DataFrame(beta_1s_ft, columns=column_names_2[1:], index=feature_labels_ft)
    result_df_2['beta_0'] = beta_0s_ft
    result_df_2['sigma_sq'] = sigma_sqs_ft
    result_df_2 = result_df_2[['beta_0'] + column_names[1:] + ['sigma_sq']]  # Ensure correct order

    # Desired row order
    feature_order = [
        'power',
        'power, mean ws',
        'power, ws at 10 loc',
        'power, all ws',
        'power, all ws, time bin'
    ]

    # Reindex both DataFrames to match desired order
    result_df = result_df.reindex(feature_order)
    result_df_2 = result_df_2.reindex(feature_order)

    result_df = result_df.round(3)
    result_df_2 = result_df_2.round(3)


    display(result_df)
    print("Estimated model parameters for baseline regression models using different feature configurations for Q4 training. The beta1 coefficients are reordered to place 'power_t-96' first.")

    display(result_df_2)
    print("Estimated model parameters for baseline regression models using different feature configurations for 2016-2022 training. The beta1 coefficients are reordered to place 'power_t-96' first.")

    return result_df, result_df_2

import pickle
import numpy as np

def load_scores_id_1():
    file = 'C:/Users/Minu/Documents/results/tabpfn/experiment_results_1.pkl'

    with open(file, 'rb') as f:
        experiment_1_results = pickle.load(f)

    nll_df = experiment_1_results[experiment_1_results["Metric"].str.contains("nll")].reset_index(drop=True)
    crps_df = experiment_1_results[experiment_1_results["Metric"].str.contains("crps")].reset_index(drop=True)

    nll_df['Metric'] = nll_df['Metric'].replace('nll_5000', 'nll_5000_smoothed')

    # Optionally, you can do the same for "crps_5000" (if you want to add "smoothed" there too):
    crps_df['Metric'] = crps_df['Metric'].replace('crps_5000', 'crps_5000_smoothed')

    # Inline rounding to 3 significant figures for all numeric columns
    for col in nll_df.select_dtypes(include=[np.number]).columns:
        nll_df[col] = nll_df[col].apply(lambda x: round(x, 3 - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0)

    for col in crps_df.select_dtypes(include=[np.number]).columns:
        crps_df[col] = crps_df[col].apply(lambda x: round(x, 3 - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0)
    
    return nll_df, crps_df


import pickle
import numpy as np
import pandas as pd

def tabpfn_summary_2022_training():
    combinations = [
        [42, 43, 44, 45],
        [46, 47, 48, 49],
        [50, 51, 52, 53],
        [38, 39, 40, 41]
    ]
    
    # Initialize 2D lists to collect NLL and CRPS values
    all_nlls = []
    all_crps = []

    for combination in combinations:
        print(f"Processing combination: {combination}")

        # Generate file paths for the current combination
        filepaths = [f"C:/Users/Minu/Documents/results/tabpfn/quantiles/experiment_results_{experiment_id}.pkl" for experiment_id in combination]
        
        # Lists to store all the NLL and CRPS values for the current combination
        nll_all = []
        crps_all = []

        # Iterate over the filepaths and extract the NLL and CRPS values
        for filepath in filepaths:
            with open(filepath, "rb") as file:
                experiment_result = pickle.load(file)

            # Extract NLL and CRPS values
            nll = experiment_result.loc[0, experiment_result.columns.str.startswith('score')].values
            crps = experiment_result.loc[1, experiment_result.columns.str.startswith('score')].values

            # Append the NLL and CRPS values to the lists for the current combination
            nll_all.extend(nll)
            crps_all.extend(crps)

        # Calculate the mean of NLL and CRPS
        nll_mean = np.mean(nll_all)
        crps_mean = np.mean(crps_all)

        # Calculate quantiles (5th, 25th, 75th, 95th percentiles)
        quantile_percentages = np.array([5, 25, 75, 95])
        nll_quantiles = np.percentile(nll_all, quantile_percentages)
        crps_quantiles = np.percentile(crps_all, quantile_percentages)

        # Append the results to the 2D arrays
        all_nlls.append([nll_mean] + nll_quantiles.tolist())  # Append mean + quantiles
        all_crps.append([crps_mean] + crps_quantiles.tolist())  # Append mean + quantiles

    # Create a DataFrame for the final result
    result_df = pd.DataFrame({
        'Combination': [str(comb) for comb in combinations],
        'NLL_Mean': [row[0] for row in all_nlls],
        'NLL_Quantiles_5th': [row[1] for row in all_nlls],
        'NLL_Quantiles_25th': [row[2] for row in all_nlls],
        'NLL_Quantiles_75th': [row[3] for row in all_nlls],
        'NLL_Quantiles_95th': [row[4] for row in all_nlls],
        'CRPS_Mean': [row[0] for row in all_crps],
        'CRPS_Quantiles_5th': [row[1] for row in all_crps],
        'CRPS_Quantiles_25th': [row[2] for row in all_crps],
        'CRPS_Quantiles_75th': [row[3] for row in all_crps],
        'CRPS_Quantiles_95th': [row[4] for row in all_crps],
    })

    result_df = result_df.round(3)

    # Return the final results as a DataFrame
    return result_df


def load_experiment_results(ids, method, q4=False):
    """
    Baseline 
    Table 5.1, 5.2 IDs:
        Feature                  | 2016–2022 | Q4 2022
        ------------------------|-----------|---------
        Power                   | 33        | 24
        Power, mean ws          | 32        | 25
        Power, ws at 10 loc     | 31        | 26
        Power, all ws           | 30        | 27
        Power, all ws, t-bin    | 29        | 28

    TabPFN
    Table 5.3 IDs:
        Score Type              | NLL / CRPS
        ------------------------|-------------
        Smoothed bin            | 1
        Stepwise bin            | 1
        Decile Linear           | 1
        Decile Pchip            | 1
        Decile Hybrid           | 1

    Table 5.4 IDs:
        Score Source            | 23Q1 | 23Q2 | 23Q3 | 23Q4
        ------------------------|------|------|------|------
        22Q1                   | 1    | 14   | 17   | 16
        22Q2                   | 18   | 2    | 19   | 20
        22Q3                   | 21   | 15   | 3    | 22
        22Q4                   | 5    | 6    | 7    | 4

    Table 5.5, 5.6 IDs:
        Feature                | 2022 IDs         | Q4 2022 IDs
        -----------------------|------------------|-------------
        Power                  |                  | 
        Power, mean ws         | 42,43,44,45      | 4,5,6,7
        Power, ws at 10 loc    | 46,47,48,49      | 8,9
        Power, all ws          | 50,51,52,53      | 10,11
        Power, all ws, t-bin   | 38,39,40,41      | 12,13

    NGBoost
    Table 5.7 (ngboost/full_year/case):
        Feature                | NLL | CRPS
        ------------------------|-----|-----
        Power                  | 2   | 1
        Power, mean ws         | 6   | 5
        Power, ws at 10 loc    | 8   | 7
        Power, all ws          | 10  | 9
        Power, all ws, t-bin   | 12  | 11

    Table 5.8, 5.9 (full_year/case, q4_train/case):
        Feature                | 2016–2022 | Q4 2022
        ------------------------|------------|---------
        Power                  | 2          | 2
        Power, mean ws         | 6          | 6
        Power, ws at 10 loc    | 8          | 8
        Power, all ws          | 10         | 10
        Power, all ws, t-bin   | 12         | 12
    """
    nll_list = []
    crps_list = []

    if method == "ngboost":
        for id in ids:
            if q4:
                pkl_file_path = f"C:/Users/Minu/Documents/results/{method}/q4_train/case{id}.xlsx"
            else:
                pkl_file_path = f"C:/Users/Minu/Documents/results/{method}/full_year/case{id}.xlsx"
            
            df = pd.read_excel(pkl_file_path, sheet_name="Detailed_Scores")
            crps = df['CRPS_gaussian'].values
            nll = df['NLL'].values
            
            crps_list.append(crps)
            nll_list.append(nll)
    
    elif method == "baseline":
        for id in ids:
            file_path = f"C:/Users/Minu/Documents/results/{method}/quantiles/experiment_results_{id}.pkl"
            try:
                with open(file_path, "rb") as f:
                    df = pickle.load(f)
                    # Select only columns starting with "score"
                    score_cols = [col for col in df.columns if col.startswith("score")]
                    nll = df.iloc[0][score_cols].values
                    crps = df.iloc[1][score_cols].values
                    nll_list.append(nll)
                    crps_list.append(crps)
            except Exception as e:
                print(f"Error with file {file_path}: {e}")

    elif method == "tabpfn":
        for id in ids:
            file_path = f"C:/Users/Minu/Documents/results/{method}/quantiles/experiment_results_{id}.pkl"
            try:
                with open(file_path, "rb") as f:
                    df = pickle.load(f)
                    score_cols = df.columns[5:]  # Assuming first 5 cols are metadata

                    if ids == [1]:
                        # Special case: only ID 1
                        nll = df[df['Metric'].str.startswith('nll')][score_cols].values
                        crps = df[df['Metric'].str.startswith('crps')][score_cols].values
                        return np.array(nll).reshape(-1, nll.shape[-1]), np.array(crps).reshape(-1, crps.shape[-1])
                    
                    elif id == 1:
                        nll_row = df[df["Metric"] == "nll_5000_smoothed"][score_cols].values.squeeze()
                        crps_row = df[df["Metric"] == "crps_5000_raw"][score_cols].values.squeeze()
                        nll_list.append(nll_row)
                        crps_list.append(crps_row)
                    else:
                        # CASE B.2: Other IDs
                        nll_row = df.iloc[0][score_cols].values
                        crps_row = df.iloc[1][score_cols].values
                        nll_list.append(nll_row)
                        crps_list.append(crps_row)
            except Exception as e:
                    print(f"Error with file {file_path}: {e}")
    else:
            raise ValueError(f"Unsupported method: {method}")
    
    return np.array(nll_list), np.array(crps_list)