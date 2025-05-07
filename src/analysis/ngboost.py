import os
from tkinter import Image
import warnings

from matplotlib import pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

from analysis.splits import to_train_validation_test_data
from ngboost.scores import LogScore, CRPScore
from ngboost.distns import Normal
from ngboost.distns import LogNormal
from ngboost import NGBRegressor
from scipy.stats.distributions import norm
from scipy.stats import lognorm
from pathlib import Path
warnings.filterwarnings("ignore")


def evaluate_ngboost_model(
              entsoe, 
              target_column='power', 
              dist=Normal, 
              case=1, 
              n_estimators=100, 
              learning_rate=0.03, 
              random_state=42, 
              output_file='C:/Users/Minu/Documents/NGboost/',
              train_start = "2016-01-01",
              train_end = "2022-12-31",
              validation_start = "2023-01-01",
              validation_end = "2023-12-31"
    ):
    
    baseline_dir = os.path.join(output_file, "ngboost")
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)


    if train_start == "2022-10-01" and train_end == "2022-12-31":
           output_file=f"{baseline_dir}/q4_train/"

    if train_start == "2016-01-01" and train_end == "2022-12-31":
           output_file=f"{baseline_dir}/full_year/"

    # Scale power data
    max_power_value = entsoe[target_column].max()
    max_power_value_rounded = np.ceil(max_power_value / 1000) * 1000
    #epsilon = 1e-9
    epsilon = 1e-5
    entsoe[target_column] = np.log(entsoe[target_column] / max_power_value_rounded + epsilon)
    entsoe['power_t-96'] = entsoe[target_column].shift(96)
    entsoe['interval_index'] = ((entsoe.index.hour * 60 + entsoe.index.minute) // 15) + 1
    entsoe.dropna(inplace=True)

    # Train-test split
    train_X, train_y, validation_X, validation_y, test_X, test_y = to_train_validation_test_data(entsoe, train_start, train_end, validation_start, validation_end)

    #display(validation_y)
    #train, validation, test = to_train_validation_test_data(entsoe, "2022-12-31 23:45:00", "2023-12-31 23:45:00")
    
    output_dir = output_file
    os.makedirs(output_dir, exist_ok=True)
    print("Creating output_dir:", output_dir)


    if case == 1:
            feature_columns = ['power_t-96']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            print(f"output file case = {case}: {output_file}")
            feature_abbr = "P"
    
    if case == 2:
            feature_columns = ['power_t-96']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "P"


    if case == 3:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean"

    
    if case == 4:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean"
    
    elif case == 5:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean"


    elif case == 6:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean"


    elif case == 7:
            feature_columns = ['power_t-96', 'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_10_loc"



    elif case == 8:
            feature_columns = ['power_t-96', 'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_10_loc"


    elif case == 9:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc"


    elif case == 10:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc"

    
    elif case == 11:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10', 'interval_index']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc, t_index"

    elif case == 12:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10', 'interval_index']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc, t_index"

    elif case == 13:
            feature_columns = ['power_t-96', 'interval_index']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, t_index"

    elif case == 14:
            feature_columns = ['power_t-96', 'interval_index']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, t_index"

    elif case == 15:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean', 'interval_index']
            loss_function = CRPScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean, t_index"

    elif case == 16:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean', 'interval_index']
            loss_function = LogScore
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean, t_index"



    #X_train, y_train = train_X[feature_columns], train_y[target_column]
    #X_validation, y_validation = validation_X[feature_columns], validation_y[target_column]
   
    X_train, y_train = train_X[feature_columns], train_y
    X_validation, y_validation = validation_X[feature_columns], validation_y

    with warnings.catch_warnings():
       warnings.simplefilter("ignore", category=FutureWarning)
       # Train model
       model = NGBRegressor(
                Dist=dist, Score=loss_function, 
                n_estimators=n_estimators, learning_rate=learning_rate, 
                random_state=random_state, verbose=True, verbose_eval=True
        )
       model.fit(X_train, y_train.squeeze(), X_val=X_validation, Y_val=y_validation.squeeze())

    # Split validation data into 96 intervals
    X_validation_sub_arrays = [X_validation[i::96] for i in range(96)]
    y_validation_sub_arrays = [y_validation[i::96] for i in range(96)]

    model_scores_intervals = [model.score(np.array(X_validation_sub_arrays[i]), np.array(y_validation_sub_arrays[i])) for i in range(96)]
    model_scores_overall = model.score(np.array(X_validation), np.array(y_validation))

    y_val_pred = model.predict(X_validation)
    y_val_dists = model.pred_dist(X_validation)


    # Compute predictions
    #y_val_pred = model.predict(X_validation)
    #y_val_dists = model.pred_dist(X_validation)

    # Compute CRPS and NLL per sample
    crps_gaussian, crps_log_gaussian, nll, pit_values = [], [], [], []

    for i in range(len(y_val_pred)):
        y = y_validation.iloc[i]
        sigma, mu = y_val_dists[i].scale, y_val_dists[i].loc

        if dist == Normal:
               pit_value = norm.cdf(y, scale=sigma, loc=mu) # Note: loc = mean, scale = standard deviation (scipy)
               z = (y - mu) / sigma
               crps_gaussian.append(
                      sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))
               
               crps_log_gaussian.append(0)

               nll.append(-norm.logpdf(y, scale=sigma, loc=mu))
        
        # NGBoost uses the CRPS formula of the Normal distribution with y -> Ln(y) rather than the correct CRPS formula for the LogNormal distribution
        # If dist == Normal only crps_gaussian is to be used.
        # If dist == LogNormal then CRPS log_gaussian is the correct formula. CRPS_Gaussian is calculated to double check that this is the score that NGBoost returns
        else:
               pit_value = lognorm.cdf(y, s=sigma, scale=np.exp(mu)) # Note: s = sigma and scale = exp(mu) (scipy)
               ylog = np.log(y)
               z = (ylog - mu) / sigma
               crps_gaussian.append(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
                                    )
               crps_log_gaussian.append(
                      y * (2 * norm.cdf(z) - 1) - 2 * np.exp(mu + 0.5 * sigma**2) * (norm.cdf(z - sigma) + norm.cdf(sigma/np.sqrt(2)) - 1)
                      )
               nll.append(-lognorm.logpdf(y, s=sigma, scale=np.exp(mu)))


        pit_values.append(pit_value)   


    # Compute per-interval statistics
    crps_gaussian_sub_arrays = [crps_gaussian[i::96] for i in range(96)]

    nll_sub_arrays = [nll[i::96] for i in range(96)]
    pit_sub_arrays = [pit_values[i::96] for i in range(96)]
    
    crps_lognormal_sub_arrays = [crps_log_gaussian[i::96] for i in range(96)]

    crps_lognormal_stats = {
           'mean': [np.mean(arr) for arr in crps_lognormal_sub_arrays],
           'min': [np.min(arr) for arr in crps_lognormal_sub_arrays],
           'max': [np.max(arr) for arr in crps_lognormal_sub_arrays]
    }

    crps_gaussian_stats = {
        'mean': [np.mean(arr) for arr in crps_gaussian_sub_arrays],
        'min': [np.min(arr) for arr in crps_gaussian_sub_arrays],
        'max': [np.max(arr) for arr in crps_gaussian_sub_arrays]
    }

    nll_stats = {
        'mean': [np.mean(arr) for arr in nll_sub_arrays],
        'min': [np.min(arr) for arr in nll_sub_arrays],
        'max': [np.max(arr) for arr in nll_sub_arrays]
    }

    # Calculates deciles per time interval
    deciles = []
    
    for i in range(0, 96):
        pit_a = pit_sub_arrays[i]
        bin_edges = np.arange(0, 1.1, 0.1)  # Creating bin edges from 0 to 1 with a step of 0.1
        decile, bins = np.histogram(pit_a, bins=bin_edges, density=True)
        #decile, bin_edges = np.histogram(pit_a, bins=10, density=True)
        deciles.append(decile)

    bin_edges = np.arange(0, 1.1, 0.1)  # Creating bin edges from 0 to 1 with a step of 0.1
    sum_deciles, bins = np.histogram(pit_values, bins=bin_edges, density=True)
    
    # Create DataFrames
    results_per_time_interval_df = pd.DataFrame({
       'Interval': list(range(1, 97)),
        **{f'CRPS_gaussian_{k}': v for k, v in crps_gaussian_stats.items()},
        **{f'CRPS_lognormal_{k}': v for k, v in crps_lognormal_stats.items()},
        **{f'NLL_{k}': v for k, v in nll_stats.items()},
        'model_scores': model_scores_intervals,
        'pit_values': deciles
        })

    results_summary_stats_df = pd.DataFrame({
        'CRPS_gaussian_mean': np.mean(crps_gaussian_stats['mean']),
        'CRPS_gaussian_min': np.min(crps_gaussian_stats['min']),
        'CRPS_gaussian_max': np.max(crps_gaussian_stats['max']),
        'CRPS_lognormal_mean': np.mean(crps_lognormal_stats['mean']),
        'CRPS_lognormal_min': np.min(crps_lognormal_stats['min']),
        'CRPS_lognormal_max': np.max(crps_lognormal_stats['max']),
        'NLL_mean': np.mean(nll_stats['mean']),
        'NLL_min': np.min(nll_stats['min']),
        'NLL_max': np.max(nll_stats['max']),
        'model_scores_mean': np.mean(model_scores_intervals),
        'pit_overall': [sum_deciles]

    }, index=[0])

    results_per_row_df = pd.DataFrame({
        'Entry_no': list(range(1, len(y_validation) + 1)),
        'CRPS_gaussian': crps_gaussian,
        'CRPS_lognormal': crps_log_gaussian,
        'NLL': nll
    })

    hyperparameters_df = pd.DataFrame({
        'dataset': 'entsoe',
        'feature_abbr': feature_abbr,
        'feature_columns': [feature_columns],
        'distribution': str(dist),
        'loss_function': str(loss_function),
        'iterations': n_estimators,
        'learning_rate': learning_rate,
        'random_state': random_state
    })

    # Save results to an Excel file
    with pd.ExcelWriter(output_file) as writer:
        results_per_time_interval_df.to_excel(writer, sheet_name='Interval_Scores', index=False)
        results_summary_stats_df.to_excel(writer, sheet_name='Summary_Scores', index=False)
        results_per_row_df.to_excel(writer, sheet_name='Detailed_Scores', index=False)
        hyperparameters_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        
        
        #output_dir = os.path.join(output_dir.rstrip('/'), '')

        if train_start == "2022-10-01" and train_end == "2022-12-31":
               file_paths = list(Path(output_dir).glob("*.xlsx"))  # Use Path.glob to find files
               print("file_paths\n", file_paths)
        
        elif train_start == "2016-01-01" and train_end == "2022-12-31":
               file_paths = list(Path(output_dir).glob("*.xlsx"))  # Use Path.glob to find files
               print("output_file in dist=normal", output_file)


        #if dist == Normal:
        #       file_paths = list(Path(output_dir).glob("*.xlsx"))  # Use Path.glob to find files
        #       print("file_paths\n", file_paths)
        
        #else:
        #        file_paths = glob.glob(f"{output_file}*.xlsx")  # Update with the correct path


    # Step 1: Get all Excel files in a folder

    # Step 2: Check if there are exactly 16 Excel files
    #print(file_paths)
    if len(file_paths) == 16:
        merged_data = []

        # Step 3: Loop through each file and extract both sheets
        for file in file_paths:
                try:
                        # Read "Summary_Scores" sheet
                        df_scores = pd.read_excel(file, sheet_name="Summary_Scores")
                        df_scores["Source_File"] = file  # Optional: Track source file

                        # Read "Hyperparameters" sheet
                        df_hyperparams = pd.read_excel(file, sheet_name="Hyperparameters")
                        df_hyperparams["Source_File"] = file  # Optional: Track source file

                        # Combine the two dataframes horizontally (side by side)
                        combined_df = pd.concat([df_scores, df_hyperparams], axis=1)
                        merged_data.append(combined_df)

                except Exception as e:
                        print(f"Could not read {file}: {e}")

        # Step 4: Merge all data into one DataFrame
        final_merged_df = pd.concat(merged_data, ignore_index=True)

        # Step 5: Save to a new Excel file
        
        #final_merged_df.to_excel(f"{file_paths}/Merged_Sheet.xlsx", index=False)
        final_merged_df.to_excel(f"{output_dir}/Merged_Sheet.xlsx", index=False)

        print("Merge completed! The final file is 'Merged_Sheet.xlsx'.")

        plt.hist(bin_edges[:-1], bin_edges, weights=results_summary_stats_df['pit_overall'].values, edgecolor='black', alpha=0.7)
        plt.xlabel('Bin Edges')
        plt.ylabel('Density')
        plt.title('Histogram of Deciles')
        # Save the plot as an image (e.g., PNG format)
        image_filename = 'histogram.png'
        plt.savefig(image_filename)
        plt.close()
        
        # Load the existing Excel file (if it already exists)
        excel_file = f"{output_dir}/Merged_Sheet.xlsx"

        wb = openpyxl.load_workbook(excel_file)
        
        # Select the specific sheet where you want to insert the image
        new_sheet = wb.create_sheet('Histogram Sheet')
        
        # Load the image you saved earlier
        img = Image(image_filename)
        
        # Specify the location where you want the image to appear in the sheet (e.g., cell 'A1')
        new_sheet.add_image(img, 'A1')
        
        # Save the modified Excel file
        wb.save(excel_file)
        
    else:
           print(f"Expected 16 Excel files, but found {len(file_paths)} files. Skipping the merge step.")


    return results_per_time_interval_df, results_summary_stats_df, results_per_row_df, hyperparameters_df