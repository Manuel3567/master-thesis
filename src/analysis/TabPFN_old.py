import numpy as np
import scipy.interpolate as spi
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from analysis.splits_old import to_train_validation_test_data
from analysis.transformations_old import scale_power_data, add_lagged_features, add_interval_index
from tabpfn import TabPFNRegressor


def train_tabpfn_model( 
    entsoe, 
    target_column='power', 
    case=1,
    n_estimators=100, 
    learning_rate=0.03, 
    random_state=42, 
    output_file='../results/Tabpfn/'
):
    entsoe = scale_power_data(entsoe)
    entsoe = add_lagged_features(entsoe)
    entsoe = add_interval_index(entsoe)
    entsoe.dropna(inplace=True)

    train, validation, test = to_train_validation_test_data(entsoe, "2016-03-31 23:45:00", "2016-11-30 23:45:00")

    if case == 1:
            feature_columns = ['power_t-96']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "P"
    
    if case == 2:
            feature_columns = ['power_t-96']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "P"


    if case == 3:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean"

    
    if case == 4:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean"
    
    elif case == 5:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean"


    elif case == 6:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean"


    elif case == 7:
            feature_columns = ['power_t-96', 'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_10_loc"



    elif case == 8:
            feature_columns = ['power_t-96', 'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_10_loc"


    elif case == 9:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc"


    elif case == 10:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc"

    
    elif case == 11:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10', 'interval_index']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc, t_index"

    elif case == 12:
            feature_columns = ['power_t-96', 'ws_10m_loc_mean', 'ws_100m_loc_mean',
                               'ws_10m_loc_1', 'ws_10m_loc_2', 'ws_10m_loc_3', 'ws_10m_loc_4', 'ws_10m_loc_5', 'ws_10m_loc_6',
                               'ws_10m_loc_7', 'ws_10m_loc_8', 'ws_10m_loc_9', 'ws_10m_loc_10',
                               'ws_100m_loc_1', 'ws_100m_loc_2', 'ws_100m_loc_3', 'ws_100m_loc_4', 'ws_100m_loc_5', 'ws_100m_loc_6',
                               'ws_100m_loc_7', 'ws_100m_loc_8', 'ws_100m_loc_9', 'ws_100m_loc_10', 'interval_index']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, ws_mean, ws_10_loc, t_index"

    elif case == 13:
            feature_columns = ['power_t-96', 'interval_index']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, t_index"

    elif case == 14:
            feature_columns = ['power_t-96', 'interval_index']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "p, t_index"

    elif case == 15:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean', 'interval_index']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean, t_index"

    elif case == 16:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean', 'interval_index']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean, t_index"


    X_train, y_train = train[feature_columns], train[target_column]
    X_validation, y_validation = validation[feature_columns], validation[target_column]
    
    model = TabPFNRegressor(device='auto', ignore_pretraining_limits=True, fit_mode='low_memory')
    print("model successfully called")
    model.fit(X_train, y_train)
    print("model fit")
    #quantiles_custom = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    quantiles_custom = np.arange(0.1, 1, 0.1)

    # Predict on training data
    #means_val = model.predict(X_validation)
    probs_val = model.predict(X_validation, output_type="full", quantiles=quantiles_custom)
    print("quantiles calculated")
    all_quantiles = np.array(probs_val["quantiles"])
    # TabPFN fails to calculate the first and last quantiles. Hence correct these values.
    # Replace first row: mean of second row and -21 <= -20.6 = log(1E-9) as the scaled power P has a min of 0. 
    #all_quantiles[0,:] = (all_quantiles[1,:] + (-1000)) / 2

    #Replace last row: mean of second-to-last row and 0 = log(1) as the scaled power P/P_max <= 1.
    #all_quantiles[-1,:] = (all_quantiles[-2,:] - 0) / 2


    print("model quantiles finished")

    # Calculate Mean Squared Error (MSE) for training and validation
    #train_mse = mean_squared_error(y_train, y_train_pred)
    #val_mse = mean_squared_error(y_validation, y_val_pred)

    # Print the results
    #print(f"Training features: {[c for c in X_train.columns]} -> {y_train.name}")
    #print(f"Train MSE:\t {train_mse}")
    #print(f"Validation MSE:\t {val_mse}")

    return X_validation, y_validation, quantiles_custom, all_quantiles


def compute_cdf_pdf_interpolators(quantiles, probabilities, y_min=-30, y_max=30):
    """
    Returns interpolation functions for CDF and PDF using linear, PCHIP, and hybrid methods.

    Parameters:
    - quantiles: Array of quantiles.
    - probabilities: Array of associated probabilities for the quantiles.
    - y_min: Minimum x value for extrapolation.
    - y_max: Maximum x value for extrapolation.

    Returns:
    - cdf_linear(x): Linear interpolation CDF function.
    - cdf_pchip(x): PCHIP interpolation CDF function.
    - hybrid_cdf(x): Hybrid CDF (PCHIP + normal tails).
    - pdf_linear(x): Linear interpolation PDF function.
    - pdf_pchip(x): PCHIP derivative PDF function.
    - pdf_hybrid(x): Hybrid PDF (PCHIP + normal tails).
    """

    # Extend quantile and probability arrays
    full_quantiles = np.concatenate(([y_min], quantiles, [y_max]))
    full_probabilities = np.concatenate(([0], probabilities, [1]))

    # --- Calculate minimum difference between consecutive quantiles ---
    delta_quantiles = np.diff(quantiles)
    min_delta_quantile = np.min(delta_quantiles)


    # --- Fit Normal Distributions for Tails ---
    def fit_tail_distribution(quantiles, probabilities):
        """Fits a normal distribution to the given quantiles and associated probabilities."""
        z_scores = norm.ppf(probabilities)
        sigma = (quantiles[1] - quantiles[0]) / (z_scores[1] - z_scores[0])
        mu = quantiles[0] - sigma * z_scores[0]
        return mu, sigma

    # Fit left and right tails
    mu_left, sigma_left = fit_tail_distribution(quantiles[:2], probabilities[:2])
    mu_right, sigma_right = fit_tail_distribution(quantiles[-2:], probabilities[-2:])

    # --- Define CDF Interpolators ---
    cdf_linear_interpolator = spi.interp1d(
        full_quantiles, full_probabilities, kind="linear", fill_value=(0, 1), bounds_error=False
    )
    cdf_pchip_interpolator = spi.PchipInterpolator(full_quantiles, full_probabilities, extrapolate=True)
    #cdf_pchip_interpolator = spi.interp1d(quantiles, probabilities, kind='quadratic', fill_value=(0, 1), bounds_error=False)
    #cdf_pchip_interpolator = spi.Akima1DInterpolator(full_quantiles, full_probabilities)


    def cdf_linear(x):
        """Linear interpolation CDF function."""
        return float(np.clip(cdf_linear_interpolator(x), 0, 1))

    def cdf_pchip(x):
        """PCHIP interpolation CDF function."""
        return float(np.clip(cdf_pchip_interpolator(x), 0, 1))

    def hybrid_cdf(x):
        """
        Hybrid CDF:
        - Left normal fit for x < first quantile
        - PCHIP interpolation for middle range
        - Right normal fit for x > last quantile
        """
        if x < quantiles[0]:  # Left tail
            return norm.cdf(x, loc=mu_left, scale=sigma_left)
        elif x > quantiles[-1]:  # Right tail
            return norm.cdf(x, loc=mu_right, scale=sigma_right)
        else:  # Middle range (PCHIP interpolation)
            return float(np.clip(cdf_pchip_interpolator(x), 0, 1))

    # --- Define PDF functions (derivatives of CDFs) ---
    def pdf_linear(x):
        """Approximates PDF using finite differences of the linear CDF."""
        #eps = eps  # Small step for numerical differentiation
        eps = 0.01
        return (cdf_linear(x + eps) - cdf_linear(x - eps)) / (2 * eps)

    def pdf_linear2(x):
        """Approximates PDF using finite differences of the linear CDF."""
        eps = min_delta_quantile/2
        return (cdf_linear(x + eps) - cdf_linear(x - eps)) / (2 * eps)
    
    def pdf_pchip(x):
        """Approximates PDF using finite differences on the cubic interpolation CDF."""
        eps = min_delta_quantile/2
        print(f"epsilon {eps}")
        print("Step cdf_chip",cdf_pchip_interpolator(x + eps),cdf_pchip_interpolator(x - eps))

        return (cdf_pchip_interpolator(x + eps) - cdf_pchip_interpolator(x - eps)) / (2 * eps)
        #return cdf_pchip_interpolator.derivative()(x)

    def pdf_hybrid(x):
        """
        Hybrid PDF:
        - Left normal distribution for x < first quantile
        - PCHIP interpolation derivative for middle range
        - Right normal distribution for x > last quantile
        """
        if x < quantiles[0]:  # Left tail
            return norm.pdf(x, loc=mu_left, scale=sigma_left)
        elif x > quantiles[-1]:  # Right tail
            return norm.pdf(x, loc=mu_right, scale=sigma_right)
        else:  # Middle range (PCHIP interpolation)
            return cdf_pchip_interpolator.derivative()(x)

    return cdf_linear, cdf_pchip, hybrid_cdf, pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid

def compute_cdf_pdf_normal_distribution(quantiles, probabilities):
    
    mu_fit, sigma_fit = fit_normal_dist_to_quantiles(quantiles, probabilities)

    def cdf_normal(x):
         cdf_normal = norm.cdf(x, mu_fit, sigma_fit)
         return cdf_normal

    def pdf_normal(x):
         pdf_normal = norm.pdf(x, mu_fit, sigma_fit)
         return pdf_normal

    return cdf_normal, pdf_normal, mu_fit, sigma_fit

# Define a function that maps probabilities to normal quantiles
def normal_cdf_inverse(p, mu, sigma):
    return norm.ppf(p, loc=mu, scale=sigma)

def fit_normal_dist_to_quantiles(probabilities, quantiles):
    """ Fits a normal distribution to given quantiles and probabilities. """
    probabilities = np.asarray(probabilities)
    quantiles = np.asarray(quantiles)
    
    # Debugging: Print inputs
    print("Probabilities:", probabilities)
    print("Quantiles:", quantiles)
    
    # Ensure valid inputs
    if np.any(np.isnan(probabilities)) or np.any(np.isnan(quantiles)):
        raise ValueError("Input contains NaN values!")
    
    if np.any(np.isinf(probabilities)) or np.any(np.isinf(quantiles)):
        raise ValueError("Input contains Inf values!")

    # Check if probabilities are within (0,1)
    if np.any(probabilities <= 0) or np.any(probabilities >= 1):
        raise ValueError("Probabilities must be between 0 and 1 (exclusive).")

    # Provide a reasonable initial guess
    mu_init = np.mean(quantiles)
    sigma_init = (np.max(quantiles) - np.min(quantiles)) / 4  # Rough estimate

    try:
        params, _ = curve_fit(normal_cdf_inverse, probabilities, quantiles, p0=[mu_init, sigma_init])
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        return None, None

    mu_fit, sigma_fit = params
    print(f"Estimated Mean: {mu_fit}, Estimated Std Dev: {sigma_fit}")
    
    return mu_fit, sigma_fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_cdf_pdf_dynamic(quantiles, probabilities, y_min, y_max, log_scale=True, eps=0.01, case=1):
    """Plots the CDF and PDF for linear, Akima1DInterpolator, and hybrid interpolation methods."""

    # Generate x values for plotting
    x_values = np.linspace(y_min - 1, y_max + 1, 200)

    # Get CDF and PDF functions
    cdf_linear, cdf_pchip, hybrid_cdf, pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid = compute_cdf_pdf_interpolators(
        quantiles, probabilities, y_min, y_max
    )

    y_min_shift = 1
    cdf_linear_m, cdf_pchip_m, hybrid_cdf_m, pdf_linear_m, pdf_linear2_m, pdf_pchip_m, pdf_hybrid_m = compute_cdf_pdf_interpolators(
    quantiles, probabilities, y_min - y_min_shift, y_max + y_min_shift
    )
    
    mu, sigma = fit_normal_dist_to_quantiles(probabilities, quantiles)



    plt.figure(figsize=(11, 6))
    if case == 1:
        # Case 1
        plt.figure(figsize=(11, 6))
        plt.xlim(-1.5, 1.5)
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF Estimation of N(0,1) via 9 quantiles with different interpolation methods")
        cdf_values = {
            "Linear Interpolation": np.array([cdf_linear(x) for x in x_values]),
            #"Akima Spline Interpolation": np.array([cdf_pchip(x) for x in x_values]),
            "Pchip Interpolation": np.array([cdf_pchip(x) for x in x_values]),
            "True CDF": norm.cdf(x_values)
        }
        for label, values in cdf_values.items():
            plt.plot(x_values, values, label=label)
        plt.scatter(quantiles, probabilities, color='y', marker='o', label="Quantiles")
        plt.legend()
        plt.grid(True)
        plt.show()


        plt.figure(figsize=(11, 6))
        plt.xlim(-1.4, 1.4)
        plt.ylim(0.10, 0.45)
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("PDF Estimation of N(0,1) via 9 quantiles with different interpolation methods")
        pdf_values = {
            f"Linear Interpolation epsilon=0.01": np.array([pdf_linear(x) for x in x_values]),
            "Linear Interpolation epsilon=min_delta_quantile/2": np.array([pdf_linear2(x) for x in x_values]),
            "Pchip Interpolation": np.array([pdf_pchip(x) for x in x_values]),
            "True PDF": norm.pdf(x_values, loc=mu, scale=sigma)
        }
        for label, values in pdf_values.items():
            plt.plot(x_values, values, label=label, linestyle="--")
        plt.scatter(quantiles, [pdf_linear(q) for q in quantiles], marker='x', label="Linear PDF at Quantiles")
        plt.scatter(quantiles, [pdf_pchip(q) for q in quantiles], marker='o', label="Pchip PDF at Quantiles", color='g')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif case == 2:

        plt.figure(figsize=(11, 6))
        plt.xlim(-4, 4)
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF Estimation of N(0,1) via 9 quantiles with different interpolation methods")

        cdf_values = {
            f"Linear Interpolation, y_min={y_min}": np.array([cdf_linear(x) for x in x_values]),
            f"Linear Interpolation, y_min={y_min - y_min_shift}": np.array([cdf_linear_m(x) for x in x_values]),
            "Pchip Interpolation": np.array([cdf_pchip(x) for x in x_values]),
            "Pchip Interpolation + Normal Tails Interpolation": np.array([hybrid_cdf(x) for x in x_values]),
            "True CDF": norm.cdf(x_values)
        }

        for label, values in cdf_values.items():
            plt.plot(x_values, values, label=label)
        
        plt.scatter(quantiles, probabilities, color='y', marker='o', label="Quantiles")
        plt.legend()
        plt.grid(True)
        plt.show()

        pdf_values = {
        f"Linear Interpolation epsilon=0.01, y_min={y_min}": np.array([pdf_linear(x) for x in x_values]),
        f"Linear Interpolation epsilon=0.01, y_min={y_min - y_min_shift}": np.array([pdf_linear_m(x) for x in x_values]),
        "True PDF": norm.pdf(x_values, loc=mu, scale=sigma)
        }
        
        plt.figure(figsize=(11, 6))
        plt.xlabel("Value")
        plt.ylabel("Probability Density")

        plt.title("PDF Estimation of N(0,1) via 9 quantiles with different interpolation methods")

        for label, values in pdf_values.items():
            plt.plot(x_values, values, label=label, linestyle="--")
        plt.scatter(quantiles, [pdf_linear(q) for q in quantiles], color='r', marker='x', label="Linear PDF at Quantiles")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif case == 3:
        # Case 3
        plt.figure(figsize=(11, 6))
        plt.xlim(-4.5, 4.5)
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("PDF Estimation of N(0,1) via 9 quantiles with different interpolation methods")

        pdf_values = {
        "Pchip Interpolation": np.array([pdf_pchip(x) for x in x_values]),
        "Pchip Interpolation + Normal Tails Interpolation": np.array([pdf_hybrid(x) for x in x_values]),
        "True PDF": norm.pdf(x_values, loc=mu, scale=sigma)
        }

        for label, values in pdf_values.items():
            plt.plot(x_values, values, label=label, linestyle="--")

        plt.scatter(quantiles, [pdf_hybrid(q) for q in quantiles], color='r', marker='x', label="Pchip interpolation+ normal tails pdf at quantiles")

        plt.legend()
        plt.grid(True)
        plt.show()


import numpy as np
from scipy.integrate import quad

def evaluate(quantiles, probabilities, y, y_min, y_max):
    """++++++
    Evaluates CRPS and NLL for a single y value and stores them in separate arrays for each interpolation method.

    Parameters:
        quantiles (array-like): The predicted quantiles.
        probabilities (array-like): The corresponding probabilities.
        y (float): The true value to evaluate.
        y_min (float): Minimum y value for integration.
        y_max (float): Maximum y value for integration.

    Returns:
        tuple: Arrays containing CRPS and NLL values for each method.
    """

    # Compute interpolation functions
    cdf_linear, cdf_pchip, hybrid_cdf, cdf_normal, pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid, pdf_normal = compute_cdf_pdf_interpolators(
        quantiles, probabilities, y_min, y_max
    )

    # Arrays to store CRPS and NLL values for each method
    cdf_linear_a = []
    cdf_pchip_a = []
    hybrid_cdf_a = []

    cdf_normal_a = []

    pdf_linear_a = []
    pdf_linear2_a = []
    pdf_pchip_a = []
    pdf_hybrid_a = []

    pdf_normal_a = []

    # Compute CRPS for each CDF method
    cdf_methods = [cdf_linear, cdf_pchip, hybrid_cdf]
    cdf_arrays = [cdf_linear_a, cdf_pchip_a, hybrid_cdf_a]

    #cdf_methods = [hybrid_cdf]
    #cdf_arrays = [hybrid_cdf_a]


    # closed formula for normal pdf, cdf
    cdf_normal, pdf_normal, mu, sigma = compute_cdf_pdf_normal_distribution(quantiles, probabilities)
    z = (y - mu) / sigma
    crps_gaussian.append(
        sigma * (z * (2 * cdf_normal(z) - 1) + 2 * pdf_normal(z) - 1 / np.sqrt(np.pi)))
    cdf_normal_a.append(cdf_normal)
    pdf_normal_a.append(pdf_normal)

    for cdf, cdf_array in zip(cdf_methods, cdf_arrays):
        integrand = lambda x: (cdf(x) - (x >= y))**2
        crps_value, _ = quad(integrand, y_min, y_max)
        cdf_array.append(crps_value)
    
    # Compute NLL for each PDF method
    pdf_methods = [pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid]
    pdf_arrays = [pdf_linear_a, pdf_linear2_a, pdf_pchip_a, pdf_hybrid_a]
    #pdf_methods = [pdf_hybrid]
    #pdf_arrays = [pdf_hybrid_a]


    for pdf, pdf_array in zip(pdf_methods, pdf_arrays):
        pdf_value = pdf(y)
        nll_value = -np.log(pdf_value)
        pdf_array.append(nll_value)

    return cdf_linear_a, cdf_pchip_a, hybrid_cdf_a, pdf_linear_a, pdf_linear2_a, pdf_pchip_a, pdf_hybrid_a
    #return hybrid_cdf_a, pdf_hybrid_a
