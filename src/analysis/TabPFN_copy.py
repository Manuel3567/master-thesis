import numpy as np
import scipy.interpolate as spi
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from analysis.splits import to_train_validation_test_data
from analysis.transformations import scale_power_data, add_lagged_features, add_interval_index
from tabpfn import TabPFNRegressor
from scipy.integrate import quad
import numpy as np
from scipy.stats import lognorm


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

    train, validation, test = to_train_validation_test_data(entsoe, "2016-03-31 23:45:00", "2016-06-30 23:45:00")

    if case == 1:
            feature_columns = ['power_t-96']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "P"
    
    elif case == 2:
            feature_columns = ['power_t-96']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "P"


    elif case == 3:
            feature_columns = ['ws_10m_loc_mean', 'ws_100m_loc_mean']
            output_file = f'{output_file}case{case}.xlsx'
            feature_abbr = "ws_mean"

    
    elif case == 4:
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
    
    model = TabPFNRegressor(device='auto', ignore_pretraining_limits=True, fit_mode='low_memory', random_state=random_state)
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
    return X_validation, y_validation, quantiles_custom, all_quantiles


def compute_cdf_pdf_interpolators(quantiles, probabilities, y_min=-30, y_max=30, mu_left_asym=-1.72, sigma_left_asym=1.45, mu_right_asym=-1.65, sigma_right_asym=0.79):
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
    - pdf_hybrid(x): Hybrid PDF (PCHIP + normal tails) with a minimum sigma left.
    """
    sigma_left_min = sigma_left_asym
    sigma_right_min = sigma_right_asym
    mu_left_min = mu_left_asym
    mu_right_min = mu_right_asym

    lambda_val  = 1.5
    lambda_val_R  = 4.

    # Extend quantile and probability arrays
    full_quantiles = np.concatenate(([y_min], quantiles, [y_max]))
    full_probabilities = np.concatenate(([0], probabilities, [1]))

    # --- Calculate minimum difference between consecutive quantiles ---
    delta_quantiles = np.diff(quantiles)
    min_delta_quantile = np.min(delta_quantiles)

    # Fit left and right tails
    mu_left, sigma_left = fit_tail_distribution(quantiles[:2], probabilities[:2])
    mu_right, sigma_right = fit_tail_distribution(quantiles[-2:], probabilities[-2:])
    #sigma_left = max(sigma_left, sigma_left_min)

    #print(mu_left, sigma_left, mu_right, sigma_right)

    # --- Define CDF Interpolators ---
    cdf_linear_interpolator = spi.interp1d(
        full_quantiles, full_probabilities, kind="linear", fill_value=(0, 1), bounds_error=False
    )
    cdf_pchip_interpolator = spi.PchipInterpolator(full_quantiles, full_probabilities, extrapolate=True)
    #cdf_pchip_interpolator = spi.interp1d(quantiles, probabilities, kind='quadratic', fill_value=(0, 1), bounds_error=False)
    #cdf_pchip_interpolator = spi.Akima1DInterpolator(full_quantiles, full_probabilities)

    def cdf_linear(x):
        """Linear interpolation CDF function."""
        cdf_linear = float(np.clip(cdf_linear_interpolator(x), 0, 1))
        return cdf_linear

    def cdf_pchip(x):
        """PCHIP interpolation CDF function."""
        cdf_pchip = float(np.clip(cdf_pchip_interpolator(x), 0, 1))
        return cdf_pchip

    def hybrid_cdf(x):
        """
        Hybrid CDF:
        - Left normal fit for x < first quantile
        - PCHIP interpolation for middle range
        - Right normal fit for x > last quantile
        """
        if x < quantiles[0]:  # Left tail
            #adjusted_sigma_left = sigma_left
            #if sigma_left < sigma_left_min:
            adjusted_sigma_left = sigma_left_min + (sigma_left - sigma_left_min) * np.exp(- np.abs((x - quantiles[0])) / lambda_val)
            adjusted_mu_left = mu_left_min + (mu_left - mu_left_min) * np.exp(- np.abs((x - quantiles[0])) / lambda_val)
            return norm.cdf(x, loc=adjusted_mu_left, scale=adjusted_sigma_left)
        
        elif x > quantiles[-1]:  # Right tail
            #adjusted_sigma_right = sigma_right
            #if sigma_right < sigma_right_min:
            adjusted_sigma_right = sigma_right_min + (sigma_right - sigma_right_min) * np.exp(- np.abs((x - quantiles[-1])) / lambda_val_R)   
            adjusted_mu_right = mu_right_min + (mu_right - mu_right_min) * np.exp(- np.abs((x - quantiles[-1])) / lambda_val_R)
            return norm.cdf(x, loc=adjusted_mu_right, scale=adjusted_sigma_right)
        
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
        #eps = min_delta_quantile/2
        eps = max(min_delta_quantile / 2, 1e-8)  # Avoid zero or extremely small eps
        #print(f"epsilon {eps}")
        #print("Step cdf_chip",cdf_pchip_interpolator(x + eps),cdf_pchip_interpolator(x - eps))

        return (cdf_pchip_interpolator(x + eps) - cdf_pchip_interpolator(x - eps)) / (2 * eps)
        #return cdf_pchip_interpolator.derivative()(x)

    def pdf_hybrid(x):
        """
        Hybrid PDF:
        - Left normal distribution for (x < first quantile) based on first two quantiles
        - PCHIP interpolation derivative for middle range
        - Right normal distribution for (x > last quantile) based on last two quantiles
        """
        # nach vorne den Teil        
        if x < quantiles[0]:  # Left tail
            adjusted_sigma_left = sigma_left_min + (sigma_left - sigma_left_min) * np.exp(- np.abs((x - quantiles[0])) / lambda_val)
            adjusted_mu_left = mu_left_min + (mu_left - mu_left_min) * np.exp(- np.abs((x - quantiles[0])) / lambda_val)
            pdf_value_L = norm.pdf(x, loc=adjusted_mu_left, scale=adjusted_sigma_left)
            return pdf_value_L
        
        elif x > quantiles[-1]:  # Right tail
            adjusted_sigma_right = sigma_right_min + (sigma_right - sigma_right_min) * np.exp(- np.abs((x - quantiles[-1])) / lambda_val_R)   
            adjusted_mu_right = mu_right_min + (mu_right - mu_right_min) * np.exp(- np.abs((x - quantiles[-1])) / lambda_val_R)
            pdf_value_R = norm.pdf(x, loc=adjusted_mu_right, scale=adjusted_sigma_right)
            return pdf_value_R
         
        elif x > (quantiles[-2] + quantiles[-1])/2:  # Right tail
            adjusted_sigma_right = sigma_right_min + (sigma_right - sigma_right_min) * np.exp(- np.abs((x - quantiles[-1])) / lambda_val_R)   
            adjusted_mu_right = mu_right_min + (mu_right - mu_right_min) * np.exp(- np.abs((x - quantiles[-1])) / lambda_val_R)
            pdf_value_R = norm.pdf(x, loc=adjusted_mu_right, scale=adjusted_sigma_right)
            eps = max(min_delta_quantile / 2, 1e-5)  # Avoid zero or extremely small eps
            #return cdf_pchip_interpolator.derivative()(x)
            pdf_value_M = (cdf_pchip_interpolator(x + eps) - cdf_pchip_interpolator(x - eps)) / (2 * eps) 
            return np.max([pdf_value_R, pdf_value_M])
              
        elif x < (quantiles[0] + quantiles[1])/2:  # Left tail
            adjusted_sigma_left = sigma_left_min + (sigma_left - sigma_left_min) * np.exp(- np.abs((x - quantiles[0])) / lambda_val)
            adjusted_mu_left = mu_left_min + (mu_left - mu_left_min) * np.exp(- np.abs((x - quantiles[0])) / lambda_val)
            pdf_value_L = norm.pdf(x, loc=adjusted_mu_left, scale=adjusted_sigma_left)
            eps = max(min_delta_quantile / 2, 1e-5)  # Avoid zero or extremely small eps
            #return cdf_pchip_interpolator.derivative()(x)
            pdf_value_M = (cdf_pchip_interpolator(x + eps) - cdf_pchip_interpolator(x - eps)) / (2 * eps) 
            return np.max([pdf_value_L, pdf_value_M])
        
        else:  # Middle range (PCHIP interpolation)
            eps = max(min_delta_quantile / 2, 1e-5)  # Avoid zero or extremely small eps
            #return cdf_pchip_interpolator.derivative()(x)
            pdf_value_M = (cdf_pchip_interpolator(x + eps) - cdf_pchip_interpolator(x - eps)) / (2 * eps) 
            return pdf_value_M

    return cdf_linear, cdf_pchip, hybrid_cdf, pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid

# --- Fit Normal Distributions for Tails ---
def fit_tail_distribution(quantiles, probabilities):
        """Fits a normal distribution to the given quantiles and associated probabilities."""
        z_scores = norm.ppf(probabilities)
        sigma = (quantiles[1] - quantiles[0]) / (z_scores[1] - z_scores[0])
        mu = quantiles[0] - sigma * z_scores[0]
        return mu, sigma



def evaluate(quantiles, probabilities, y, y_min, y_max, mu_left_asym, sigma_left_asym, mu_right_asym, sigma_right_asym):
    """
    Evaluates CRPS and NLL for a single y value, storing results for only linear and hybrid methods.

    Parameters:
        quantiles (array-like): The predicted quantiles.
        probabilities (array-like): The corresponding probabilities.
        y (float): The true value to evaluate.
        y_min (float): Minimum y value for integration.
        y_max (float): Maximum y value for integration.

    Returns:
        tuple: Linear CRPS, Hybrid CRPS, normal CRPS, Linear NLL, Hybrid NLL, and normal NLL.
    """
    
    # Compute interpolation functions (CDFs and PDFs)
    cdf_linear, _, hybrid_cdf, pdf_linear, _, _, pdf_hybrid = compute_cdf_pdf_interpolators(
        quantiles, probabilities, y_min, y_max, mu_left_asym, sigma_left_asym, mu_right_asym, sigma_right_asym
    )

    # Compute CRPS for linear and hybrid CDF methods
    cdf_methods = [cdf_linear, hybrid_cdf]
    crps_values = []

    for cdf in cdf_methods:
        integrand = lambda x: (cdf(x) - (x >= y))**2
        crps_value, _ = quad(integrand, y_min, y_max)
        crps_values.append(crps_value)
    
      # Compute NLL for linear and hybrid PDF methods
    pdf_methods = [pdf_linear, pdf_hybrid]
    nll_values = []

    for pdf in pdf_methods:
        pdf_value = pdf(y)
        nll_value = -np.log(pdf_value)
        nll_values.append(nll_value)

    _, _, mu, sigma = compute_cdf_pdf_normal_distribution(quantiles, probabilities)
    #print("mu", mu)
    #print("sigma", sigma)
    z = (y - mu) / sigma
    crps_normal = sigma * ( 
            z * (2 * norm.cdf(z) - 1)
            + 2 * norm.pdf(z) 
            - 1/np.sqrt(np.pi)
    )
    crps_values.append(crps_normal)

    nll_normal = - norm.logpdf(y, loc=mu, scale=sigma)
    nll_values.append(nll_normal)

    return crps_values[0], crps_values[1], crps_values[2], nll_values[0], nll_values[1], nll_values[2]


def fit_normal_dist_to_quantiles(quantiles, probabilities):
    """ Fits a normal distribution to given quantiles and probabilities. """
  
    # Provide a reasonable initial guess
    mu_init = np.mean(quantiles)
    sigma_init = (np.max(quantiles) - np.min(quantiles)) / 4  # Rough estimate

    try:
        params, _ = curve_fit(normal_cdf_inverse, probabilities, quantiles, p0=[mu_init, sigma_init])
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        return None, None

    mu_fit, sigma_fit = params
    #print(f"Estimated Mean: {mu_fit}, Estimated Std Dev: {sigma_fit}")
    
    return mu_fit, sigma_fit


def normal_cdf_inverse(p, mu, sigma):
    """Inverse CDF for normal distribution."""
    return norm.ppf(p, loc=mu, scale=sigma)

def compute_cdf_pdf_normal_distribution(quantiles, probabilities):
    """Compute CDF and PDF for a normal distribution fitted to quantiles."""
    mu, sigma = fit_normal_dist_to_quantiles(quantiles, probabilities)

    def cdf_normal(x):
        return norm.cdf(x, mu, sigma)

    def pdf_normal(x):
        return norm.pdf(x, mu, sigma)

    return cdf_normal, pdf_normal, mu, sigma


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_cdf_pdf_dynamic(ax, quantiles, probabilities, y_min, y_max, x_lim, y_lim, log_scale, eps=0.01, case=1, mu_left_asym=-1.72, sigma_left_asym=1.45, mu_right_asym=-1.65, sigma_right_asym=0.79):
    """Plots the CDF and PDF for linear, cubic interpolation, and hybrid interpolation (cubic + normal tails) methods.
    - case 1: CDF of linear, cubic, gaussian between the quantiles, PDF of Linear Interpolation epsilon=0.01, Linear Interpolation epsilon=min_delta_quantile/2, cubic interpolation, gaussian interpolation
    - case 2: CDF of linear, linear y_min={y_min - y_min_shift}, cubic interpolation, cubic + normal tails, gaussian interpolation beyond range of quantiles, PDF of linear Interpolation epsilon=0.01, y_min={y_min - y_min_shift}, Gaussian interpolation beyond range of quantiles
    - case 3: PDF of cubic, cubic + normal tails, Gaussian interpolation beyond range of quantiles
    """
    # Generate x values for plotting
    x_values = np.linspace(y_min - 1, y_max + 1, 18500)

    # Get CDF and PDF functions
    cdf_linear, cdf_pchip, hybrid_cdf, pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid = compute_cdf_pdf_interpolators(
        quantiles, probabilities, y_min, y_max, mu_left_asym, sigma_left_asym, mu_right_asym, sigma_right_asym
    )

    y_min_shift = 1
    cdf_linear_m, cdf_pchip_m, hybrid_cdf_m, pdf_linear_m, pdf_linear2_m, pdf_pchip_m, pdf_hybrid_m = compute_cdf_pdf_interpolators(
    quantiles, probabilities, y_min - y_min_shift, y_max + y_min_shift, mu_left_asym, sigma_left_asym, mu_right_asym, sigma_right_asym
    )
    
    mu, sigma = fit_normal_dist_to_quantiles(quantiles, probabilities)

    plt.figure(figsize=(11, 6))
    if case == 1:
        # Case 1
        plt.figure(figsize=(11, 6))
        plt.xlim(quantiles[0] - 0.1, quantiles[-1] + 0.1)
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF Estimation of one set of quantiles of TabPFN with different interpolation methods")
        cdf_values = {
            "Linear Interpolation": (np.array([cdf_linear(x) for x in x_values]), "blue", "-"),
            #"Akima Spline Interpolation": np.array([cdf_pchip(x) for x in x_values]),
            "Pchip Interpolation": (np.array([cdf_pchip(x) for x in x_values]), "orange", "-"),
            "Gaussian interpolation": (norm.cdf(x_values, loc=mu, scale=sigma), "green", "--")
        }
        
        #for label, values in cdf_values.items():
        #    plt.plot(x_values, values, label=label)

        for label, (values, color, linestyle) in cdf_values.items():
            plt.plot(x_values, values, label=label, color=color, linestyle=linestyle, linewidth=1.3)

        plt.scatter(quantiles, probabilities, color='y', marker='o', label="Quantiles")
        plt.legend()
        plt.grid(True)
        #plt.show()


        plt.figure(figsize=(11, 6))
        plt.xlim(quantiles[0] - 0.1, quantiles[-1] + 0.1)
        #plt.ylim(0, 1.5)
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("PDF Estimation of one set of quantiles of TabPFN with different interpolation methods")

        pdf_values = {
            f"Linear Interpolation epsilon=0.01": (np.array([pdf_linear(x) for x in x_values]), "blue", "-"),
            "Linear Interpolation epsilon=min_delta_quantile/2": (np.array([pdf_linear2(x) for x in x_values]), "lightblue", "-"),
            "Pchip Interpolation": (np.array([pdf_pchip(x) for x in x_values]), "orange", "-"),
            "Gaussian interpolation": (norm.pdf(x_values, loc=mu, scale=sigma), "green", "--")
        }

        for label, (values, color, linestyle) in pdf_values.items():
            plt.plot(x_values, values, label=label, color=color, linestyle=linestyle, linewidth=1.3)

        plt.scatter(quantiles, [pdf_linear(q) for q in quantiles], marker='x', label="Linear PDF at Quantiles", color="blue")
        plt.scatter(quantiles, [pdf_pchip(q) for q in quantiles], marker='o', label="Pchip PDF at Quantiles", color="orange")
        print(f"Quantiles: {quantiles}, PDFs: {np.array([pdf_pchip(q) for q in quantiles])}")
        plt.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
        plt.grid(True)
        #plt.show()

    elif case == 2:

        plt.figure(figsize=(11, 6))
        #plt.xlim(-4, 4)
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.yscale("log")
        plt.ylim(1e-5)
        plt.title("CDF Estimation of one set of quantiles of TabPFN with different interpolation methods")

        cdf_values = {
            f"Linear Interpolation, y_min={y_min}": (np.array([cdf_linear(x) for x in x_values]), "blue", "-"),
            f"Linear Interpolation, y_min={y_min - y_min_shift}": (np.array([cdf_linear_m(x) for x in x_values]), "purple", "-"),
            "Pchip Interpolation": (np.array([cdf_pchip(x) for x in x_values]), "orange", "-"),
            "Pchip Interpolation + Normal Tails Interpolation": (np.array([hybrid_cdf(x) for x in x_values]), "red", "-"),
            "Gaussian interpolation": (norm.cdf(x_values, loc=mu, scale=sigma), "green", "--")
        }

        for label, (values, color, linestyle) in cdf_values.items():
            plt.plot(x_values, values, label=label, color=color, linestyle=linestyle, linewidth=1.3)
        plt.vlines(y_min, -0.1, 0.4, linestyle="--", linewidth=0.9, color="blue")
        plt.vlines(y_min - y_min_shift, -0.1, 0.4, linestyle="--", linewidth=0.9, color="purple")
        plt.scatter(quantiles, probabilities, color='y', marker='o', label="Quantiles")
        plt.legend()
        plt.grid(True)
        #plt.show()

        pdf_values = {
        f"Linear Interpolation epsilon=0.01, y_min={y_min}": (np.array([pdf_linear(x) for x in x_values]), "blue", "-"),
        f"Linear Interpolation epsilon=0.01, y_min={y_min - y_min_shift}": (np.array([pdf_linear_m(x) for x in x_values]), "purple", "-"),
        "Gaussian interpolation": (norm.pdf(x_values, loc=mu, scale=sigma), "green", "--")
        }
       
        plt.figure(figsize=(11, 6))
        plt.xlabel("Value")
        plt.ylabel("Probability Density")

        plt.title("PDF Estimation of one set of quantiles of TabPFN with different interpolation methods")

        for label, (values, color, linestyle) in pdf_values.items():
            plt.plot(x_values, values, label=label, color=color, linestyle=linestyle, linewidth=1.3)
        plt.scatter(quantiles, [pdf_linear(q) for q in quantiles], color='blue', marker='x', label="Linear PDF at Quantiles")
        plt.legend()
        plt.grid(True)
        #plt.show()


    elif case == 3:
        ax.set_xlim(x_lim)  # Use externally defined limits
        ax.set_ylim(y_lim)  
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.set_title("PDF Estimation of One Set of Quantiles")
        ax.set_yscale(log_scale)
        #ax.set_yscale("log")


        # Case 3
        #plt.figure(figsize=(11, 6))
        #plt.xlim(-3, 0)
        #plt.xlabel("Value")
        #plt.ylabel("Probability Density")
        #plt.title("PDF Estimation of one set of quantiles of TabPFN with different interpolation methods")
        #plt.yscale("log")

        pdf_values = {
        #"Pchip Interpolation": (np.array([pdf_pchip(x) for x in x_values]), "orange", "-"),
        "Pchip Interpolation + Normal Tails Interpolation": (np.array([pdf_hybrid(x) for x in x_values]), "black", "-"),
        #"Gaussian interpolation": (norm.pdf(x_values, loc=mu, scale=sigma), "green", "--")
        }

        for label, (values, color, linestyle) in pdf_values.items():
            ax.plot(x_values, values, label=label, color=color, linestyle=linestyle, linewidth=1.3)

        ax.scatter(quantiles, [pdf_hybrid(q) for q in quantiles], color='black', marker='x', label="Pchip interpolation + normal tails pdf at quantiles")

        ax.legend()
        ax.grid(True)
        #plt.show()

import torch
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

def plot_pdf_from_logits(ax, logits, borders, x_lim, y_lim, log_scale, id=0):
     

    # Assuming logits_2 is a tensor of shape (N, 5000)
    probs_t = torch.softmax(logits, dim=1)  # Convert logits to probabilities (N, 5000)
    probs_np = probs_t[id, :].cpu().numpy()  # Convert first sample to NumPy (1D array)

    # Convert bin borders to NumPy if necessary
    borders_np = borders.cpu().numpy() if isinstance(borders, torch.Tensor) else np.array(borders)

    # Compute bin widths (size 5000)
    bin_widths = np.diff(borders_np)  # Difference between consecutive bin edges

    # Compute PDF values (divide probability mass by bin width)
    pdf_values = probs_np / bin_widths  # Element-wise division

    # Compute bin midpoints
    midpoints = (borders_np[:-1] + borders_np[1:]) / 2  # Midpoints of bins

    # Create an interpolation function using midpoints
    pdf_function_linear = spi.interp1d(
    midpoints, pdf_values, kind='linear', fill_value="extrapolate", bounds_error=False
    )

    pdf_function_pchip = spi.CubicSpline(
    midpoints, pdf_values, extrapolate=True)

    # Generate x values for plotting
    x_plot = np.linspace(midpoints[0], midpoints[-1], 5000)  # Smooth x range
    y_plot = pdf_function_pchip(x_plot)  # Evaluate the PDF function
    y_plot_2 = pdf_function_linear(x_plot)  # Evaluate the PDF function

    # Plot the PDF
    ax.set_xlim(x_lim)  # Use externally defined limits
    ax.set_ylim(y_lim)  
    ax.plot(x_plot, y_plot, label="Interpolated PDF (cubic) for 5000 logits", color="blue")
    ax.plot(x_plot, y_plot_2, label="Interpolated PDF (linear) for 5000 logits", color="orange")
    ax.scatter(midpoints, pdf_values, color="red", s=5, label="Original PDF Points at 5000 logits")  
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    #ax.set_yscale("log")
    ax.set_yscale(log_scale)

    #plt.figure(figsize=(11, 6))
    #ax.plot(x_plot, y_plot, label="Interpolated PDF (cubic) for 5000 logits")
    #ax.plot(x_plot, y_plot_2, label="Interpolated PDF (linear) for 5000 logits")
    #ax.scatter(midpoints, pdf_values, s=5, label="Original PDF Points at 5000 logits")  # Show original data
    #ax.legend()

    #ax.set_xlabel("x")
    #ax.set_yscale("log")
    #ax.set_xlim(-3, 0)
    #ax.set_ylabel("PDF")
    #ax.set_title("Probability Density Function (PDF) using Bin Midpoints")
    #ax.grid(True)
    #plt.show()







