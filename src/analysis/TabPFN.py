import numpy as np
import scipy.interpolate as spi
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import quad



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
    #cdf_pchip_interpolator = spi.PchipInterpolator(full_quantiles, full_probabilities, extrapolate=True)
    #cdf_pchip_interpolator = spi.interp1d(quantiles, probabilities, kind='quadratic', fill_value=(0, 1), bounds_error=False)
    cdf_pchip_interpolator = spi.Akima1DInterpolator(full_quantiles, full_probabilities)


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
            "Akima Spline Interpolation": np.array([cdf_pchip(x) for x in x_values]),
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
            "Akima Spline Interpolation": np.array([pdf_pchip(x) for x in x_values]),
            "True PDF": norm.pdf(x_values, loc=mu, scale=sigma)
        }
        for label, values in pdf_values.items():
            plt.plot(x_values, values, label=label, linestyle="--")
        plt.scatter(quantiles, [pdf_linear(q) for q in quantiles], marker='x', label="Linear PDF at Quantiles")
        plt.scatter(quantiles, [pdf_pchip(q) for q in quantiles], marker='o', label="Akima Spline PDF at Quantiles", color='g')
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
            "Akima Spline Interpolation": np.array([cdf_pchip(x) for x in x_values]),
            "Akima Spline + Normal Tails Interpolation": np.array([hybrid_cdf(x) for x in x_values]),
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
        "Akima Spline Interpolation": np.array([pdf_pchip(x) for x in x_values]),
        "Akima Spline + Normal Tails Interpolation": np.array([pdf_hybrid(x) for x in x_values]),
        "True PDF": norm.pdf(x_values, loc=mu, scale=sigma)
        }

        for label, values in pdf_values.items():
            plt.plot(x_values, values, label=label)

        plt.scatter(quantiles, [pdf_hybrid(q) for q in quantiles], color='r', marker='x', label="Akima spline + normal tails pdf at quantiles")

        plt.legend()
        plt.grid(True)
        plt.show()


def evaluate(quantiles, probabilities, y, y_min, y_max):

    cdf_linear, cdf_pchip, hybrid_cdf, pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid = compute_cdf_pdf_interpolators(
        quantiles, probabilities, y_min, y_max
    )
    cdfs = [cdf_linear, cdf_pchip, hybrid_cdf]
    pdfs = [pdf_linear, pdf_linear2, pdf_pchip, pdf_hybrid]

    crps = []
    nlls = []

    for cdf in cdfs:
        integrand = lambda x: (cdf(x) - (x >= y))**2
        crps_value, _ = quad(integrand, y_min, y_max)
        crps.append(crps_value)

    for pdf in pdfs:
        pdf_value = pdf(y)
        negative_log_pdf = - np.log(pdf_value)
        nlls.append(negative_log_pdf)
    
    return (
        ('crps_cdf_linear', crps[0]),
        ('crps_cdf_pchip', crps[1]),
        ('crps_hybrid_cdf', crps[2]),
        ('nll_pdf_linear (eps = 0.01)', nlls[0]),
        ('nll_pdf_linear2 (eps = min_delta_quantile/2)', nlls[1]),
        ('nll_pdf_pchip', nlls[2]),
        ('nll_pdf_hybrid', nlls[3]),
    )