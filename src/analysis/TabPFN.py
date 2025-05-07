from dataclasses import dataclass
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import rv_continuous, rv_histogram
from scipy.interpolate import make_interp_spline
from scipy.special import softmax
from tabpfn import TabPFNRegressor
from scipy.integrate import quad
from scipy.optimize import curve_fit


import numpy as np
import pandas as pd
from analysis.splits import to_train_validation_test_data
from analysis.datasets import load_entsoe
from analysis.preprocessor import *
from analysis.experiment_mapper import *


class CRPSMixin:
    def crps(self, y, smoothing=True, window_size=101):
        """
        Calculate the Continuous Ranked Probability Score (CRPS) for a given sample `y`.
        
        Parameters:
        - y: np.array of shape (n,): Values where CRPS is to be computed.
        
        Returns:
        - crps_score: The CRPS score.
        """

        if smoothing:
            pdfs = self.probits / self.bin_widths
            window = np.ones(window_size) / window_size
            smooth_pdf = np.convolve(pdfs, window, mode='same')
            probs = self.bin_widths * smooth_pdf
        else:
            probs = self.probits
        
        cdfs = np.cumsum(probs)

        indicator_f = lambda x: x >= y[:, np.newaxis]
        indicators = indicator_f(self.borders[1:])

        crps_scores = np.sum((cdfs - indicators)**2 * self.bin_widths, axis=1)

        return crps_scores
    
    def crps_normal(mu, sigma, y):
        """
        Compute the CRPS for a normal distribution with given mean (mu), std (sigma), and target y.
        Closed-form solution.
        """
        z = (y - mu) / sigma
        return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


class HistogramBase:
    def __init__(self, logits, borders, *args, **kwargs):
        """
        Args:
        - logits (np.ndarray): An array of shape (5000,) representing logits.
        - borders (np.ndarray): An array of shape (5001,) representing the bin edges for each sample.
        """
        self.logits = logits
        self.probits = softmax(logits) # == np.exp(x)/sum(np.exp(x))
        self.borders = borders
        self.bin_widths = self.get_bin_widths(borders)
        self.bin_midpoints = self.get_midpoints(borders)
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_bin_widths(borders):
        return (borders[1:] - borders[:-1])

    @staticmethod
    def get_midpoints(borders):
        """
        input borders: np.array[5001]
        output midpoints: np.array[5000]
        """
        midpoints = (borders[1:] + borders[:-1]) / 2
        return midpoints
    
class LinearInterpolatedDist(rv_continuous):
    def __init__(self, x, pdf_values, *args, **kwargs):
        self.x = x
        self.pdf_values = pdf_values
        super().__init__(*args, **kwargs)

    def _pdf(self, x):
        pdf = make_interp_spline(x, self.pdf_values, k=1)
        return pdf(x)


class LinearInterpolatedHistWithCRPS(HistogramBase, rv_continuous, CRPSMixin):
    
    def _pdf(self, x):
        pdf = make_interp_spline(self.bin_midpoints, self.probits/self.bin_widths, k=1)
        return pdf(x)


class HistogramWithCRPS(HistogramBase, rv_histogram, CRPSMixin):
    def __init__(self, logits, borders, *args, **kwargs):
        probits = softmax(logits)
        super().__init__(logits=logits, borders=borders, histogram=(probits, borders), density=False, *args, **kwargs)

class HistogramWithCRPSSmoothed(HistogramBase, rv_histogram, CRPSMixin):
    def __init__(self, logits, borders, window_size=5, *args, **kwargs):
        probits = softmax(logits)
        bin_widths = borders[1:] - borders[:-1]
        pdfs = probits / bin_widths
        #window_size = 5
        window = np.ones(window_size) / window_size
        smooth_pdf = np.convolve(pdfs, window, mode='same')
        super().__init__(logits=logits, borders=borders, histogram=(smooth_pdf, borders), density=True, *args, **kwargs)

import scipy.interpolate as spi
from scipy.stats import norm

class CDFPDFInterpolator:
#    def __init__(self, quantiles, probabilities, y_min=-30, y_max=30,
#                 mu_left_asym=-1.72, sigma_left_asym=1.45, mu_right_asym=-1.65, sigma_right_asym=0.79):
        
    def __init__(self, quantiles, probabilities, y_min=-20, y_max=5,
                  mu_left_asym=-1.718314259036157, sigma_left_asym=1.4459376285208483, mu_right_asym=-1.646182571554056, sigma_right_asym=0.7849568217992877):

        self.quantiles = quantiles
        self.probabilities = probabilities
        self.y_min = y_min
        self.y_max = y_max
        self.mu_left_asym = mu_left_asym
        self.sigma_left_asym = sigma_left_asym
        self.mu_right_asym = mu_right_asym
        self.sigma_right_asym = sigma_right_asym
        self.mu_normal, self.sigma_normal = self._fit_full_normal_distribution(self.quantiles, self.probabilities)


        self.full_quantiles = np.concatenate(([self.y_min], self.quantiles, [self.y_max]))

        self.full_probabilities = np.concatenate(([0], self.probabilities, [1]))
        
        self._initialize_interpolators()
    
    def _initialize_interpolators(self):

        self.cdf_linear_interpolator = spi.interp1d(
            self.full_quantiles, self.full_probabilities, kind="linear", fill_value=(0, 1), bounds_error=False
        )
        
        self.cdf_pchip_interpolator = spi.PchipInterpolator(
            self.full_quantiles, self.full_probabilities, extrapolate=True
        )
        
        self.min_delta_quantile = np.min(np.diff(self.quantiles))
    
    def cdf_linear(self, x):
        return float(np.clip(self.cdf_linear_interpolator(x), 0, 1))
        #return np.clip(self.cdf_linear_interpolator(x), 0, 1)
    
    def cdf_pchip(self, x):
        return float(np.clip(self.cdf_pchip_interpolator(x), 0, 1))
        #return np.clip(self.cdf_pchip_interpolator(x), 0, 1)
    
    def pdf_linear(self, x, eps=0.01):
        return (self.cdf_linear(x + eps) - self.cdf_linear(x - eps)) / (2 * eps)
    
    def pdf_pchip(self, x):
        eps = max(self.min_delta_quantile / 2, 1e-8)
        return (self.cdf_pchip_interpolator(x + eps) - self.cdf_pchip_interpolator(x - eps)) / (2 * eps)
    
    def hybrid_cdf(self, x):
        lambda_val, lambda_val_R = 4e0, 4.0
        mu_left, sigma_left = self._fit_tail_distribution(self.quantiles[:2], self.probabilities[:2])
        mu_right, sigma_right = self._fit_tail_distribution(self.quantiles[-2:], self.probabilities[-2:])

        if x < self.quantiles[0]:
            adjusted_sigma = self.sigma_left_asym + (sigma_left - self.sigma_left_asym) * np.exp(-np.abs((x - self.quantiles[0])) / lambda_val)
            adjusted_mu = self.mu_left_asym + (mu_left - self.mu_left_asym) * np.exp(-np.abs((x - self.quantiles[0])) / lambda_val)
            return norm.cdf(x, loc=adjusted_mu, scale=adjusted_sigma)
        
        elif x > self.quantiles[-1]:
            adjusted_sigma = self.sigma_right_asym + (sigma_right - self.sigma_right_asym) * np.exp(-np.abs((x - self.quantiles[-1])) / lambda_val_R)
            adjusted_mu = self.mu_right_asym + (mu_right - self.mu_right_asym) * np.exp(-np.abs((x - self.quantiles[-1])) / lambda_val_R)
            return norm.cdf(x, loc=adjusted_mu, scale=adjusted_sigma)
        else:
            return float(np.clip(self.cdf_pchip_interpolator(x), 0, 1))
            #return np.clip(self.cdf_pchip_interpolator(x), 0, 1)
    

    def pdf_hybrid(self, x):
        """
        Hybrid PDF:
        - Left normal distribution for (x < first quantile) based on first two quantiles
        - PCHIP interpolation derivative for middle range
        - Right normal distribution for (x > last quantile) based on last two quantiles
        """
        lambda_val = 4e0
        lambda_val_R = 4.0

        # Ensure the tail parameters are computed dynamically
        mu_left, sigma_left = self._fit_tail_distribution(self.quantiles[:2], self.probabilities[:2])
        mu_right, sigma_right = self._fit_tail_distribution(self.quantiles[-2:], self.probabilities[-2:])

        sigma_left_min = self.sigma_left_asym
        sigma_right_min = self.sigma_right_asym
        mu_left_min = self.mu_left_asym
        mu_right_min = self.mu_right_asym

        if x < self.quantiles[0]:  # Left tail
            #print(f"x < self.quantiles[0]: {x} < {self.quantiles[0]}")
            adjusted_sigma_left = sigma_left_min + (sigma_left - sigma_left_min) * np.exp(-np.abs((x - self.quantiles[0])) / lambda_val)
            adjusted_mu_left = mu_left_min + (mu_left - mu_left_min) * np.exp(-np.abs((x - self.quantiles[0])) / lambda_val)
            
            #print(f"Left Tail -> adjusted_mu: {adjusted_mu_left}, adjusted_sigma: {adjusted_sigma_left}")
            pdf_value = norm.pdf(x, loc=adjusted_mu_left, scale=adjusted_sigma_left)
            #print(f"Left Tail PDF Value: {pdf_value}")
            return pdf_value

        elif x > self.quantiles[-1]:  # Right tail
            #print(f"x > self.quantiles[-1]: {x} > {self.quantiles[-1]}")
            adjusted_sigma_right = sigma_right_min + (sigma_right - sigma_right_min) * np.exp(-np.abs((x - self.quantiles[-1])) / lambda_val_R)
            adjusted_mu_right = mu_right_min + (mu_right - mu_right_min) * np.exp(-np.abs((x - self.quantiles[-1])) / lambda_val_R)
            #print(f"Right Tail -> adjusted_mu: {adjusted_mu_right}, adjusted_sigma: {adjusted_sigma_right}")
            pdf_value = norm.pdf(x, loc=adjusted_mu_right, scale=adjusted_sigma_right)
            #print(f"Right Tail PDF Value: {pdf_value}")
            return pdf_value
        
        elif x > (self.quantiles[-2] + self.quantiles[-1]) / 2:  # Transition to right tail
            #print(f"x > (self.quantiles[-2] + self.quantiles[-1]) / 2: {x} > {(self.quantiles[-2] + self.quantiles[-1]) / 2}")
            adjusted_sigma_right = sigma_right_min + (sigma_right - sigma_right_min) * np.exp(- np.abs((x - self.quantiles[-1])) / lambda_val_R)
            adjusted_mu_right = mu_right_min + (mu_right - mu_right_min) * np.exp(-np.abs((x - self.quantiles[-1])) / lambda_val_R)
            pdf_value_R = norm.pdf(x, loc=adjusted_mu_right, scale=adjusted_sigma_right)
            eps = max(self.min_delta_quantile / 2, 1e-5)
            pdf_value_M = (self.cdf_pchip_interpolator(x + eps) - self.cdf_pchip_interpolator(x - eps)) / (2 * eps)
            #print(f"Right Transition -> PDF Value Right: {pdf_value_R}, PDF Value Middle: {pdf_value_M}")
            return np.max([pdf_value_R, pdf_value_M])

        elif x < (self.quantiles[0] + self.quantiles[1]) / 2:  # Transition to left tail
            #print(f"x < (self.quantiles[0] + self.quantiles[1]) / 2: {x} < {(self.quantiles[0] + self.quantiles[1]) / 2}")
            adjusted_sigma_left = sigma_left_min + (sigma_left - sigma_left_min) * np.exp(- np.abs((x - self.quantiles[0])) / lambda_val)
            adjusted_mu_left = mu_left_min + (mu_left - mu_left_min) * np.exp(-np.abs((x - self.quantiles[0])) / lambda_val)
            pdf_value_L = norm.pdf(x, loc=adjusted_mu_left, scale=adjusted_sigma_left)
            eps = max(self.min_delta_quantile / 2, 1e-5)
            pdf_value_M = (self.cdf_pchip_interpolator(x + eps) - self.cdf_pchip_interpolator(x - eps)) / (2 * eps)
            #print(f"Left Transition -> PDF Value Left: {pdf_value_L}, PDF Value Middle: {pdf_value_M}")
            return np.max([pdf_value_L, pdf_value_M])

        else:  # Middle range (PCHIP interpolation)
            eps = max(self.min_delta_quantile / 2, 1e-5)
            pdf_value = (self.cdf_pchip_interpolator(x + eps) - self.cdf_pchip_interpolator(x - eps)) / (2 * eps)
            #print(f"Middle Region -> PDF Value: {pdf_value}")
            return pdf_value
    
    def fitted_normal_pdf(self, x):
        return norm.pdf(x, loc=self.mu_normal, scale=self.sigma_normal)

    def fitted_normal_cdf(self, x):
        return norm.cdf(x, loc=self.mu_normal, scale=self.sigma_normal)

    def _fit_tail_distribution(self, quantiles, probabilities):
        """Fits a normal distribution to the given quantiles and associated probabilities."""
        z_scores = norm.ppf(probabilities)
        sigma = (quantiles[1] - quantiles[0]) / (z_scores[1] - z_scores[0])
        mu = quantiles[0] - sigma * z_scores[0]
        return mu, sigma
    
    def _fit_full_normal_distribution(self, quantiles, probabilities):
        """Fits a single normal distribution to all quantiles."""
        z_scores = norm.ppf(probabilities)
        A = np.vstack([z_scores, np.ones_like(z_scores)]).T
        sigma, mu = np.linalg.lstsq(A, quantiles, rcond=None)[0]  # linear fit
        return mu, sigma


@dataclass
class Experiment:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    device: str = "auto"
    random_state: int = 42
    fit_mode: str = "low_memory"
    ignore_pretraining_limits: bool = False
    logits: None | np.ndarray = None
    borders: None | np.ndarray = None
    deciles: None | np.ndarray = None

    def __post_init__(self):
        self._cdf_pdf_interpolators = None  # Initialize the attribute

    def perform(self):
        """ Ensure perform() is explicitly called before interpolation. """
        model = TabPFNRegressor(device=self.device, fit_mode=self.fit_mode, random_state=self.random_state, ignore_pretraining_limits=self.ignore_pretraining_limits)
        model.fit(self.X_train, self.y_train)

        quantiles_custom = np.arange(0.1, 1, 0.1)
        probs_val_q = model.predict(self.X_validation, output_type="full", quantiles=quantiles_custom)

        self.logits = probs_val_q["logits"].numpy()
        self.borders = probs_val_q["criterion"].borders.numpy()
        self.deciles = np.array(probs_val_q["quantiles"]).T

        if self.deciles is None or self.deciles.shape[1] != 9:
            raise ValueError(f"Expected deciles to have shape (n, 9), but got {self.deciles.shape}")

        return self
    
    @property
    def hists(self) -> list[HistogramWithCRPS]:
        """
        Create a list of HistogramWithCRPS objects for each of the logits.

        Returns:
        - list: A list of HistogramWithCRPS objects of length n.
        """
        n = self.logits.shape[0]
        histograms = []
        
        # Iterate over the logits (axis 1: 5000 logits)
        for i in range(n):
            hist = HistogramWithCRPS(self.logits[i, :], self.borders)
            histograms.append(hist)
        
        return histograms

    def hists_smoothed(self, window_size=5) -> list[HistogramWithCRPSSmoothed]:
        """
        Create a list of HistogramWithCRPSSmoothed objects for each of the logits.

        Returns:
        - list: A list of HistogramWithCRPSSmoothed objects of length n.
        """
        n = self.logits.shape[0]
        histograms = []
        
        # Iterate over the logits (axis 1: 5000 logits)
        for i in range(n):
            hist = HistogramWithCRPSSmoothed(self.logits[i, :], self.borders, window_size=window_size)
            histograms.append(hist)
        
        return histograms
    
    @property
    def dists(self) -> list[HistogramWithCRPS]:
        """
        Create a list of HistogramWithCRPS objects for each of the logits.

        Returns:
        - list: A list of HistogramWithCRPS objects of length n.
        """
        n = self.logits.shape[0]
        histograms = []
        
        # Iterate over the logits (axis 1: 5000 logits)
        for i in range(n):
            hist = LinearInterpolatedHistWithCRPS(self.logits[i, :], self.borders)
            histograms.append(hist)
        
        return histograms

    @property
    def cdf_pdf_interpolators(self) -> list[CDFPDFInterpolator]:
        """Create a list of CDFPDFInterpolator objects, caching the result."""
        if self._cdf_pdf_interpolators is None:
            if self.deciles is None:
                raise ValueError("Deciles data not available. Run `experiment.perform()` first.")

            if self.deciles.shape[1] != 9:
                raise ValueError(f"Expected deciles to have shape (n, 9), but got {self.deciles.shape}")

            probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

            self._cdf_pdf_interpolators = []

            print("Initializing Interpolator objects...")
            for i in range(self.deciles.shape[0]):
                interpolator = CDFPDFInterpolator(self.deciles[i, :], probabilities)
                self._cdf_pdf_interpolators.append(interpolator)

            print(f"created {len(self._cdf_pdf_interpolators)} CDFPDFInterpolator objects")
        return self._cdf_pdf_interpolators

   
    def calculate_nll(self, data_type='validation', smoothing=False, window_size=11):
        """
        Calculate the Negative Log-Likelihood (NLL) for all histograms of a given experiment.

        Parameters:
        - data_type: 'validation' or 'test'
        - smoothing: If True, use smoothed histograms
        - interpolated: If True, use linearly interpolated histograms

        Returns:
        - total_nll: Total Negative Log-Likelihood value for all histograms
        """
        # Select the appropriate data
        if data_type == 'validation':
            y_data = self.y_validation
        elif data_type == 'test':
            y_data = self.y_test
        else:
            raise ValueError("Invalid data_type, expected 'validation' or 'test'.")

        # Select the right distribution objects
        if smoothing:
            dists = self.hists_smoothed(window_size=window_size)  # HistogramWithCRPSSmoothed
        else:
            dists = self.hists  # HistogramWithCRPS

        # Compute raw NLLs
        nlls = [-dist.logpdf(y) for dist, y in zip(dists, y_data)]

        # Replace infs with the max finite NLL
        finite_nlls = [nll for nll in nlls if np.isfinite(nll)]
        max_finite = max(finite_nlls) if finite_nlls else 0
        cleaned_nlls = [nll if np.isfinite(nll) else max_finite for nll in nlls]

        #min_nll = np.min(cleaned_nlls)
        mean_nll = np.mean(cleaned_nlls)
        #median_nll = np.median(cleaned_nlls)
        #max_nll = np.max(cleaned_nlls)

        return mean_nll
    #, median_nll, min_nll, max_nll
    
    def calculate_nll_quantiles(self, data_type='validation', smoothing=False, window_size=11):
        """
        Calculate the Negative Log-Likelihood (NLL) for all histograms of a given experiment.

        Parameters:
        - data_type: 'validation' or 'test'
        - smoothing: If True, use smoothed histograms
        - interpolated: If True, use linearly interpolated histograms

        Returns:
        - total_nll: Total Negative Log-Likelihood value for all histograms
        """
        # Select the appropriate data
        if data_type == 'validation':
            y_data = self.y_validation
        elif data_type == 'test':
            y_data = self.y_test
        else:
            raise ValueError("Invalid data_type, expected 'validation' or 'test'.")

        # Select the right distribution objects
        if smoothing:
            dists = self.hists_smoothed(window_size=window_size)  # HistogramWithCRPSSmoothed
        else:
            dists = self.hists  # HistogramWithCRPS

        # Compute raw NLLs
        nlls = [-dist.logpdf(y) for dist, y in zip(dists, y_data)]

        # Replace infs with the max finite NLL
        finite_nlls = [nll for nll in nlls if np.isfinite(nll)]
        max_finite = max(finite_nlls) if finite_nlls else 0
        cleaned_nlls = [nll if np.isfinite(nll) else max_finite for nll in nlls]
        nll_quantiles = np.percentile(cleaned_nlls, np.array([5, 25, 75, 95]))
        mean_nll = np.mean(cleaned_nlls)

        return nll_quantiles, mean_nll, cleaned_nlls
    
    def calculate_crps_quantiles(self, data_type='validation', smoothing=False, window_size=5):
        # Select the appropriate data
        if data_type == 'validation':
            y_data = self.y_validation
        elif data_type == 'test':
            y_data = self.y_test
        else:
            raise ValueError("Invalid data_type, expected 'validation' or 'test'.")

        # Select the right distribution objects
        if smoothing:
            dists = self.hists_smoothed(window_size=window_size)  # List of HistogramWithCRPSSmoothed
        else:
            dists = self.hists  # List of HistogramWithCRPS

        # Check length match
        if len(dists) != len(y_data):
            raise ValueError(f"Length mismatch: {len(dists)} dists vs {len(y_data)} y_data")

        # Compute CRPS scores
        #crps_scores = [dist.crps(np.array([y])) for dist, y in zip(dists, y_data)] old
        crps_scores = [dist.crps(np.array([y]), smoothing=smoothing, window_size=window_size) for dist, y in zip(dists, y_data)]
        mean_crps = np.mean(crps_scores)
        crps_quantiles = np.percentile(crps_scores, np.array([5,25,75,95]))


        return crps_quantiles, mean_crps, crps_scores
    
    def calculate_crps(self, data_type='validation', smoothing=False, window_size=5):
        # Select the appropriate data
        if data_type == 'validation':
            y_data = self.y_validation
        elif data_type == 'test':
            y_data = self.y_test
        else:
            raise ValueError("Invalid data_type, expected 'validation' or 'test'.")

        # Select the right distribution objects
        if smoothing:
            dists = self.hists_smoothed(window_size=window_size)  # List of HistogramWithCRPSSmoothed
        else:
            dists = self.hists  # List of HistogramWithCRPS

        # Check length match
        if len(dists) != len(y_data):
            raise ValueError(f"Length mismatch: {len(dists)} dists vs {len(y_data)} y_data")

        # Compute CRPS scores
        #crps_scores = [dist.crps(np.array([y])) for dist, y in zip(dists, y_data)] old
        crps_scores = [dist.crps(np.array([y]), smoothing=smoothing, window_size=window_size) for dist, y in zip(dists, y_data)]

        #min_crps = np.min(crps_scores)
        mean_crps = np.mean(crps_scores)
        #median_crps = np.median(crps_scores)
        #max_crps = np.max(crps_scores)

        return mean_crps
    
    def calculate_crps_quantiles_with_interpolator(self, method="linear"):
        """
        Calculate the CRPS using the specified CDF method from the CDFPDFInterpolator objects.

        Parameters:
        - method (str): Which method to use ('linear', 'pchip', 'hybrid').

        Returns:
        - float: Mean CRPS score for the specified CDF method.
        """
        crps_values = []

        for i, (dist, y) in enumerate(zip(self.cdf_pdf_interpolators, self.y_validation)):
            if method == "normal":
                deciles = self.deciles[i,:]
                probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                # Provide a reasonable initial guess
                mu_init = np.mean(deciles)
                sigma_init = (np.max(deciles) - np.min(deciles)) / 4  # Rough estimate
                #params, _ = curve_fit(lambda x, mu, sigma: norm.cdf(x, loc=mu, scale=sigma), 
                #                  deciles, probabilities, p0=[mu_init, sigma_init])
                params, _ = curve_fit(lambda x, mu, sigma: norm.ppf(x, loc=mu, scale=sigma), 
                    probabilities, deciles, p0=[mu_init, sigma_init])
                mu_fit, sigma_fit = params
                z = (y - mu_fit) / sigma_fit
                crps_value = sigma_fit * ( 
                        z * (2 * norm.cdf(z) - 1)
                        + 2 * norm.pdf(z) 
                        - 1/np.sqrt(np.pi)
                )      
            else:
                # Select the appropriate CDF method
                if method == "linear":
                    cdf = dist.cdf_linear
                elif method == "pchip":
                    cdf = dist.cdf_pchip
                elif method == "hybrid":
                    cdf = dist.hybrid_cdf
                else:
                    raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'pchip', or 'hybrid'.")

                # Define the CRPS integrands and compute it
                integrand_1 = lambda x: (cdf(x))**2
                integrand_2 = lambda x: (1 - cdf(x))**2
                crps_1, _ = quad(integrand_1, dist.y_min, y, epsabs=1e-6, epsrel=1e-6)
                crps_2, _ = quad(integrand_2, y, dist.y_max, epsabs=1e-6, epsrel=1e-6)
                crps_value = crps_1 + crps_2

            crps_values.append(crps_value)

            #if (i + 1) % 100 == 0:
            #    print(f"CRPS at sample {i+1}: {crps_value}")

        crps_quantiles = np.percentile(crps_values, np.array([5, 25, 75, 95]))
        crps_mean = np.mean(crps_values)

        return crps_quantiles, crps_mean, crps_values
    

    def calculate_crps_with_interpolator(self, method="linear"):
        """
        Calculate the CRPS using the specified CDF method from the CDFPDFInterpolator objects.

        Parameters:
        - method (str): Which method to use ('linear', 'pchip', 'hybrid').

        Returns:
        - float: Mean CRPS score for the specified CDF method.
        """
        crps_values = []

        for i, (dist, y) in enumerate(zip(self.cdf_pdf_interpolators, self.y_validation)):
            if method == "normal":
                deciles = self.deciles[i,:]
                probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                # Provide a reasonable initial guess
                mu_init = np.mean(deciles)
                sigma_init = (np.max(deciles) - np.min(deciles)) / 4  # Rough estimate
                #params, _ = curve_fit(lambda x, mu, sigma: norm.cdf(x, loc=mu, scale=sigma), 
                #                  deciles, probabilities, p0=[mu_init, sigma_init])
                params, _ = curve_fit(lambda x, mu, sigma: norm.ppf(x, loc=mu, scale=sigma), 
                    probabilities, deciles, p0=[mu_init, sigma_init])
                mu_fit, sigma_fit = params
                z = (y - mu_fit) / sigma_fit
                crps_value = sigma_fit * ( 
                        z * (2 * norm.cdf(z) - 1)
                        + 2 * norm.pdf(z) 
                        - 1/np.sqrt(np.pi)
                )      
            else:
                # Select the appropriate CDF method
                if method == "linear":
                    cdf = dist.cdf_linear
                elif method == "pchip":
                    cdf = dist.cdf_pchip
                elif method == "hybrid":
                    cdf = dist.hybrid_cdf
                else:
                    raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'pchip', or 'hybrid'.")

                # Define the CRPS integrands and compute it
                integrand_1 = lambda x: (cdf(x))**2
                integrand_2 = lambda x: (1 - cdf(x))**2
                crps_1, _ = quad(integrand_1, dist.y_min, y, epsabs=1e-6, epsrel=1e-6)
                crps_2, _ = quad(integrand_2, y, dist.y_max, epsabs=1e-6, epsrel=1e-6)
                crps_value = crps_1 + crps_2

            crps_values.append(crps_value)

            #if (i + 1) % 100 == 0:
            #    print(f"CRPS at sample {i+1}: {crps_value}")

        min_crps = np.min(crps_values)
        mean_crps = np.mean(crps_values)
        median_crps = np.median(crps_values)
        max_crps = np.max(crps_values)

        return mean_crps, median_crps, min_crps, max_crps

    def calculate_nll_with_interpolator(self, method="linear"):
        """
        Calculate the NLL using the specified PDF method from the CDFPDFInterpolator objects.

        Parameters:
        - method (str): Which method to use ('linear', 'pchip', 'hybrid').

        Returns:
        - float: Mean NLL score for the specified PDF method.
        """
        nll_values = []

        for i, (dist, y) in enumerate(zip(self.cdf_pdf_interpolators, self.y_validation)):

            if method == "normal":
                deciles = self.deciles[i,:]
                probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                # Provide a reasonable initial guess
                mu_init = np.mean(deciles)
                sigma_init = (np.max(deciles) - np.min(deciles)) / 4  # Rough estimate
                #params, _ = curve_fit(lambda x, mu, sigma: norm.cdf(x, loc=mu, scale=sigma), 
                #                  deciles, probabilities, p0=[mu_init, sigma_init])
                params, _ = curve_fit(lambda x, mu, sigma: norm.ppf(x, loc=mu, scale=sigma), 
                    probabilities, deciles, p0=[mu_init, sigma_init])
                mu_fit, sigma_fit = params
                pdf_value = norm.pdf(y, loc=mu_fit, scale=sigma_fit)
                # Calculate the NLL for the normal distribution
                nll_value = -np.log(pdf_value)
   
            else:
                # Select the appropriate CDF method
                if method == "linear":
                    pdf = dist.pdf_linear
                elif method == "pchip":
                    pdf = dist.pdf_pchip
                elif method == "hybrid":
                    pdf = dist.pdf_hybrid
                else:
                    raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'pchip', or 'hybrid'.")

                # Calculate the PDF at the observed value y
                pdf_value = pdf(y)
                # Calculate the NLL for the current distribution
                nll_value = -np.log(pdf_value)

            # Append NLL value for current sample
            nll_values.append(nll_value)

            # Print progress every 100 samples
            #if (i + 1) % 100 == 0:
            #    print(f"NLL at sample {i+1}: {nll_value}")

        # Return the mean NLL value across all samples

        min_nll = np.min(nll_values)
        mean_nll = np.mean(nll_values)
        median_nll = np.median(nll_values)
        max_nll = np.max(nll_values)

        return mean_nll, median_nll, min_nll, max_nll
    
    def calculate_nll_quantiles_with_interpolator(self, method="linear"):
        """
        Calculate the NLL using the specified PDF method from the CDFPDFInterpolator objects.

        Parameters:
        - method (str): Which method to use ('linear', 'pchip', 'hybrid').

        Returns:
        - float: Mean NLL score for the specified PDF method.
        """
        nll_values = []

        for i, (dist, y) in enumerate(zip(self.cdf_pdf_interpolators, self.y_validation)):

            if method == "normal":
                deciles = self.deciles[i,:]
                probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                # Provide a reasonable initial guess
                mu_init = np.mean(deciles)
                sigma_init = (np.max(deciles) - np.min(deciles)) / 4  # Rough estimate
                #params, _ = curve_fit(lambda x, mu, sigma: norm.cdf(x, loc=mu, scale=sigma), 
                #                  deciles, probabilities, p0=[mu_init, sigma_init])
                params, _ = curve_fit(lambda x, mu, sigma: norm.ppf(x, loc=mu, scale=sigma), 
                    probabilities, deciles, p0=[mu_init, sigma_init])
                mu_fit, sigma_fit = params
                pdf_value = norm.pdf(y, loc=mu_fit, scale=sigma_fit)
                # Calculate the NLL for the normal distribution
                nll_value = -np.log(pdf_value)
   
            else:
                # Select the appropriate CDF method
                if method == "linear":
                    pdf = dist.pdf_linear
                elif method == "pchip":
                    pdf = dist.pdf_pchip
                elif method == "hybrid":
                    pdf = dist.pdf_hybrid
                else:
                    raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'pchip', or 'hybrid'.")

                # Calculate the PDF at the observed value y
                pdf_value = pdf(y)
                # Calculate the NLL for the current distribution
                nll_value = -np.log(pdf_value)

            # Append NLL value for current sample
            nll_values.append(nll_value)

            # Print progress every 100 samples
            #if (i + 1) % 100 == 0:
            #    print(f"NLL at sample {i+1}: {nll_value}")

        # Return the mean NLL value across all samples
        nll_quantiles = np.percentile(nll_values, np.array([5, 25, 75, 95]))
        nll_mean = np.mean(nll_values)

        return nll_quantiles, nll_mean, nll_values
    
class ExperimentTracker:
    def __init__(self):
        self.experiments: list[Experiment] = []

    def track(self, X_train, y_train, X_validation, y_validation, 
            device: str = "auto", fit_mode: str = "low_memory", random_state: int = 42, ignore_pretraining_limits=False) -> Experiment:
        experiment = Experiment(X_train=X_train, y_train=y_train, X_validation=X_validation, y_validation=y_validation, device=device, fit_mode=fit_mode, random_state=random_state, ignore_pretraining_limits=ignore_pretraining_limits)
        experiment.perform()
        self.experiments.append(experiment)
        return experiment
    
    def __str__(self):
        return f"ExperimentTracker(len(experiments)=={len(self.experiments)})"
        

def get_histograms_from(experiment: Experiment) -> list[HistogramWithCRPS]:
    """
    Create a list of HistogramWithCRPS objects for each of the logits.

    Returns:
    - list: A list of HistogramWithCRPS objects of length n.
    """
    n = experiment.logits.shape[0]
    histograms = []
    
    # Iterate over the logits (axis 1: 5000 logits)
    for i in range(n):
        hist = HistogramWithCRPS(experiment.logits[i, :], experiment.borders)
        histograms.append(hist)
    
    return histograms

def get_linear_interpolated_dists_from(experiment: Experiment) -> list[HistogramWithCRPS]:
    """
    Create a list of HistogramWithCRPS objects for each of the logits.

    Returns:
    - list: A list of HistogramWithCRPS objects of length n.
    """
    n = experiment.logits.shape[0]
    histograms = []
    
    # Iterate over the logits (axis 1: 5000 logits)
    for i in range(n):
        hist = LinearInterpolatedHistWithCRPS(experiment.logits[i, :], experiment.borders)
        histograms.append(hist)
    
    return histograms

import pickle
import os

class ExperimentStorage:
    def __init__(self, file_path):
        self.file_path = file_path

    def save(self, experiment: Experiment):
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

class Visualization:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def old_compare_all_pdf_methods(self, id: int, case: int, window_size: int = 101):
        interpolator = self.experiment.cdf_pdf_interpolators[id]
        dist = self.experiment.dists[id]
        quantiles = interpolator.quantiles

        # X-axis range for interpolated PDFs
        x_min_full = np.floor(quantiles[0] - 5) 
        x_max_full = np.min([0.5, np.floor(quantiles[-1] + 2)])
        #x_values = np.linspace(x_min_full, x_max_full, 18500)
        bin_x = dist.bin_midpoints
        x_values = bin_x

        # Interpolated PDFs
        pdf_linear = np.array([interpolator.pdf_linear(x) for x in x_values])
        pdf_pchip = np.array([interpolator.pdf_pchip(x) for x in x_values])
        pdf_hybrid = np.array([interpolator.pdf_hybrid(x) for x in x_values])
        pdf_normal = np.array([interpolator.fitted_normal_pdf(x) for x in x_values])

        # Raw + Smoothed PDF
        raw_pdf = dist.pdf(bin_x)
        smooth_pdf = np.convolve(raw_pdf, np.ones(window_size) / window_size, mode='same')

        # Case-specific zoom regions
        if case == 1:
            title = "Full Distribution"
            x_min, x_max = x_min_full, x_max_full
        elif case == 2:
            peak_index = np.argmax(pdf_hybrid)
            peak_x = x_values[peak_index]
            x_min, x_max = peak_x - 0.2, peak_x + 0.2
            x_min = np.floor(10 * quantiles[0]) / 10
            x_max = np.ceil(10 * quantiles[-1]) / 10
            title = "Peak Region"
        elif case == 3:
            x_min, x_max = quantiles[0] - 3.1, quantiles[0] - 2.9
            title = "Left Tail"
        else:
            raise ValueError("Invalid case. Choose 1 (full), 2 (peak), or 3 (left tail)")

        # Filter values for y-limits
        def in_range(x, y): return y[(x >= x_min) & (x <= x_max)]

        if (case == 1 or case == 2):
            y_all = in_range(bin_x, raw_pdf)
        
        else:

            y_all = np.concatenate([
                in_range(x_values, pdf_linear),
                in_range(x_values, pdf_pchip),
                in_range(x_values, pdf_hybrid),
                in_range(bin_x, raw_pdf),
                in_range(bin_x, smooth_pdf)
            ])
            
        y_all = y_all[y_all > 0]  # Remove zeros or negatives before log scale
        y_min = 10 ** np.ceil(np.log10(np.min(y_all))) * 0.9 if len(y_all) > 0 else 1e-7
        y_max = 10 ** np.floor(np.log10(np.max(y_all))) * 10 if len(y_all) > 0 else 1e1
        y_min = 10 ** np.floor(np.log10(np.min(y_all))) * 1.0 if len(y_all) > 0 else 1e-7
        y_max = 10 ** np.ceil(np.log10(np.max(y_all))) * 1.0 if len(y_all) > 0 else 1e1

        # Plot
        plt.figure(figsize=(10, 5))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.yscale("log")

        # Plot all methods
        plt.plot(x_values, pdf_linear, label="Linear Interpolation", linewidth=1)
        plt.plot(x_values, pdf_pchip, label="PCHIP Interpolation", linewidth=1)
        plt.plot(x_values, pdf_hybrid, label="Hybrid Interpolation", linewidth=1)
        plt.plot(bin_x, raw_pdf, 'o', color='red', markersize=1, label="Raw PDF")
        plt.plot(bin_x, smooth_pdf, linestyle="dashdot", color='blue', linewidth=1, label=f"Smoothed PDF (window={window_size})")
        plt.plot(bin_x, pdf_normal, linewidth=1, label=f"Normal PDF")
        plt.scatter(quantiles, np.array([interpolator.pdf_hybrid(q) for q in quantiles]), color="black", s=30, marker="x")

        plt.xlabel("Value")
        plt.ylabel("PDF (log scale)")
        plt.title(f"PDF Comparison for Sample {id} - {title}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # For case 2, make a separate plot with linear y scale
        if case == 2:
            y_max_linear = np.max([np.max(pdf_linear), np.max(pdf_pchip), np.max(pdf_hybrid), np.max(raw_pdf), np.max(smooth_pdf)]) * 1.1

            plt.figure(figsize=(10, 5))
            plt.xlim(x_min, x_max)
            plt.ylim(0, y_max_linear)

            # Plot all methods with linear scale
            plt.plot(x_values, pdf_linear, label="Linear Interpolation", linewidth=1)
            plt.plot(x_values, pdf_pchip, label="PCHIP Interpolation", linewidth=1)
            plt.plot(x_values, pdf_hybrid, label="Hybrid Interpolation", linewidth=1)
            plt.plot(bin_x, raw_pdf, 'o', color='red', markersize=1, label="Raw PDF")
            plt.plot(bin_x, smooth_pdf, linestyle="dashdot", color='blue', linewidth=1, label=f"Smoothed PDF (window={window_size})")
            plt.plot(bin_x, pdf_normal, linewidth=1, label=f"Normal PDF")
            plt.scatter(quantiles, np.array([interpolator.pdf_hybrid(q) for q in quantiles]), color="black", s=30, marker="x")

            plt.xlabel("Value")
            plt.ylabel("PDF")
            plt.title(f"PDF Comparison for Sample {id} - {title} (Linear Scale)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def compare_all_cdf_methods(self, id: int, case: int, save_plots, save_path="C:/Users/Minu/Documents/results"):
        
        def make_step_cdf_data(raw_cdf, bin_x_edges):
            """
            Generate step-wise x and y arrays for plotting a CDF as a step function.

            Parameters:
            - raw_cdf (np.ndarray): CDF values at bin midpoints (len = len(bin_x_edges) - 1)
            - bin_x_edges (np.ndarray): The edges of the bins (len = len(raw_cdf) + 1)

            Returns:
            - x_step (np.ndarray): Step x-values
            - cdf_step (np.ndarray): Step y-values
            """
            x_step = np.repeat(bin_x_edges, 2)[1:-1]  # exclude first and last repeats
            cdf_step = np.repeat(raw_cdf, 2)
            return x_step, cdf_step

        interpolator = self.experiment.cdf_pdf_interpolators[id]
        dist = self.experiment.hists[id]
        quantiles = interpolator.quantiles

        # X-axis range for interpolated CDFs
        x_min_full = np.floor(quantiles[0] - 5) 
        x_max_full = np.min([0.5, np.floor(quantiles[-1] + 2)])
        bin_x = dist.bin_midpoints
        x_values = bin_x

        # Interpolated CDFs
        cdf_linear = np.array([interpolator.cdf_linear(x) for x in x_values])
        cdf_pchip = np.array([interpolator.cdf_pchip(x) for x in x_values])
        cdf_hybrid = np.array([interpolator.hybrid_cdf(x) for x in x_values])
        cdf_normal = np.array([interpolator.fitted_normal_cdf(x) for x in x_values])

        # Raw + Smoothed CDF
        raw_cdf = dist.cdf(bin_x)
        #smooth_cdf = np.convolve(raw_cdf, np.ones(window_size) / window_size, mode='same') is not correct moving average

        # Case-specific zoom regions
        if case == 1:
            title = "Wide Region"
            x_min, x_max = x_min_full, x_max_full
        elif case == 2:
            peak_index = np.argmax(cdf_hybrid)
            peak_x = x_values[peak_index]
            x_min, x_max = peak_x - 0.2, peak_x + 0.2
            x_min = np.floor(10 * quantiles[0]) / 10
            x_max = np.ceil(10 * quantiles[-1]) / 10
            title = "Decile Region"
        elif case == 3:
            x_min, x_max = quantiles[0] - 3.1, quantiles[0] - 2.9
            title = "Left Tail"
        else:
            raise ValueError("Invalid case. Choose 1 (full), 2 (peak), or 3 (left tail)")

        # Filter values for y-limits
        def in_range(x, y): return y[(x >= x_min) & (x <= x_max)]

        if (case == 1 or case == 2):
            y_all = in_range(bin_x, raw_cdf)
            y_min = np.min(y_all) * 0.95 if len(y_all) > 0 else 0
            y_max = np.max(y_all) * 1.05 if len(y_all) > 0 else 1
        else:
            y_all = np.concatenate([
                #in_range(x_values, cdf_linear),
                #in_range(x_values, cdf_pchip),
                #in_range(x_values, cdf_hybrid),
                in_range(bin_x, raw_cdf),
                #in_range(bin_x, smooth_cdf)
            ])
            y_min = np.min(y_all) * 0.95 if len(y_all) > 0 else 0
            y_max = np.max(y_all) * 1.05 if len(y_all) > 0 else 1
        
        plt.rcParams.update({
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6
        })
        
        # Plot for log scale (for case 1 or case 2)
        plt.figure(figsize=(5, 3))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        if case == 3:
            plt.yscale("linear")
            plt.title(f"CDF Comparison for Sample {id} - {title} (Linear Scale)")
            x_step_cdf, cdf_step = make_step_cdf_data(raw_cdf, dist.borders)
            plt.plot(x_step_cdf, cdf_step, drawstyle="steps-post", linestyle="solid", color="red", linewidth=1, label="Raw CDF (step)")
            plt.scatter(x_values, raw_cdf, s=10, marker="x", label="Raw CDF (midpoints)")
            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.legend()
            plt.grid(True)
            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_plots:
                filename = f"CDF Comparison for Sample {id} - {title} (Linear Scale).pdf"
                plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            plt.show()

        
        elif case == 1:
            plt.plot(x_values, cdf_linear, label="Linear Interpolation", linewidth=1)
            plt.plot(x_values, cdf_pchip, label="PCHIP Interpolation", linewidth=1)
            plt.plot(x_values, cdf_hybrid, label="Hybrid Interpolation", linewidth=1)
            plt.plot(bin_x, cdf_normal, linewidth=1, label=f"Normal CDF")
            x_step_cdf, cdf_step = make_step_cdf_data(raw_cdf, dist.borders)
            plt.plot(x_step_cdf, cdf_step, drawstyle="steps-post", linestyle="solid", linewidth=1, label="Raw CDF (step)")
            plt.title(f"CDF Comparison for Sample {id} - {title} (Log Scale)")
            plt.yscale("log")
            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            if save_plots:
                    filename = f"CDF Comparison for Sample {id} - {title} (Log Scale).pdf"
                    plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            
            plt.show()

        elif case == 2:
            plt.plot(x_values, cdf_linear, label="Linear Interpolation", linewidth=1)
            plt.plot(x_values, cdf_pchip, label="PCHIP Interpolation", linewidth=1)
            plt.plot(x_values, cdf_hybrid, label="Hybrid Interpolation", linewidth=1)
            plt.plot(bin_x, cdf_normal, linewidth=1, label=f"Normal CDF")
            plt.scatter(quantiles, np.array([interpolator.hybrid_cdf(q) for q in quantiles]), color="black", s=10, marker="x", label="Deciles")

            #plt.scatter(x_values, raw_cdf, s=3, marker="x", label="Raw CDF (midpoints)")
            plt.title(f"CDF Comparison for Sample {id} - {title} (Log Scale)")
            plt.yscale("log")
            x_step_cdf, cdf_step = make_step_cdf_data(raw_cdf, dist.borders)
            plt.plot(x_step_cdf, cdf_step, drawstyle="steps-post", linestyle="solid", linewidth=1, label="Raw CDF (step)")
            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_plots:
                    filename = f"CDF Comparison for Sample {id} - {title} (Log Scale).pdf"
                    plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            plt.show()

            plt.figure(figsize=(5, 3))
            y_max_linear = np.max([np.max(cdf_linear), np.max(cdf_pchip), np.max(cdf_hybrid), np.max(raw_cdf)]) * 1.1

            plt.xlim(x_min, x_max)
            plt.ylim(0, y_max_linear)
            plt.yscale("linear")

            # Plot all methods with linear scale
            plt.plot(x_values, cdf_linear, label="Linear Interpolation", linewidth=1)
            plt.plot(x_values, cdf_pchip, label="PCHIP Interpolation", linewidth=1)
            plt.plot(x_values, cdf_hybrid, label="Hybrid Interpolation", linewidth=1)
            #plt.plot(bin_x, raw_cdf, 'o', color='red', markersize=1, label="Raw CDF")
            x_step_cdf, cdf_step = make_step_cdf_data(raw_cdf, dist.borders)
            plt.plot(x_step_cdf, cdf_step, drawstyle="steps-post", linestyle="solid", linewidth=1, label="Raw CDF (step)")

            #plt.plot(bin_x, smooth_cdf, linestyle="dashdot", color='blue', linewidth=1, label=f"Smoothed CDF (window={window_size})")
            plt.plot(bin_x, cdf_normal, linewidth=1, label=f"Normal CDF")
            plt.scatter(quantiles, np.array([interpolator.hybrid_cdf(q) for q in quantiles]), color="black", s=10, marker="x", label="Deciles")

            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.title(f"CDF Comparison for Sample {id} - {title} (Linear Scale)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_plots:
                filename = f"cdf_sample_{id}_case_{case}_{title.lower().replace(' ', '_')}_linear.pdf"
                plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            plt.show()

    def compare_all_pdf_methods(self, id: int, case: int, window_size: int = 101, show_smoothed_only: bool = False, save_plots: bool = False, save_path="C:/Users/Minu/Documents/results"):

        def make_step_plot_data(raw_pdf, bin_x_edges):
            """
            Generate step-wise x and y arrays for plotting a histogram-like PDF.

            Parameters:
            - raw_pdf (np.ndarray): The probability density values per bin (length = len(bin_x_edges) - 1).
            - bin_x_edges (np.ndarray): The edges of the bins (length = len(raw_pdf) + 1).

            Returns:
            - x_step (np.ndarray): The x-values repeated to form step edges.
            - pdf_step (np.ndarray): The y-values repeated to match the step x-values.
            """
            # Repeat each PDF value twice for horizontal steps
            pdf_step = np.repeat(raw_pdf, 2)
            
            # Repeat bin edges (excluding the last) twice for vertical transitions
            x_step = np.repeat(bin_x_edges[:-1], 2)
            
            # Append the final bin edge and last pdf value to close the step
            x_step = np.append(x_step, bin_x_edges[-1])
            pdf_step = np.append(pdf_step, pdf_step[-1])
            
            return x_step, pdf_step


        interpolator = self.experiment.cdf_pdf_interpolators[id]
        dist = self.experiment.hists[id]
        quantiles = interpolator.quantiles

        # X-axis range for interpolated PDFs
        x_min_full = np.floor(quantiles[0] - 5) 
        x_max_full = np.min([0.5, np.floor(quantiles[-1] + 2)])
        bin_x = dist.bin_midpoints
        x_values = bin_x
        bin_x_edges = dist.borders

        # Interpolated CDFs
        pdf_linear = np.array([interpolator.pdf_linear(x) for x in x_values])
        pdf_pchip = np.array([interpolator.pdf_pchip(x) for x in x_values])
        pdf_hybrid = np.array([interpolator.pdf_hybrid(x) for x in x_values])
        pdf_normal = np.array([interpolator.fitted_normal_pdf(x) for x in x_values])
        
        # Raw + Smoothed CDF
        raw_pdf = dist.pdf(bin_x)
        smooth_pdf = np.convolve(raw_pdf, np.ones(window_size) / window_size, mode='same')
        smooth_pdf2 = np.convolve(raw_pdf, np.ones(3) / 3, mode='same')
        smooth_pdf3 = np.convolve(raw_pdf, np.ones(55) / 55, mode='same')
        smooth_pdf4 = np.convolve(raw_pdf, np.ones(5) / 5, mode='same')

        # Case-specific zoom regions
        if case == 1:
            title = "Wide Region"
            x_min, x_max = x_min_full, x_max_full
        elif case == 2:
            peak_index = np.argmax(pdf_hybrid)
            peak_x = x_values[peak_index]
            x_min, x_max = peak_x - 0.2, peak_x + 0.2
            x_min = np.floor(10 * quantiles[0]) / 10
            x_max = np.ceil(10 * quantiles[-1]) / 10
            title = "Decile Region"
        elif case == 3:
            x_min, x_max = quantiles[0] - 3.1, quantiles[0] - 2.9
            #print("x_min", x_min)
            #print("x_max", x_max)
            title = "Left Tail"
        else:
            raise ValueError("Invalid case. Choose 1 (full), 2 (peak), or 3 (left tail)")

        # Filter values for y-limits
        def in_range(x, y): return y[(x >= x_min) & (x <= x_max)]

        if (case == 1 or case == 2):
            y_all = in_range(bin_x, raw_pdf)
            y_min = np.min(y_all) * 0.95 if len(y_all) > 0 else 0
            y_max = np.max(y_all) * 1.05 if len(y_all) > 0 else 1
        else:
            y_all = np.concatenate([
                #in_range(x_values, cdf_linear),
                #in_range(x_values, cdf_pchip),
                #in_range(x_values, cdf_hybrid),
                in_range(bin_x, raw_pdf),
                #in_range(bin_x, smooth_pdf)
            ])
            #print("y_all", y_all)
            
            y_min = np.min(y_all)
            #print("y_min", y_min)
            y_min_exponent = int(np.floor(np.log10(abs(y_min)))) # extract the exponent
            y_min_scaled = y_min / (10**y_min_exponent) # scale y_min to scientific notation, e.g. 1.2345E-6
            y_min = np.round(y_min_scaled, 1) * (10**y_min_exponent) # round to 2 s.f. and scaöe back to original exponent

            y_max = np.max(y_all)
            #print("y_max", y_max)
            y_max_exponent = int(np.floor(np.log10(abs(y_max)))) # extract the exponent
            #print("y_max_exponent", y_max_exponent)
            y_max_scaled = y_max / (10**y_max_exponent) # scale y_min to scientific notation, e.g. 1.2345E-6
            #print("y_max_scaled",y_max_scaled)
            y_max = np.ceil(y_max_scaled * 10)/10
            #print("y_max", y_max)
            y_max = y_max * (10**y_max_exponent) # round to 2 s.f. and scaöe back to original exponent
            #print("y_max", y_max)

        plt.rcParams.update({
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6
        })
        # Plot for log scale (for case 1 or case 2)
        plt.figure(figsize=(5, 3))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # case 3 show only pdfs based on 5000 bins
        if case == 3:
           plt.yscale("linear")
           plt.title(f"Smoothed PDFs for Sample {id} - {title} (Linear Scale)")
           #plt.plot(bin_x, raw_pdf, color='red', linestyle="-", label="Raw PDF")
           #pdf_step = np.repeat(smooth_pdf, 2)
           #pdf_step = np.repeat(raw_pdf, 2)
           #x_step = np.repeat(bin_x_edges[:-1], 2)  # Avoid duplicating first/last edge
           
           # Add first and last edges
           #x_step = np.insert(x_step, 0, bin_x_edges[0])
           #x_step = np.append(x_step, bin_x_edges[-1])
           #pdf_step = np.insert(pdf_step, 0, pdf_step[0])
           #pdf_step = np.append(pdf_step, pdf_step[-1])

           x_step, pdf_step = make_step_plot_data(raw_pdf, bin_x_edges)
           
           plt.plot(x_step, pdf_step, drawstyle="steps-post", linestyle="solid", color="black", label="Histogram Step PDF")
           plt.plot(bin_x, smooth_pdf, linestyle="-", linewidth=1, label=f"Smoothed PDF (window={window_size})")
           plt.plot(bin_x, smooth_pdf2, linestyle="-", linewidth=1, label=f"Smoothed PDF (window={3})")
           plt.plot(bin_x, smooth_pdf3, linestyle="-", linewidth=1, label=f"Smoothed PDF (window={55})")
           plt.plot(bin_x, smooth_pdf4, linestyle="-", linewidth=1, label=f"Smoothed PDF (window={5})")
           plt.xlabel("Value")
           plt.ylabel("PDF")
           plt.legend()
           plt.grid(True)
           plt.tight_layout()
           if save_plots:
                filename = f"PDF Comparison for Sample {id} - {title} (Linear Scale).pdf"
                plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
           plt.show()

        elif case == 2 and show_smoothed_only:
            plt.yscale("log")
            plt.title(f"Smoothed PDFs for Sample {id} - {title} (Log Scale)")

            # Step plot
            x_step, pdf_step = make_step_plot_data(raw_pdf, bin_x_edges)
            plt.plot(x_step, pdf_step, drawstyle="steps-post", linestyle="solid", color="black", label="Histogram Step PDF")

            # Smoothed PDFs with different windows
            for w in [5, window_size, 55]:
                smooth_pdf = np.convolve(raw_pdf, np.ones(w) / w, mode='same')
                plt.plot(bin_x, smooth_pdf, linestyle='-', linewidth=1, label=f"Smoothed PDF (window={w})")
            plt.xlabel("Value")
            plt.ylabel("PDF")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_plots:
                filename = f"PDF Comparison for Sample {id} - {title} - Smoothing.pdf"
                plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            plt.show()

        elif case == 2:
            plt.plot(x_values, pdf_linear, label="Linear Interpolation", linewidth=1)
            plt.plot(x_values, pdf_pchip, label="PCHIP Interpolation", linewidth=1)
            plt.plot(x_values, pdf_hybrid, label="Hybrid Interpolation", linewidth=1)
            plt.plot(bin_x, pdf_normal, linewidth=1, label=f"Normal PDF")
            #plt.plot(bin_x, smooth_pdf, linestyle="dashdot", color='blue', linewidth=1, label=f"Smoothed PDF (window={window_size})")
            plt.scatter(quantiles, np.array([interpolator.pdf_hybrid(q) for q in quantiles]), color="black", s=30, marker="x", label="Deciles")
            plt.title(f"PDF Comparison for Sample {id} - {title} (Log Scale)")
            plt.yscale("log")
            #plt.plot(bin_x, raw_pdf, 'o', color='red', markersize=1, label="Raw PDF (at bin midpoints)")
            x_step, pdf_step = make_step_plot_data(raw_pdf, bin_x_edges)
            plt.plot(x_step, pdf_step, drawstyle="steps-post", linestyle="solid", color="black", label="Histogram Step PDF")

            plt.plot(bin_x, smooth_pdf, linestyle="-", color='blue', linewidth=1, label=f"Smoothed PDF (window={window_size})")
            plt.xlabel("Value")
            plt.ylabel("PDF")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_plots:
                filename = f"PDF Comparison for Sample {id} - {title} - all.pdf"
                plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
        else:
            plt.plot(x_values, pdf_linear, label="Linear Interpolation", linewidth=1)
            plt.plot(x_values, pdf_pchip, label="PCHIP Interpolation", linewidth=1)
            plt.plot(x_values, pdf_hybrid, label="Hybrid Interpolation", linewidth=1)
            plt.plot(bin_x, pdf_normal, linewidth=1, label=f"Normal PDF")
            #plt.plot(bin_x, smooth_pdf, linestyle="dashdot", color='blue', linewidth=1, label=f"Smoothed PDF (window={window_size})")
            plt.scatter(quantiles, np.array([interpolator.pdf_hybrid(q) for q in quantiles]), color="black", s=30, marker="x", label="Deciles")
            plt.title(f"PDF Comparison for Sample {id} - {title} (Log Scale)")
            plt.yscale("log")
            plt.plot(bin_x, raw_pdf, 'o', color='red', markersize=1, label="Raw PDF (at bin midpoints)")
            #x_step, pdf_step = make_step_plot_data(raw_pdf, bin_x_edges)
            #plt.plot(x_step, pdf_step, drawstyle="steps-post", linestyle="solid", color="black", label="Histogram Step PDF")

            plt.plot(bin_x, smooth_pdf, linestyle="-", color='blue', linewidth=1, label=f"Smoothed PDF (window={window_size})")
            plt.xlabel("Value")
            plt.ylabel("PDF")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_plots:
                filename = f"PDF Comparison for Sample {id} - {title} - Full Range (Log Scale).pdf"
                plt.savefig(os.path.join(save_path, "tabpfn", filename), format="pdf", bbox_inches="tight", dpi=300)
            plt.show()

    def plot_raw_pdf(self, id: int):
        """Plots the raw PDF for a given sample ID from the experiment."""
        # Access the distribution data from the experiment's `dists` property
        dist = self.experiment.dists[id]
        
        # Retrieve bin midpoints and raw PDF values
        x = dist.bin_midpoints
        pdf = dist.pdf(x)
        
        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.xlim(-5, 0)
        plt.ylim(1e-11, 1e1)
        plt.yscale("log")
        plt.scatter(x, pdf, color="red", s=5, label="Raw PDF Points")
        plt.xlabel("x (Midpoints of bins)")
        plt.ylabel("PDF Value (log scale)")
        plt.title(f"Raw Conditional PDF for ID {id}")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_smoothed_pdf(self, id: int, window_size=101):
        """Plots the smoothed PDF for a given sample ID from the experiment."""
        dist = self.experiment.dists[id]
        
        x = dist.bin_midpoints
        pdf = dist.pdf(x)
        
        # Case 1: Full Distribution
        x_min = -5
        x_max = 0

        mask = (x >= x_min) & (x <= x_max) # Filter PDF values within the specified x-range
        x_in_range = x[mask]
        pdf_in_range = pdf[mask]

        y_min = np.min(pdf_in_range)
        y_max = np.max(pdf_in_range)
        y_min = 10**np.floor(np.log10(y_min)) if y_min > 0 else 1e-11
        y_max = 10**np.ceil(np.log10(y_max))

        window = np.ones(window_size) / window_size
        smooth_pdf = np.convolve(pdf, window, mode='same')

        plt.figure(figsize=(8, 5))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.yscale("log")
        #plt.scatter(x, pdf, color="red", s=5, label="Original PDF Points")
        plt.plot(x, smooth_pdf, color="blue", label=f"Smoothed PDF (window={window_size})")
        plt.xlabel("x (Midpoints of bins)")
        plt.ylabel("PDF Value (log scale)")
        plt.title(f"Smoothed Conditional PDF for ID {id} (Case 1 - Full Distribution)")
        plt.legend()
        plt.grid()
        plt.show()   
    
    def plot_deciles_pdf(self, id: int):
        """Plots the raw PDF for a given sample ID from the experiment."""
        # Access the distribution data from the experiment's `dists` property
        interpolator = self.experiment.cdf_pdf_interpolators[id]
        quantiles = interpolator.quantiles
        x_min = quantiles[0] - 5
        x_max = quantiles[-1] + 5
        x_values = np.linspace(x_min, x_max, 18500)

        pdf_linear = np.array([interpolator.pdf_linear(x) for x in x_values])
        pdf_pchip = np.array([interpolator.pdf_pchip(x) for x in x_values])
        pdf_hybrid = np.array([interpolator.pdf_hybrid(x) for x in x_values])


        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, pdf_linear, label="Linear", linewidth=2)
        plt.plot(x_values, pdf_pchip, label="PCHIP", linewidth=2)
        plt.plot(x_values, pdf_hybrid, label="Hybrid", linewidth=2)

        # Add raw PDF point estimates (optional)
        #raw_pchip = interpolator.pdf_pchip(quantiles)
        plt.scatter(quantiles, [interpolator.pdf_pchip(q) for q in quantiles], color="red", s=20, label="Quantile Points")
        plt.yscale("log")
        plt.xlim(x_min, x_max)
        plt.xlabel("Value")
        plt.ylabel("PDF (log scale)")
        plt.title(f"Interpolated PDF Methods for Sample {id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_deciles_cdf(self, id: int):
        """Plots the raw PDF for a given sample ID from the experiment."""
        # Access the distribution data from the experiment's `dists` property
        interpolator = self.experiment.cdf_pdf_interpolators[id]
        quantiles = interpolator.quantiles
        x_min = quantiles[0] - 5
        x_max = quantiles[-1] + 5
        x_values = np.linspace(x_min, x_max, 18500)
        
        cdf_linear = np.array([interpolator.cdf_linear(x) for x in x_values])
        cdf_pchip = np.array([interpolator.cdf_pchip(x) for x in x_values])
        cdf_hybrid = np.array([interpolator.hybrid_cdf(x) for x in x_values])

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.xlim(x_min, x_max)
        
        #plt.ylim(1e-7, 1e3)
        #plt.yscale("log")
        
        plt.scatter(quantiles, [interpolator.hybrid_cdf(q) for q in quantiles], marker='o', label="CDF at Quantiles", color="red")
        plt.plot(x_values, cdf_linear, label="Linear Interpolation")
        plt.plot(x_values, cdf_pchip, label="PCHIP Interpolation")
        plt.plot(x_values, cdf_hybrid, label="Hybrid Interpolation")

        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("CDF Estimation of one set of quantiles of TabPFN with different interpolation methods")
        plt.legend()
        plt.grid()
        plt.show()


def calculate_scores_tabpfn(experiment_id, storage_path="experiments/", all_scores=False):
    """
    Calculate and save NLL, CRPS scores for a specific experiment.

    This function loads a previously saved experiment, calculates key performance metrics (NLL and CRPS),
    and optionally includes additional scores based on different interpolators. The results are saved to a 
    pickle file for later use.

    Parameters:
    - experiment_id (int): The ID of the experiment whose results are to be calculated.
    - storage_path (str, optional): The directory where the experiment results and scores will be saved. 
      Default is "experiments/".
    - all_scores (bool, optional): If True, the function will calculate and include additional scores 
      using different interpolators (linear, pchip, hybrid). Default is False.
    """
    experiment_filename = f"{storage_path}/tabpfn/experiment_{experiment_id}.pkl"
    print("experiment_filename", experiment_filename)
    storage = ExperimentStorage(experiment_filename)
    experiment = storage.load()

    # nll 5000 logits with moving average
    #mean_nll_5000, median_nll_5000, min_nll_5000, max_nll_5000 = experiment.calculate_nll(smoothing=True, window_size=11)

    nll_quantiles, mean_nll_5000, nll_scores = experiment.calculate_nll_quantiles(smoothing=True, window_size=11)
    crps_quantiles, mean_crps_5000, crps_scores = experiment.calculate_crps_quantiles()

    scores = {
        'Metric': ['nll_5000_smoothed', 'crps_5000_smoothed'],
        'q5': [nll_quantiles[0], crps_quantiles[0]],
        'q25': [nll_quantiles[1], crps_quantiles[1]],
        'q75': [nll_quantiles[2], crps_quantiles[2]],
        'q95': [nll_quantiles[3], crps_quantiles[3]],
        'Mean': [mean_nll_5000, mean_crps_5000],
    }

    # Stack the scores first
    score_columns = {f'score_{i}': [nll_scores[i], crps_scores[i]] for i in range(len(nll_scores))}

    # Merge with original scores
    scores.update(score_columns)

    df = pd.DataFrame(scores)

    if all_scores:
        # --- Additional NLL Quantiles ---
        #nll_raw_quantiles, nll_raw_mean = experiment.calculate_nll_quantiles(smoothing=False, window_size=11)
        #nll_linear_quantiles, nll_linear_mean = experiment.calculate_nll_quantiles_with_interpolator(method="linear")
        #nll_pchip_quantiles, nll_pchip_mean = experiment.calculate_nll_quantiles_with_interpolator(method="pchip")
        #nll_hybrid_quantiles, nll_hybrid_mean = experiment.calculate_nll_quantiles_with_interpolator(method="hybrid")

        #crps_raw_quantiles, crps_raw_mean = experiment.calculate_crps_quantiles(smoothing=False, window_size=11)
        #crps_linear_quantiles, crps_linear_mean = experiment.calculate_crps_quantiles_with_interpolator(method="linear")
        #crps_pchip_quantiles, crps_pchip_mean = experiment.calculate_crps_quantiles_with_interpolator(method="pchip")
        #crps_hybrid_quantiles, crps_hybrid_mean = experiment.calculate_crps_quantiles_with_interpolator(method="hybrid")
    
        nll_raw_quantiles, nll_raw_mean, nll_raw_scores = experiment.calculate_nll_quantiles(smoothing=False, window_size=11)
        nll_linear_quantiles, nll_linear_mean, nll_linear_scores = experiment.calculate_nll_quantiles_with_interpolator(method="linear")
        nll_pchip_quantiles, nll_pchip_mean, nll_pchip_scores = experiment.calculate_nll_quantiles_with_interpolator(method="pchip")
        nll_hybrid_quantiles, nll_hybrid_mean, nll_hybrid_scores = experiment.calculate_nll_quantiles_with_interpolator(method="hybrid")

        crps_raw_quantiles, crps_raw_mean, crps_raw_scores = experiment.calculate_crps_quantiles(smoothing=False, window_size=11)
        crps_linear_quantiles, crps_linear_mean, crps_linear_scores = experiment.calculate_crps_quantiles_with_interpolator(method="linear")
        crps_pchip_quantiles, crps_pchip_mean, crps_pchip_scores = experiment.calculate_crps_quantiles_with_interpolator(method="pchip")
        crps_hybrid_quantiles, crps_hybrid_mean, crps_hybrid_scores = experiment.calculate_crps_quantiles_with_interpolator(method="hybrid")


        
        interpolator_scores = {
            'Metric': [
                'nll_5000_raw', 'nll_linear', 'nll_pchip', 'nll_hybrid',
                'crps_5000_raw', 'crps_linear', 'crps_pchip', 'crps_hybrid'
            ],
            'q5': [
                nll_raw_quantiles[0], nll_linear_quantiles[0], nll_pchip_quantiles[0], nll_hybrid_quantiles[0],
                crps_raw_quantiles[0], crps_linear_quantiles[0], crps_pchip_quantiles[0], crps_hybrid_quantiles[0]
            ],
            'q25': [
                nll_raw_quantiles[1], nll_linear_quantiles[1], nll_pchip_quantiles[1], nll_hybrid_quantiles[1],
                crps_raw_quantiles[1], crps_linear_quantiles[1], crps_pchip_quantiles[1], crps_hybrid_quantiles[1]
            ],
            'q75': [
                nll_raw_quantiles[2], nll_linear_quantiles[2], nll_pchip_quantiles[2], nll_hybrid_quantiles[2],
                crps_raw_quantiles[2], crps_linear_quantiles[2], crps_pchip_quantiles[2], crps_hybrid_quantiles[2]
            ],
            'q95': [
                nll_raw_quantiles[3], nll_linear_quantiles[3], nll_pchip_quantiles[3], nll_hybrid_quantiles[3],
                crps_raw_quantiles[3], crps_linear_quantiles[3], crps_pchip_quantiles[3], crps_hybrid_quantiles[3]
            ],
            'Mean': [
                nll_raw_mean, nll_linear_mean, nll_pchip_mean, nll_hybrid_mean,
                crps_raw_mean, crps_linear_mean, crps_pchip_mean, crps_hybrid_mean
            ]
        }

        score_columns = {f'score_{i}': [nll_raw_scores[i], nll_linear_scores[i], nll_pchip_scores[i], nll_hybrid_scores[i], 
                                        crps_raw_scores[i], crps_linear_scores[i], crps_pchip_scores[i], crps_hybrid_scores[i]] for i in range(len(nll_raw_scores))}

        # Merge with original scores
        interpolator_scores.update(score_columns)

        interpolator_df = pd.DataFrame(interpolator_scores)
        df = pd.concat([df, interpolator_df], ignore_index=True)

    # Save results
    file_name = f"{storage_path}/tabpfn/quantiles/experiment_results_{experiment_id}.pkl"
    print("Saving to:", file_name)
    storage = ExperimentStorage(file_name)
    storage.save(df)
    print("Save complete:", os.path.exists(file_name))

    return df

from datetime import datetime
import time
import warnings

def run_tabpfn(experiment_ids, storage_path="experiments/", ignore_pretraining_limits=False):
    """
    Run multiple experiments with different configurations and save the results.

    This function takes a list (or a single integer) of experiment IDs, retrieves the corresponding 
    configurations, preprocesses the data, trains a model, and saves the results to the specified storage path.
    
    Parameters:
    - experiment_ids (int or list of int): One or more experiment IDs. If a single integer is provided, it will be converted to a list containing that ID.
    - storage_path (str, optional): The directory where the experiment results will be saved. Default is "experiments/".
    Returns:
    - None: This function does not return any value; it performs the experiment runs and saves results to files.

    Example:
    >>> run_tabpfn([1, 2, 3], storage_path="my_results/")
    
    This will run experiments with IDs 1, 2, and 3, processing the data, training the model, 
    and saving the results in the "my_results/" directory.
    """
    storage_path = os.path.join(storage_path, "tabpfn")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)


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

                experiment = Experiment(X_train=train_X, y_train=train_y, X_validation=validation_X, y_validation=validation_y, random_state=random_state, ignore_pretraining_limits=ignore_pretraining_limits)
                experiment = experiment.perform()

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"⏱️ Experiment {experiment_id} completed in {elapsed:.2f} seconds")

            print("- Saving experiment results...")
            experiment_filename = f"{storage_path}/experiment_{experiment_id}.pkl"
            storage = ExperimentStorage(experiment_filename)
            storage.save(experiment)
            print(f"Experiment saved to: {experiment_filename}")

    print("All experiments completed and saved.")


def _calculate_scores_tabpfn(experiment_id, storage_path="experiments/", all_scores=False):
    """
    Calculate and save NLL, CRPS scores for a specific experiment.

    This function loads a previously saved experiment, calculates key performance metrics (NLL and CRPS),
    and optionally includes additional scores based on different interpolators. The results are saved to a 
    pickle file for later use.

    Parameters:
    - experiment_id (int): The ID of the experiment whose results are to be calculated.
    - storage_path (str, optional): The directory where the experiment results and scores will be saved. 
      Default is "experiments/".
    - all_scores (bool, optional): If True, the function will calculate and include additional scores 
      using different interpolators (linear, pchip, hybrid). Default is False.
    """

    storage_path = os.path.join(storage_path, "tabpfn")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    experiment_filename = f"{storage_path}/experiment_{experiment_id}.pkl"
    print("experiment_filename: ", experiment_filename)
    storage = ExperimentStorage(experiment_filename)
    experiment = storage.load()

    # nll 5000 logits with moving average
    mean_nll_5000, median_nll_5000, min_nll_5000, max_nll_5000 = experiment.calculate_nll(smoothing=True, window_size=11)
    print("mean_nll_5000:", mean_nll_5000)
    # crps 5000 logits
    mean_crps_5000, median_crps_5000, min_crps_5000, max_crps_5000 = experiment.calculate_crps()
    print("mean_crps_5000:", mean_crps_5000)
    scores = {
        'Metric': ['nll_5000', 'crps_5000'],
        'Mean': [mean_nll_5000, mean_crps_5000],
        'Median': [median_nll_5000, median_crps_5000],
        'Min': [min_nll_5000, min_crps_5000],
        'Max': [max_nll_5000, max_crps_5000]
    }

    df = pd.DataFrame(scores)

    if all_scores:

        mean_nll_5000_raw, median_nll_5000_raw, min_nll_5000_raw, max_nll_5000_raw = experiment.calculate_nll(smoothing=False, window_size=11)
        print("mean_nll_5000_raw:", mean_nll_5000_raw)

        # nll for the different interpolators
        mean_nll_linear, median_nll_linear, min_nll_linear, max_nll_linear = experiment.calculate_nll_with_interpolator(method="linear")
        print("mean_nll_linear:", mean_nll_linear)
        mean_nll_pchip, median_nll_pchip, min_nll_pchip, max_nll_pchip = experiment.calculate_nll_with_interpolator(method="pchip")
        print("mean_nll_pchip:", mean_nll_pchip)
        mean_nll_hybrid, median_nll_hybrid, min_nll_hybrid, max_nll_hybrid = experiment.calculate_nll_with_interpolator(method="hybrid")
        print("mean_nll_hybrid:", mean_nll_hybrid)
        # crps for the different interpolators
        mean_crps_linear, median_crps_linear, min_crps_linear, max_crps_linear = experiment.calculate_crps_with_interpolator(method="linear")
        print("mean_crps_linear:", mean_crps_linear)
        mean_crps_pchip, median_crps_pchip, min_crps_pchip, max_crps_pchip = experiment.calculate_crps_with_interpolator(method="pchip")
        print("mean_crps_pchip", mean_crps_pchip)
        mean_crps_hybrid, median_crps_hybrid, min_crps_hybrid, max_crps_hybrid = experiment.calculate_crps_with_interpolator(method="hybrid")
        print("mean_crps_hybrid", mean_crps_hybrid)

        # Adding interpolator scores to the dataframe
        interpolator_scores = {
            'Metric': [
                'nll_5000_raw', 'nll_linear', 'nll_pchip', 'nll_hybrid',
                'crps_linear', 'crps_pchip', 'crps_hybrid'
            ],
            'Mean': [
                mean_nll_5000_raw, mean_nll_linear, mean_nll_pchip, mean_nll_hybrid,
                mean_crps_linear, mean_crps_pchip, mean_crps_hybrid
            ],
            'Median': [
                median_nll_5000_raw, median_nll_linear, median_nll_pchip, median_nll_hybrid,
                median_crps_linear, median_crps_pchip, median_crps_hybrid
            ],
            'Min': [
                min_nll_5000_raw, min_nll_linear, min_nll_pchip, min_nll_hybrid,
                min_crps_linear, min_crps_pchip, min_crps_hybrid
            ],
            'Max': [
                max_nll_5000_raw, max_nll_linear, max_nll_pchip, max_nll_hybrid,
                max_crps_linear, max_crps_pchip, max_crps_hybrid
            ]
        }

        # Concatenating the interpolator scores
        interpolator_df = pd.DataFrame(interpolator_scores)
        df = pd.concat([df, interpolator_df], ignore_index=True)

    file_name = f"{storage_path}/experiment_results_{experiment_id}.pkl"
    storage = ExperimentStorage(file_name)
    storage.save(df)


    return df