from dataclasses import dataclass
import numpy as np
from scipy.stats import rv_continuous, rv_histogram
from scipy.interpolate import interp1d
from scipy.special import softmax
from tabpfn import TabPFNRegressor



class CRPSMixin:
    def crps(self, y):
        """
        Calculate the Continuous Ranked Probability Score (CRPS) for a given sample `y`.
        
        Parameters:
        - y: np.array of shape (n,): Values where CRPS is to be computed.
        
        Returns:
        - crps_score: The CRPS score.
        """
        cdfs = np.cumsum(self.probits, axis=1)

        indicator_f = lambda x: x >= y[:, np.newaxis]
        indicators = indicator_f(self.borders[1:])

        crps_scores = np.sum((cdfs - indicators)**2 * self.bin_widths, axis=1)

        return crps_scores
    


class LinearInterpolatedDistWithCRPS(rv_continuous, CRPSMixin):
    def __init__(self, logits, borders, *args, **kwargs):
        """
        Args:
        - logits (np.ndarray): An array of shape (n, 5000) representing logits for n samples.
        - borders (np.ndarray): An array of shape (5001,) representing the bin edges for each sample.
        """
        self.logits = logits
        self.probits = softmax(logits, axis=1) # == np.exp(x)/sum(np.exp(x))
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
    
    def _pdf(self, x):
        pdf = interp1d(self.bin_midpoints, self.probits/self.bin_widths, kind='linear')
        return pdf


class HistogramWithCRPS(rv_histogram, CRPSMixin):
    pass

def get_empirical_dists(logits: np.ndarray, borders: np.ndarray) -> list[HistogramWithCRPS]:
    """
    Create a list of HistogramWithCRPS objects for each of the logits.

    Args:
    - logits (np.ndarray): An array of shape (n, 5000) representing logits for n samples.
    - borders (np.ndarray): An array of shape (n, 5001) representing the bin edges for each sample.

    Returns:
    - list: A list of HistogramWithCRPS objects of length n.
    """
    n = logits.shape[0]
    histograms = []
    
    # Iterate over the logits (axis 1: 5000 logits)
    for i in range(n):
        hist = HistogramWithCRPS((logits[i, :], borders[i, :]), density=False)
        histograms.append(hist)
    
    return histograms

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

    def perform(self):
        # Train model
        model = TabPFNRegressor(device=self.device, fit_mode=self.fit_mode, random_state=self.random_state, ignore_pretraining_limits=self.ignore_pretraining_limits)
        model.fit(self.X_train, self.y_train)

        # Define custom quantiles (from 0.1 to 0.9)
        quantiles_custom = np.arange(0.1, 1, 0.1)

        # Predict with custom quantiles
        probs_val_q = model.predict(self.X_validation, output_type="full", quantiles=quantiles_custom)

        # Extract relevant data from the prediction
        logits_q = probs_val_q["logits"]  # logits (N, 5000)
        borders_q = probs_val_q["criterion"].borders  # borders (5001,)
        all_quantiles_q = np.array(probs_val_q["quantiles"])  # quantiles (N, 9)
        self.logits = logits_q
        self.borders = borders_q
        self.deciles = all_quantiles_q




class ExperimentTracker:
    def __init__(self):
        self.experiments = []

    def track(self, X_train, y_train, X_validation, y_validation, 
            device: str = "auto", fit_mode: str = "low_memory", random_state: int = 42, ignore_pretraining_limits=False) -> Experiment:
        experiment = Experiment(X_train=X_train, y_train=y_train, X_validation=X_validation, y_validation=y_validation, device=device, fit_mode=fit_mode, random_state=random_state, ignore_pretraining_limits=ignore_pretraining_limits)
        experiment.perform()
        self.experiments.append(experiment)
        return experiment
        

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
        hist = HistogramWithCRPS((experiment.logits[i, :], experiment.borders), density=False)
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
        hist = LinearInterpolatedDistWithCRPS(experiment.logits[i, :], experiment.borders)
        histograms.append(hist)
    
    return histograms

tracker = ExperimentTracker()
experiment = tracker.track(train_X, train_y, validation_X, validation_y)

dists = get_linear_interpolated_dists_from(experiment)
dists[0].crps(np.array([5]))

hists = get_histograms_from(experiment)
hists[0].crps(5)