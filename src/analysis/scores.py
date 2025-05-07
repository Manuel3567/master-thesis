import numpy as np
import torch


def compute_crps_pytorch(logits, bin_edges, y_values):
    """
    Computes the CRPS for multiple rows of logits and corresponding y-values using PyTorch.

    Args:
        logits: Tensor of shape (N, 5000) - unnormalized logits for each row.
        bin_edges: Tensor of shape (5001,) - common bin edges for all rows.
        y_values: Tensor of shape (N,) - target values for each row.

    Returns:
        Tensor of shape (N,) containing the CRPS values for each row.
    """

    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)  # (N, 5000)

    # Compute CDF (cumulative sum of probabilities)
    cdf = torch.cumsum(probs, dim=1)  # (N, 5000)

    # Compute the indicator function (1 if bin edge >= y, else 0)
    # We need to compare each y_value with bin_edges and broadcast correctly
    indicators = (bin_edges[1:].unsqueeze(0) >= y_values.unsqueeze(1)).float()  # (N, 5000)

    # Step 4: Compute bin widths
    bin_widths = (bin_edges[1:] - bin_edges[:-1]).unsqueeze(0)  # (1, 5000)

    # Step 5: Compute CRPS integral for each row
    crps = torch.sum((cdf - indicators) ** 2 * bin_widths, dim=1)  # (N,)

    return crps


def compute_nll_smoothed_5000_logits(logits: torch.Tensor, borders: torch.Tensor, y_values: torch.Tensor, window_size = 5):
    """
    Computes the smoothed negative log-likelihood (NLL) for each value in `y_values` using 5000 logits and 5001 borders by
    applying a moving average smoothing to the pdf using a specified window size.

    Ensure that borders is a 1D tensor of shape [8640] and not a 2D tensor e.g. [5001, 1]

    Args:
        logits (torch.Tensor): Logits tensor of shape (N, 5000), where N is the number of samples and 5000 is the number of bins.
        borders (torch.Tensor): Array containing bin edges of length M+1, used for binning the values in `y_values`.
        y_values (torch.Tensor): y-values to compute the NLL for. Each value should correspond to a bin in the `borders` array.
        window_size (int): Window size for the moving average smoothing. Defaults to 5. If even, it is adjusted to the nearest odd number.

    Returns:
        list of smoothed NLL values for each entry in y_values
    """

    probs = torch.nn.functional.softmax(logits, dim=1)  # Convert logits to PMF
    bin_widths = np.diff(borders.squeeze().T)  # Difference between consecutive bin edges
    pdfs = probs / bin_widths  # Convert PMF to PDF
    epsilon = 1e-10  # Avoid log(0)

    if window_size % 2 == 0:
        window_size = window_size + 1

    #window_size = 5  # Example: considering a window of 5 bins (2 before, 2 after)
    kernel = np.ones(window_size) / window_size  # Simple moving average kernel
    rolling_mean_pdf = np.zeros_like(y_values) # initialize

    left_offset = int((window_size-1)/2)
    right_offset = left_offset + 1 # +1 because python column selection excludes

    count_significant_differences = 0  # Counter for significant changes

    smoothed_nlls = []

    #print("values where the difference between nll of the bin y lies and averaged nll is greater than 500%")

    for i in range(len(y_values)):
        bin_indices = torch.searchsorted(borders, y_values, right=True) -1 #right=True: return the index of the first bin where the value is greater than the element in y_values
        
        idx = bin_indices[i] #For each y_value, the corresponding bin index is obtained from bin_indices

        # Define the window boundaries for convolution 2 -> (window_size - 1)/2 assuming window_size is odd
        window_start = max(idx - left_offset, 0)
        window_end = min(idx + right_offset, pdfs.shape[1])  # Ensure the window stays within bounds; note 3 because select is inclusive on the left but exclusive on the right
        #window_start = max(idx - 2, 0)
        #window_end = min(idx + 3, pdfs.shape[1])

        # Extract the PDF values in the window
        pdf_window = pdfs[i, window_start:window_end]

        # Apply np.convolve with 'same' mode to get the rolling mean for this sample
        convolved_pdf = np.convolve(pdf_window, kernel, mode='same')

        rolling_mean_pdf[i] = convolved_pdf[len(convolved_pdf) // 2]  # Take the center value for the window
        #rolling_mean_pdf[i] = convolved_pdf[left_offset]
        
        # Compute NLL values
        nll_at_bin = -np.log(pdfs[i, idx] + epsilon)
        smoothed_nll = -np.log(rolling_mean_pdf[i] + epsilon)

        # Print only if the difference is greater than 500%
        if abs(nll_at_bin - smoothed_nll) / abs(nll_at_bin) > 5.0:
            count_significant_differences += 1
            #print(f"Sample {i}: y_value={y_values[i]:.4f}, bin={idx}, NLL at bin: {nll_at_bin:.4f}, Smoothed NLL: {smoothed_nll:.4f}")
    
        smoothed_nlls.append(smoothed_nll)

    #print(f"total mean: {np.mean(smoothed_nlls)}")

    #print(f"Total NLL: {np.mean(nll_values)}")
    #print(f"Total significant differences: {count_significant_differences}")

    return smoothed_nlls