import pandas as pd
import numpy as np
from scipy.stats import linregress
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np

import numpy as np

def fit_one_cathode(hist_counts, bin_centers, use_auto_bounds=True):
    """
    Fit a cathode histogram using linear fit on the Compton edge.
    This replicates the C++ MapADCToKeV::fitCathode logic.
    
    Parameters:
    -----------
    hist_counts : np.ndarray
        Histogram bin counts (length = NUM_BINS)
    bin_centers : np.ndarray
        Centers of each histogram bin
    use_auto_bounds : bool
        If True, uses 10%-90% bounds (default C++ behavior)
        If False, uses 5%-30% bounds (alternative C++ behavior)
        
    Returns:
    --------
    slope, intercept, photopeak_adc, (range_lo, range_hi)
    or None if fitting fails
    """
    
    # ==== Step 1: Find the maximum bin ====
    if len(hist_counts) == 0 or np.max(hist_counts) == 0:
        return None
    
    max_bin_index = np.argmax(hist_counts)
    max_height = hist_counts[max_bin_index]
    
    if max_height == 0:
        return None
    
    # ==== Step 2: Calculate height thresholds ====
    if use_auto_bounds:
        lo_height = max_height * 0.10  # 10% of max
        hi_height = max_height * 0.90  # 90% of max
    else:
        lo_height = max_height * 0.05  # 5% of max
        hi_height = max_height * 0.30  # 30% of max
    
    # ==== Step 3: Find bin indices for lo and hi heights ====
    lo_height_index = 0
    hi_height_index = 0
    
    for i in range(len(hist_counts)):
        # Keep updating loHeightIndex while bins are <= loHeight
        # This gives us the LAST bin at or below loHeight
        if hist_counts[i] <= lo_height:
            lo_height_index = i
        
        # Stop at FIRST bin above hiHeight
        if hist_counts[i] > hi_height:
            hi_height_index = i
            break
    
    # ==== Step 4: Convert bin indices to ADC range ====
    range_lo = bin_centers[lo_height_index]
    range_hi = bin_centers[hi_height_index]
    
    # ==== Step 5: Extract data in fitting range ====
    mask = (bin_centers >= range_lo) & (bin_centers <= range_hi)
    x_fit = bin_centers[mask]
    y_fit = hist_counts[mask]
    
    # Safety check: need at least 3 points to fit
    if len(x_fit) < 3:
        return None
    
    # ==== Step 6: Perform linear fit (pol1: y = slope*x + intercept) ====
    # Using numpy polyfit with degree 1
    # Note: polyfit returns [slope, intercept] for degree 1
    try:
        coeffs = np.polyfit(x_fit, y_fit, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
    except:
        return None
    
    # ==== Step 7: Calculate photopeak ADC position ====
    # This is where the fitted line crosses zero: y = 0 = slope*x + intercept
    # Solving for x: x = -intercept/slope
    if slope == 0:
        return None
    
    photopeak_adc = -intercept / slope
    
    # Sanity check: photopeak should be positive and within reasonable range
    if photopeak_adc < 0 or photopeak_adc > 4095:
        return None
    
    return slope, intercept, photopeak_adc, (range_lo, range_hi)


# Helper function for debugging/visualization
def plot_cathode_fit(hist_counts, bin_centers, fit_result):
    """
    Visualize the cathode fit for debugging.
    
    Parameters:
    -----------
    hist_counts : np.ndarray
        Histogram bin counts
    bin_centers : np.ndarray
        Centers of histogram bins
    fit_result : tuple or None
        Result from fit_one_cathode()
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.step(bin_centers, hist_counts, where='mid', label='Cathode Histogram', color='blue')
    
    if fit_result is not None:
        slope, intercept, photopeak_adc, (range_lo, range_hi) = fit_result
        
        # Plot fitting region
        mask = (bin_centers >= range_lo) & (bin_centers <= range_hi)
        x_fit = bin_centers[mask]
        y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear Fit')
        
        # Mark the photopeak position
        plt.axvline(photopeak_adc, color='green', linestyle='--', 
                   label=f'Photopeak ADC: {photopeak_adc:.1f}')
        
        # Mark fitting range
        plt.axvline(range_lo, color='orange', linestyle=':', alpha=0.5)
        plt.axvline(range_hi, color='orange', linestyle=':', alpha=0.5)
        
        plt.title(f'Cathode Fit: Photopeak at {photopeak_adc:.1f} ADC')
    else:
        plt.title('Cathode Fit: FAILED')
    
    plt.xlabel('ADC')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()