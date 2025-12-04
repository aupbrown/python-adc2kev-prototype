import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from anode_fit import photopeak_model
import multiprocessing as mp

def _worker_two_point_calibration(args):
    """
    Multiprocessing worker function that performs the two-point fit and calibration 
    for a single channel. This contains the logic from the user's main loop.
    
    Args:
        args (tuple): (channel_key, hist1, bin_centers1, kev1, hist2, bin_centers2, kev2, fit_func, peak_func)
    """
    channel, hist1, bin_centers1, kev1, hist2, bin_centers2, kev2, fit_func, peak_func = args
    
    # 1. Skip bad histograms (Low stats/Single spike checks - based on user's main script)
    if np.count_nonzero(hist1) <= 1 or np.count_nonzero(hist2) <= 1:
        return (channel, None, "Single spike/Zero counts")
    
    if np.sum(hist1) < 100 or np.sum(hist2) < 100:
        return (channel, None, "Low statistics")

    # 2. Fit Histogram 1 (Isotope 1)
    # Using fixed bounds from user's main script for 662 keV source (file1)
    popt1, _, _, _ = fit_func(
        bin_centers1, hist1, auto_region=False, 
        lower_bound=1800, upper_bound=3200
    )
    
    # 3. Fit Histogram 2 (Isotope 2)
    # Using fixed bounds from user's main script for 511 keV source (file2)
    popt2, _, _, _ = fit_func(
        bin_centers2, hist2, auto_region=False,
        lower_bound=1500, upper_bound=2800
    )
    
    if popt1 is None or popt2 is None:
        return (channel, None, "Fit function failed")

    # 4. Extract Peak ADC Values
    peak_info1 = peak_func(popt1, None) # pcov not needed for peak extraction
    peak_info2 = peak_func(popt2, None)
    adc1 = peak_info1['photopeak_position']
    adc2 = peak_info2['photopeak_position']
    
    # 5. Validation Checks (from user's main script)
    if adc1 <= 0 or adc2 <= 0:
        return (channel, None, "Invalid peak ADC")
    
    # Ensure peaks are separated enough (user's check: abs(adc1 - adc2) < 50)
    if abs(adc1 - adc2) < 50:
        return (channel, None, "Peaks too close (Separation < 50 ADC)")
    
    # 6. Calculate Calibration Parameters (Linear Fit: keV = slope * ADC + intercept)
    
    # Calculate slope: (y2 - y1) / (x2 - x1)
    # y = keV, x = ADC
    slope = (kev1 - kev2) / (adc1 - adc2)
    
    # Calculate intercept: y - slope * x
    intercept = kev2 - slope * adc2 
    
    # 7. Final Validation Checks (from user's main script)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0 or slope > 2:
        return (channel, None, "Invalid slope/intercept")
    
    # Success: Return the channel key and the serializable calibration parameters
    return (channel, (float(slope), float(intercept)), "Success")


def calibrate_anodes_in_parallel(
    anode_hists1, anode_bin_centers1, kev1, 
    anode_hists2, anode_bin_centers2, kev2, 
    fit_func, peak_func, verbose=True
):
    """
    Manages the parallel execution of the two-point calibration process 
    for all common anode channels.
    """
    
    common_anode_channels = set(anode_hists1.keys()).intersection(set(anode_hists2.keys()))
    
    if verbose:
        print(f"\nStarting parallel two-point calibration for {len(common_anode_channels)} common channels...")
    
    # 1. Prepare tasks list for multiprocessing
    tasks = []
    for channel in common_anode_channels:
        # Args for the worker function: (channel, hist1, bc1, kev1, hist2, bc2, kev2, fit_func, peak_func)
        tasks.append((
            channel, 
            anode_hists1[channel], anode_bin_centers1, kev1, 
            anode_hists2[channel], anode_bin_centers2, kev2, 
            fit_func, peak_func
        ))
    
    # 2. Execute parallel processing
    num_cores = mp.cpu_count()
    if verbose:
        print(f"Using {num_cores} cores for parallel processing...")
        
    calibrations = {}
    failed_fits = 0
    
    try:
        # Using fork context for compatibility, especially on Unix/Linux
        with mp.get_context('fork').Pool(num_cores) as pool:
            # imap_unordered is best for distributing independent tasks
            results = pool.imap_unordered(_worker_two_point_calibration, tasks)
            
            count = 0
            for channel, result_tuple, reason in results:
                count += 1
                if result_tuple is not None:
                    # result_tuple is (slope, intercept)
                    calibrations[channel] = result_tuple
                else:
                    failed_fits += 1
                    # Optional: log the reason for failure here if needed
                
                if verbose and count % 1000 == 0:
                    print(f"  Processed {count}/{len(common_anode_channels)} channels...")
                    
    except Exception as e:
        print(f"FATAL ERROR during parallel processing: {e}")
        # Return whatever was processed so far
        return calibrations, len(common_anode_channels) - len(calibrations)

    if verbose:
        print(f"Parallel calibration complete. Successfully calibrated {len(calibrations)} channels.")
        
    return calibrations, failed_fits

def fit_one_channel2(bin_centers, hist, auto_region=True, lower_bound=None, upper_bound=None):
    """
    FIXED VERSION: Ensures all initial guesses are within bounds.
    """
    if auto_region:
        # Find the peak by searching for max bin idx that lies above ADC value 2100
        peak_idx = np.argmax(hist[bin_centers > 2100])
        peak_pos = bin_centers[peak_idx]
        
        # Wider range: peak ± 200 to capture tails
        lower_bound = peak_pos - 200
        upper_bound = peak_pos + 200
        
        print(f"Auto-detected region: [{lower_bound:.0f}, {upper_bound:.0f}]")
    
    # Apply bounds
    mask = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
    hist_fit = hist[mask]
    bins_fit = bin_centers[mask]
    
    # RELAXED threshold for low-statistics data
    background_est = np.median(hist_fit[:min(20, len(hist_fit)//4)])
    
    # Lower threshold: just require counts > 0 OR background level
    threshold = max(0.5, background_est)
    
    significant_mask = hist_fit > threshold
    n_significant = np.sum(significant_mask)
    
    print(f"Bins in range: {len(bins_fit)}, Significant bins: {n_significant}")
    print(f"Background estimate: {background_est:.2f}, Threshold: {threshold:.2f}")
    
    # Require at least 25 significant bins
    min_bins = 25
    if n_significant < min_bins:
        print(f"Warning: Only {n_significant} significant bins (need >{min_bins})")
        print("Trying with all bins in range...")
    
    hist_fit_all = hist_fit
    bins_fit_all = bins_fit
    
    # Initial guesses
    H0 = hist_fit_all.max()
    x0_0 = bins_fit_all[np.argmax(hist_fit_all)]
    
    # Sigma from FWHM
    half_max = H0 / 2
    above_half = hist_fit_all > half_max
    if np.sum(above_half) > 2:
        fwhm = np.sum(above_half) * (bins_fit_all[1] - bins_fit_all[0])
        sigma0 = fwhm / 2.355
    else:
        sigma0 = 50
    
    sigma0 = max(sigma0, 20)
    
    sigma_e0 = sigma0 * 2
    t0 = sigma0 * 1.5
    h0 = 0.3
    B0 = max(0.1, background_est)
    
    # FIX: Ensure t0 is within reasonable bounds
    t0 = min(t0, 250)  # Cap t0 so it doesn't exceed upper bound
    
    p0 = [H0, x0_0, sigma0, sigma_e0, t0, h0, B0]
    
    print(f"\nInitial guesses:")
    print(f"  H={H0:.1f}, x0={x0_0:.1f}, σ={sigma0:.1f}")
    print(f"  σe={sigma_e0:.1f}, t={t0:.1f}, h={h0:.2f}, B={B0:.2f}")
    
    # Setting bounds for curve_fit()
    if H0 == 0:
        H_low = 0
        H_high = 100
    else:
        H_low = H0 * 0.01
        H_high = H0 * 5
    
    # Ensure sigma_e0 fits within bounds
    sigma_e_high = max(500, sigma_e0 * 1.2)
    
    # Ensure t0 fits within bounds  
    t_high = max(300, t0 * 1.2)
    
    bounds = (
        [H_low,   x0_0-200,  1,    10,   1,      0,   0],
        [H_high,  x0_0+200,  300,  sigma_e_high,  t_high,  5,   max(10, B0*10)]
    )
    
    # Verify all initial guesses are within bounds
    param_names = ['H', 'x0', 'σ', 'σe', 't', 'h', 'B']
    for i, (name, val, low, high) in enumerate(zip(param_names, p0, bounds[0], bounds[1])):
        if not (low <= val <= high):
            print(f"⚠️  Initial {name}={val:.2f} outside bounds [{low:.2f}, {high:.2f}]")
            p0[i] = np.clip(val, low, high)
            print(f"   Clipped to {p0[i]:.2f}")
    
    try:
        
        # Use Poisson weights
        weights = 1.0 / np.sqrt(hist_fit_all + 1)
        
        # Dummy photopeak model - replace with your actual model
        def photopeak_model(x, H, x0, sigma, sigma_e, t, h, B):
            return H * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + B
        
        popt, pcov = curve_fit(
            photopeak_model,
            bins_fit_all, hist_fit_all,
            p0=p0,
            bounds=bounds,
            sigma=weights,
            absolute_sigma=False,
            maxfev=30000,
            method='trf'
        )
        
        return popt, pcov, bins_fit_all, hist_fit_all
        
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None, None, None, None
    except ValueError as e:
        print(f"Bounds error: {e}")
        return None, None, None, None

def check_continuity(popt):
    """
    Check if the model has a discontinuity at the transition.
    """
    H, x0, sigma, sigma_e, t, h, B = popt
    
    transition = x0 - t
    
    # Evaluate just above and below transition
    x_above = transition + 0.001
    x_below = transition - 0.001
    
    F_above = photopeak_model(np.array([x_above]), *popt)[0]
    F_below = photopeak_model(np.array([x_below]), *popt)[0]
    
    ratio = F_below / F_above if F_above > 0 else np.inf
    
    print(f"\n>>> CONTINUITY CHECK at x={transition:.1f}:")
    print(f"    F(transition + ε) = {F_above:.2f}")
    print(f"    F(transition - ε) = {F_below:.2f}")
    print(f"    Ratio = {ratio:.4f} (should be ~1.0)")
    
    if abs(ratio - 1.0) > 0.1:
        print("    ⚠ WARNING: Large discontinuity detected!")
        return False
    else:
        print("    ✓ Continuity looks good")
        return True
    
def plot_fit_with_residuals(bins_fit, hist_fit, popt, key):
    """
    Plot fit with residuals subplot.
    """
    H, x0, sigma, sigma_e, t, h, B = popt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot
    F = photopeak_model(bins_fit, *popt)
    
    ax1.step(bins_fit, hist_fit, where='mid', label="Data", linewidth=1.5, color='black')
    ax1.plot(bins_fit, F, label="Fit", color="red", linewidth=2)
    ax1.axvline(x0, color='orange', linestyle='--', alpha=0.7, label=f'Peak x₀={x0:.1f}')
    ax1.axvline(x0-t, color='green', linestyle='--', alpha=0.7, label=f'Transition={x0-t:.1f}')
    
    ax1.set_ylabel("Counts", fontsize=12)
    ax1.set_title(f"Photopeak Fit - Channel {key}", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=0.5)
    
    # Residuals
    residuals = hist_fit - F
    sigma_i = np.sqrt(hist_fit)
    sigma_i = np.where(sigma_i > 0, sigma_i, 1)
    normalized_residuals = residuals / sigma_i
    
    ax2.scatter(bins_fit, normalized_residuals, s=10, alpha=0.6, color='blue')
    ax2.axhline(0, color='red', linestyle='-', linewidth=1.5)
    ax2.axhline(3, color='orange', linestyle='--', alpha=0.5, label='±3σ')
    ax2.axhline(-3, color='orange', linestyle='--', alpha=0.5)
    ax2.fill_between(bins_fit, -1, 1, alpha=0.2, color='green', label='±1σ')
    
    ax2.set_xlabel("ADC Value", fontsize=12)
    ax2.set_ylabel("Normalized Residuals", fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([-5, 5])
    
    plt.tight_layout()
    plt.show()
    
def extract_photopeak_info(popt, pcov=None):
    """
    Extract photopeak information from fit parameters.
    
    Parameters:
    -----------
    popt : array
        Fitted parameters [H, x0, sigma, sigma_e, t, h, B]
    pcov : array, optional
        Covariance matrix from curve_fit
        
    Returns:
    --------
    dict with photopeak information
    """
    H, x0, sigma, sigma_e, t, h, B = popt
    
    # Photopeak position
    photopeak_position = x0
    
    # Photopeak position uncertainty (if covariance available)
    if pcov is not None:
        photopeak_uncertainty = np.sqrt(pcov[1, 1])  # Uncertainty in x0
    else:
        photopeak_uncertainty = None
    
    # Energy resolution (FWHM)
    # FWHM = 2.355 * sigma (for Gaussian)
    fwhm = 2.355 * sigma
    
    # Energy resolution as percentage
    resolution_percent = (fwhm / x0) * 100 if x0 > 0 else np.inf
    
    # Peak height (total counts at peak)
    peak_height = H + B
    
    # Net peak area (integral of Gaussian component only, approximately)
    # For a Gaussian: Area = H * sigma * sqrt(2*pi)
    net_peak_area = H * sigma * np.sqrt(2 * np.pi)
    
    results = {
        'photopeak_position': photopeak_position,
        'photopeak_uncertainty': photopeak_uncertainty,
        'fwhm': fwhm,
        'resolution_percent': resolution_percent,
        'peak_height': peak_height,
        'net_peak_area': net_peak_area,
        'sigma': sigma,
        'background': B
    }
    
    return results


def print_photopeak_results(results, source_energy_kev=None):
    """
    Print photopeak analysis results in a readable format.
    
    Parameters:
    -----------
    results : dict
        Output from extract_photopeak_info()
    source_energy_kev : float, optional
        Known energy of the source (e.g., 662 keV for Cs-137)
        If provided, calculates the ADC-to-keV calibration
    """
    print("\n" + "="*60)
    print("PHOTOPEAK ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nPhotopeak Position (x₀):")
    print(f"  ADC value: {results['photopeak_position']:.2f}")
    if results['photopeak_uncertainty'] is not None:
        print(f"  Uncertainty: ± {results['photopeak_uncertainty']:.2f} ADC")
    
    if source_energy_kev is not None:
        calibration = source_energy_kev / results['photopeak_position']
        print(f"\nEnergy Calibration:")
        print(f"  {calibration:.4f} keV/ADC")
        print(f"  or {1/calibration:.2f} ADC/keV")
    
    print(f"\nEnergy Resolution:")
    print(f"  FWHM: {results['fwhm']:.2f} ADC")
    print(f"  Resolution: {results['resolution_percent']:.2f}%")
    
    print(f"\nPeak Characteristics:")
    print(f"  Peak height: {results['peak_height']:.0f} counts")
    print(f"  Net peak area: {results['net_peak_area']:.0f} counts")
    print(f"  Sigma (σ): {results['sigma']:.2f} ADC")
    print(f"  Background: {results['background']:.2f} counts/bin")
    
    print("="*60)


def find_all_photopeaks(histograms, bin_centers, source_energy_kev=662):
    """
    Fit all channels and extract photopeak positions.
    
    Parameters:
    -----------
    histograms : dict
        Dictionary of {channel_key: histogram_array}
    bin_centers : array
        ADC bin centers
    source_energy_kev : float
        Known source energy (default: 662 for Cs-137)
        
    Returns:
    --------
    DataFrame with photopeak info for all channels
    """
    results_list = []
    
    for channel_key, hist in histograms.items():
        print(f"\nProcessing channel {channel_key}...")
        
        # Fit the channel
        popt, pcov = fit_one_channel2(bin_centers, hist, 
                                       lower_bound=0, 
                                       upper_bound=bin_centers.max())
        
        if popt is None:
            print(f"  Fit failed for channel {channel_key}")
            continue
        
        # Extract photopeak info
        peak_info = extract_photopeak_info(popt, pcov)
        
        # Calculate calibration
        calibration = source_energy_kev / peak_info['photopeak_position']
        
        # Store results
        results_list.append({
            'channel': channel_key,
            'photopeak_adc': peak_info['photopeak_position'],
            'photopeak_unc': peak_info['photopeak_uncertainty'],
            'fwhm_adc': peak_info['fwhm'],
            'resolution_pct': peak_info['resolution_percent'],
            'calibration_kev_per_adc': calibration,
            'net_area': peak_info['net_peak_area'],
            'background': peak_info['background']
        })
    
    # Convert to DataFrame for easy analysis
    import pandas as pd
    df = pd.DataFrame(results_list)
    
    return df