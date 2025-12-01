import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from anode_fit import photopeak_model

import numpy as np
from scipy.optimize import curve_fit
from anode_fit import photopeak_model

def fit_one_channel3(bin_centers, hist, auto_region=True, lower_bound=None, upper_bound=None):
    """
    Fit histogram with robust bounds checking and quality filters.
    """
    
    # ==== Quality Check 1: Minimum total counts ====
    total_counts = np.sum(hist)
    if total_counts < 100:
        print(f"  ✗ Insufficient statistics: {total_counts:.0f} total counts")
        return None, None, None, None
    
    if auto_region:
        # Find the peak
        peak_idx = np.argmax(hist)
        peak_pos = bin_centers[peak_idx]
        peak_height = hist[peak_idx]
        
        # ==== Quality Check 2: Minimum peak height ====
        if peak_height < 10:
            print(f"  ✗ Peak too low: {peak_height:.0f} counts")
            return None, None, None, None
        
        # Estimate width from FWHM
        half_max = peak_height / 2
        above_half = hist > half_max
        
        if np.any(above_half):
            left_idx = np.where(above_half)[0][0]
            right_idx = np.where(above_half)[0][-1]
            estimated_width = bin_centers[right_idx] - bin_centers[left_idx]
        else:
            estimated_width = 200
        
        # ==== Quality Check 3: Peak width sanity ====
        # A real photopeak should have FWHM between ~20-400 ADC units
        if estimated_width < 10 or estimated_width > 800:
            print(f"  ✗ Unrealistic peak width: {estimated_width:.0f} ADC")
            return None, None, None, None
        
        # Wider range: peak ± 10*width to capture tails
        lower_bound = max(bin_centers[0], peak_pos - 10 * estimated_width)
        upper_bound = min(bin_centers[-1], peak_pos + 5 * estimated_width)
        
        # ==== Quality Check 4: Fitting range sanity ====
        fitting_range = upper_bound - lower_bound
        if fitting_range > 2000:  # More than half the ADC range is suspicious
            print(f"  ✗ Fitting range too wide: {fitting_range:.0f} ADC")
            return None, None, None, None
        
        print(f"Auto-detected region: [{lower_bound:.0f}, {upper_bound:.0f}]")
    
    # Apply bounds
    mask = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
    hist_fit = hist[mask]
    bins_fit = bin_centers[mask]
    
    # Background estimate
    background_est = np.median(hist_fit[:min(20, len(hist_fit)//4)])
    
    # Lower threshold
    threshold = max(0.5, background_est)
    
    significant_mask = hist_fit > threshold
    n_significant = np.sum(significant_mask)
    
    print(f"Bins in range: {len(bins_fit)}, Significant bins: {n_significant}")
    print(f"Background estimate: {background_est:.2f}, Threshold: {threshold:.2f}")
    
    # Require at least 25 significant bins
    min_bins = 25
    if n_significant < min_bins:
        print(f"Warning: Only {n_significant} significant bins (need >{min_bins})")
        # Try with all bins, but this is risky
        if len(bins_fit) < min_bins:
            print(f"  ✗ Even total bins ({len(bins_fit)}) < minimum")
            return None, None, None, None
    
    # Initial guesses
    H0 = hist_fit.max()
    x0_0 = bins_fit[np.argmax(hist_fit)]
    
    # Sigma from FWHM
    half_max = H0 / 2
    above_half = hist_fit > half_max
    if np.sum(above_half) > 2:
        fwhm = np.sum(above_half) * (bins_fit[1] - bins_fit[0])
        sigma0 = fwhm / 2.355
    else:
        sigma0 = 50
    
    sigma0 = max(sigma0, 20)  # Minimum sigma = 20
    sigma0 = min(sigma0, 200)  # Maximum sigma = 200 (cap for sanity)
    
    sigma_e0 = sigma0 * 2
    t0 = sigma0 * 1.5
    h0 = 0.3
    B0 = max(0.1, background_est)
    
    # ==== CRITICAL: Ensure initial guesses are within bounds ====
    # We need to make sure our bounds can accommodate the initial guesses
    
    # Dynamic bounds based on initial guesses
    H_low = max(0.01 * H0, 0.1)
    H_high = max(5 * H0, 100)
    
    x0_low = max(bin_centers[0], x0_0 - 300)
    x0_high = min(bin_centers[-1], x0_0 + 300)
    
    sigma_low = 1
    sigma_high = max(300, sigma0 * 3)  # Make sure it can fit initial guess
    
    sigma_e_low = 10
    sigma_e_high = max(500, sigma_e0 * 2)  # Make sure it can fit initial guess
    
    t_low = 1
    t_high = max(300, t0 * 2)  # Make sure it can fit initial guess
    
    h_low = 0
    h_high = 5
    
    B_low = 0
    B_high = max(10, B0 * 10)
    
    p0 = [H0, x0_0, sigma0, sigma_e0, t0, h0, B0]
    
    bounds = (
        [H_low, x0_low, sigma_low, sigma_e_low, t_low, h_low, B_low],
        [H_high, x0_high, sigma_high, sigma_e_high, t_high, h_high, B_high]
    )
    
    # Verify initial guesses are within bounds
    for i, (param, p_val, low, high) in enumerate(zip(
        ['H', 'x0', 'σ', 'σe', 't', 'h', 'B'],
        p0, bounds[0], bounds[1]
    )):
        if not (low <= p_val <= high):
            print(f"  ✗ Initial guess {param}={p_val:.2f} outside bounds [{low:.2f}, {high:.2f}]")
            return None, None, None, None
    
    print(f"\nInitial guesses:")
    print(f"  H={H0:.1f}, x0={x0_0:.1f}, σ={sigma0:.1f}")
    print(f"  σe={sigma_e0:.1f}, t={t0:.1f}, h={h0:.2f}, B={B0:.2f}")
    
    try:
        # Use Poisson weights
        weights = 1.0 / np.sqrt(hist_fit + 1)
        
        popt, pcov = curve_fit(
            photopeak_model,
            bins_fit, hist_fit,
            p0=p0,
            bounds=bounds,
            sigma=weights,
            absolute_sigma=False,
            maxfev=30000,
            method='trf'
        )
        
        print(f"  ✓ Fit succeeded")
        return popt, pcov, bins_fit, hist_fit
        
    except RuntimeError as e:
        print(f"  ✗ Fit failed: {e}")
        return None, None, None, None
    except ValueError as e:
        print(f"  ✗ Bounds error: {e}")
        return None, None, None, None

def fit_one_channel2(bin_centers, hist, auto_region=True, lower_bound=None, upper_bound=None):
    """
    Fit histogram with adjusted thresholds for low-statistics data.
    """
    if auto_region:
        # Find the peak from values
        
        #find max that lies above ADC 2100
        peak_idx = np.argmax(hist[bin_centers > 2100])
        peak_pos = bin_centers[peak_idx]
        
        # Wider range: peak ± 10*width to capture tails
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
    threshold = max(0.5, background_est)  # Much more permissive
    
    significant_mask = hist_fit > threshold
    n_significant = np.sum(significant_mask)
    
    print(f"Bins in range: {len(bins_fit)}, Significant bins: {n_significant}")
    print(f"Background estimate: {background_est:.2f}, Threshold: {threshold:.2f}")
    
    # Require at least 3x parameters
    min_bins = 25  # 3-4x the number of parameters
    if n_significant < min_bins:
        print(f"Warning: Only {n_significant} significant bins (need >{min_bins})")
        print("Trying with all bins in range...")
        # Use all bins in range instead
        hist_fit_all = hist_fit
        bins_fit_all = bins_fit
        n_significant = len(bins_fit_all)
    else:
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
        sigma0 = 50  # Default for wide peaks
    
    sigma0 = max(sigma0, 20)  # Minimum sigma = 20
    
    sigma_e0 = sigma0 * 2
    t0 = sigma0 * 1.5
    h0 = 0.3
    B0 = max(0.1, background_est)
    
    p0 = [H0, x0_0, sigma0, sigma_e0, t0, h0, B0]
    
    print(f"\nInitial guesses:")
    print(f"  H={H0:.1f}, x0={x0_0:.1f}, σ={sigma0:.1f}")
    print(f"  σe={sigma_e0:.1f}, t={t0:.1f}, h={h0:.2f}, B={B0:.2f}")
    
    # Setting bounds for curve_fit()
    
    #making sure that for edge cases, lower bound < upper bound
    if H0 == 0:
        H_low = 0
        H_high = 100
    else:
        H_low = H0 * 0.01
        H_high = H0 * 5
        
    bounds = (
        [H_low,   x0_0-200,  1,    10,   1,    0,     0],
        [H_high,      x0_0+200,  300,  500,  300,  5,     max(10, B0*10)]
    )
    
    try:
        # Use Poisson weights
        weights = 1.0 / np.sqrt(hist_fit_all + 1)
        
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