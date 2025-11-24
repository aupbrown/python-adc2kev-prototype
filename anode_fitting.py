import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

def photopeak_model(x, H, x0, sigma, sigma_e, t, h, B):
    """
    Photopeak model WITH continuity enforcement at transition.
    """
    boundary = x0 - t
    mask_right = x >= boundary
    mask_left = ~mask_right
    
    F = np.zeros_like(x, dtype=float)
    
    # RIGHT SIDE (x >= x0 - t): Gaussian region
    if np.any(mask_right):
        xr = x[mask_right]
        gaussian = np.exp(-((xr - x0)**2) / (2 * sigma**2))
        erf_arg = -((xr - x0)**2) / (2 * sigma_e**2)
        erf_arg = np.clip(erf_arg, -10, 10)
        erf_term = erf(erf_arg)
        F[mask_right] = H * (gaussian + h * erf_term) + B
    
    # LEFT SIDE (x < x0 - t): Exponential tail with continuity fix
    if np.any(mask_left):
        xl = x[mask_left]
        
        # Calculate what the right-side formula gives at transition
        gauss_at_t = np.exp(-(t**2) / (2 * sigma**2))
        erf_arg_at_t = -(t**2) / (2 * sigma_e**2)
        erf_arg_at_t = np.clip(erf_arg_at_t, -10, 10)
        erf_at_t = erf(erf_arg_at_t)
        right_value_at_t = H * (gauss_at_t + h * erf_at_t) + B
        
        # Calculate what the left-side formula would give at transition
        exp_at_t_arg = (t * (x0 - t)) / (2 * sigma**2)
        exp_at_t_arg = np.clip(exp_at_t_arg, -100, 10)
        exp_at_t = np.exp(exp_at_t_arg)
        left_value_at_t = H * (exp_at_t + h * erf_at_t) + B
        
        # Scale factor to enforce continuity
        if abs(left_value_at_t) > 1e-10:
            continuity_scale = right_value_at_t / left_value_at_t
        else:
            continuity_scale = 1.0
        
        # Apply paper's formula with scaling
        exp_arg = (t * (2*xl - x0 + t)) / (2 * sigma**2)
        exp_arg = np.clip(exp_arg, -100, 10)
        exp_tail = np.exp(exp_arg)
        
        erf_arg = -((xl - x0)**2) / (2 * sigma_e**2)
        erf_arg = np.clip(erf_arg, -10, 10)
        erf_term = erf(erf_arg)
        
        F[mask_left] = continuity_scale * (H * (exp_tail + h * erf_term) + B)
    
    return F

def plot_fit_components(bin_centers, hist, popt, key):
    H, x0, sigma, sigma_e, t, h, B = popt
    x = bin_centers

    # Compute masks again for components
    boundary = x0 - t
    mask_right = x >= boundary
    mask_left  = ~mask_right

    gaussian = np.zeros_like(x)
    exp_tail = np.zeros_like(x)
    erf_component = np.zeros_like(x)
    background = np.ones_like(x) * B

    # Gaussian component on right side
    gaussian[mask_right] = H * np.exp(-((x[mask_right] - x0)**2) / (2*sigma**2))

    # Exponential tail on left side
    exp_arg = (t*(2*x[mask_left] - x0 + t)) / (2*sigma**2)
    exp_tail[mask_left] = H * np.exp(exp_arg - np.max(exp_arg))  # normalized

    # Error function exists on both sides
    erf_component = H*h*erf(-(x - x0) / (np.sqrt(2)*sigma_e))

    # Background
    background = np.ones_like(x) * B

    # Full model
    F = photopeak_model(x, H, x0, sigma, sigma_e, t, h, B)

    # ------------------------ PLOT ------------------------
    plt.figure(figsize=(11,6))
    plt.step(x, hist, where='mid', label="Histogram", linewidth=1.5)

    plt.plot(x, H*gaussian, label="Gaussian", color="black")
    plt.plot(x, H*exp_tail, label="Exponential Tail", color="purple")
    plt.plot(x, H*h*erf_component, label="Error Function Term", color="orange")
    plt.plot(x, background, label="Background", color="green")
    plt.plot(x, F, label="Full Fit (Shape Function)", color="red", linewidth=2)
    plt.xlabel("ADC Value")
    plt.ylabel("Counts")
    plt.title(f"Fit for Channel {key}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()
    
def calculate_chi_squared(bin_centers, hist_counts, popt, reduced=True):
    """
    Calculate chi-squared goodness of fit.
    
    Parameters:
    -----------
    bin_centers : array
        ADC bin center values (x_i)
    hist_counts : array
        Histogram counts (observed data)
    popt : array
        Fitted parameters [H, x0, sigma, sigma_e, t, h, B]
    reduced : bool
        If True, return reduced chi-squared (chi²/dof)
        
    Returns:
    --------
    chi2 : float
        Chi-squared value (or reduced chi-squared if reduced=True)
    """
    # Get fitted values F(x_i)
    F_xi = photopeak_model(bin_centers, *popt)
    
    # Measurement error: σ_xi = sqrt(counts)
    # For zero or very low counts, use minimum of 1 to avoid division by zero
    sigma_xi = np.sqrt(hist_counts)
    sigma_xi = np.where(sigma_xi > 0, sigma_xi, 1.0)
    
    # Chi-squared: sum of [(x_i - F(x_i))^2 / σ_xi^2]
    chi2 = np.sum(((hist_counts - F_xi)**2) / (sigma_xi**2))
    
    if reduced:
        # Degrees of freedom = number of data points - number of parameters
        n_params = len(popt)
        dof = len(bin_centers) - n_params
        if dof > 0:
            chi2 = chi2 / dof
        else:
            print("Warning: degrees of freedom <= 0")
    
    return chi2


def calculate_chi_squared_only_fitted_region(bin_centers, hist_counts, popt, reduced=True):
    """
    Calculate chi-squared only for bins with significant counts (above threshold).
    This is more appropriate when you filtered out low-count bins before fitting.
    
    Parameters:
    -----------
    bin_centers : array
        ADC bin center values
    hist_counts : array
        Histogram counts
    popt : array
        Fitted parameters [H, x0, sigma, sigma_e, t, h, B]
    reduced : bool
        If True, return reduced chi-squared
        
    Returns:
    --------
    chi2 : float
        Chi-squared value
    """
    # Only include bins with counts > threshold (e.g., background + 3*sigma)
    background = popt[6]  # B parameter
    threshold = background + 3 * np.sqrt(background + 1)
    
    mask = hist_counts > threshold
    
    if np.sum(mask) < len(popt):
        print("Warning: fewer valid bins than parameters!")
        return np.inf
    
    bin_centers_valid = bin_centers[mask]
    hist_valid = hist_counts[mask]
    
    # Calculate chi-squared on valid bins only
    F_xi = photopeak_model(bin_centers_valid, *popt)
    sigma_xi = np.sqrt(hist_valid)
    sigma_xi = np.where(sigma_xi > 0, sigma_xi, 1.0)
    
    chi2 = np.sum(((hist_valid - F_xi)**2) / (sigma_xi**2))
    
    if reduced:
        dof = len(bin_centers_valid) - len(popt)
        if dof > 0:
            chi2 = chi2 / dof
    
    return chi2


def print_fit_quality(bin_centers, hist_counts, popt):
    """
    Print comprehensive fit quality metrics.
    
    Parameters:
    -----------
    bin_centers : array
        ADC bin centers
    hist_counts : array
        Histogram counts
    popt : array
        Fitted parameters [H, x0, sigma, sigma_e, t, h, B]
    """
    chi2 = calculate_chi_squared(bin_centers, hist_counts, popt, reduced=False)
    chi2_reduced = calculate_chi_squared(bin_centers, hist_counts, popt, reduced=True)
    
    n_params = len(popt)
    n_bins = len(bin_centers)
    dof = n_bins - n_params
    
    print("\n" + "="*60)
    print("FIT QUALITY ASSESSMENT")
    print("="*60)
    print(f"Number of bins:        {n_bins}")
    print(f"Number of parameters:  {n_params}")
    print(f"Degrees of freedom:    {dof}")
    print(f"\nχ² (total):            {chi2:.2f}")
    print(f"χ²/dof (reduced):      {chi2_reduced:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if chi2_reduced < 1.5:
        print("  ✓ Good fit (χ²/dof ≈ 1)")
    elif chi2_reduced < 3.0:
        print("  ⚠ Acceptable fit (χ²/dof slightly elevated)")
    else:
        print("  ✗ Poor fit (χ²/dof >> 1)")
        print("    Consider:")
        print("    - Different initial guesses")
        print("    - Checking for systematic issues")
        print("    - Model may not describe data well")
    
    # Residual analysis
    F_xi = photopeak_model(bin_centers, *popt)
    residuals = hist_counts - F_xi
    sigma_xi = np.sqrt(hist_counts)
    sigma_xi = np.where(sigma_xi > 0, sigma_xi, 1.0)
    normalized_residuals = residuals / sigma_xi
    
    print(f"\nResidual statistics:")
    print(f"  Mean residual:         {np.mean(residuals):.2f}")
    print(f"  Std of residuals:      {np.std(residuals):.2f}")
    print(f"  Max |residual|:        {np.max(np.abs(residuals)):.2f}")
    print(f"  Mean normalized res.:  {np.mean(normalized_residuals):.4f}")
    print(f"  Std normalized res.:   {np.std(normalized_residuals):.4f}")
    print("="*60)
    
    return chi2, chi2_reduced

def fit_one_channel(bin_centers, hist, auto_region=True, lower_bound=None, upper_bound=None):
    """
    Fit histogram with adjusted thresholds for low-statistics data.
    """
    if auto_region:
        # Find the peak
        peak_idx = np.argmax(hist)
        peak_pos = bin_centers[peak_idx]
        
        # Estimate width from FWHM
        half_max = hist[peak_idx] / 2
        above_half = hist > half_max
        if np.any(above_half):
            left_idx = np.where(above_half)[0][0]
            right_idx = np.where(above_half)[0][-1]
            estimated_width = bin_centers[right_idx] - bin_centers[left_idx]
        else:
            estimated_width = 200  # Wider default
        
        # Wider range: peak ± 10*width to capture tails
        lower_bound = max(bin_centers[0], peak_pos - 10 * estimated_width)
        upper_bound = min(bin_centers[-1], peak_pos + 5 * estimated_width)
        
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
    
    if len(bins_fit_all) < min_bins:
        print(f"ERROR: Only {len(bins_fit_all)} bins available. Need wider range!")
        return None, None, None, None
    
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
    
    sigma0 = max(sigma0, 20)  # Minimum sigma
    
    sigma_e0 = sigma0 * 2
    t0 = sigma0 * 1.5
    h0 = 0.3
    B0 = max(0.1, background_est)
    
    p0 = [H0, x0_0, sigma0, sigma_e0, t0, h0, B0]
    
    print(f"\nInitial guesses:")
    print(f"  H={H0:.1f}, x0={x0_0:.1f}, σ={sigma0:.1f}")
    print(f"  σe={sigma_e0:.1f}, t={t0:.1f}, h={h0:.2f}, B={B0:.2f}")
    
    # Wider bounds
    bounds = (
        [H0*0.01,   x0_0-200,  1,    10,   1,    0,     0],
        [H0*5,      x0_0+200,  300,  500,  150,  5,     max(10, B0*10)]
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
        
        # Check continuity
        check_continuity(popt)
        
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