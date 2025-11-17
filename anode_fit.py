import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

def photopeak_model_original(x, H, x0, sigma, sigma_e, t, h, B):
    """
    Full photopeak shape model with:
    - Gaussian peak
    - Error function tail
    - Exponential tail
    - Constant background
    
    Based on the piecewise definition from your PDF.
    """

    # Transition point between formulas
    boundary = x0 - t

    # Masks for left and right sides
    mask_right = x >= boundary   # Gaussian-dominated region
    mask_left  = ~mask_right     # exponential-tail region

    # Output array (same shape as x)
    F = np.zeros_like(x, dtype=float)

    # --------------------------
    # RIGHT SIDE (x >= x0 − t)
    # --------------------------
    if np.any(mask_right):
        xr = x[mask_right]

        gaussian = np.exp(-((xr - x0)**2) / (2 * sigma**2))
        erf_term = erf(-(xr - x0)**2 / (2.0 * sigma_e))

        F[mask_right] = H * (gaussian + h * erf_term) + B

    # --------------------------
    # LEFT SIDE (x < x0 − t)
    # --------------------------
    if np.any(mask_left):
        xl = x[mask_left]

        # Exponential tail term: exp[ t*(2x − x0 + t) / (2σ²) ]
        exp_tail = np.exp((t * (2*xl - x0 + t)) / (2 * sigma**2))

        erf_term = erf(-(xl - x0)**2 / (2.0 * sigma_e))

        F[mask_left] = H * (exp_tail + h * erf_term) + B

    return F

def fit_one_channel(bin_centers, hist, lower_bound, upper_bound):
    """
    Fit histogram data for one channel using the photopeak model.
    Returns popt (best-fit parameters) and pcov.
    """
    mask = (bin_centers > lower_bound) & (bin_centers < upper_bound)
    hist = hist[mask]
    bin_centers = bin_centers[mask]
    
    
    # Initial guesses (robust defaults)
    H0 = hist.max()
    x0_0 = bin_centers[np.argmax(hist)]
    
    # Rough estimate of sigma from the peak region
    peak_idx = np.argmax(hist)
    left = max(0, peak_idx - 200)
    right = min(len(bin_centers), peak_idx + 20)
    sigma0 = np.std(bin_centers[left:right])
    
    sigma_e0 = 30
    t0 = 15
    h0 = 0.3
    B0 = max(1.0, np.median(hist[:200]))   # background near ADC=0

    p0 = [H0, x0_0, sigma0, sigma_e0, t0, h0, B0]
    
    print("Initial guesses:")
    print("H0 =", H0)
    print("x0_0 =", x0_0)
    print("sigma0 =", sigma0)
    print("sigma_e0 =", sigma_e0)
    print("t0 =", t0)
    print("h0 =", h0)
    print("B0 =", B0)

    # Bounds to stabilize fitting
    bounds = (
        [0,   bin_centers.min(),  1,   1,   0,   0,    0],         # lower
        [np.inf, bin_centers.max(), 500, 500, 100, 100,   hist.max()]  # upper
    )

    try:
        popt, pcov = curve_fit(
            photopeak_model,
            bin_centers, hist,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return None, None

    return popt, pcov



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