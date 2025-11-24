from build_hists import build_anode_histograms
import matplotlib.pyplot as plt
from anode_fit import plot_fit_components
from anode_fit import fit_one_channel
from anode_fit import photopeak_model
from anode_fit import calculate_chi_squared_only_fitted_region
from anode_fit import calculate_chi_squared
from anode_fit import print_fit_quality
from anode_fit_test import fit_one_channel2
from anode_fit_test import plot_fit_with_residuals
from anode_fit_test import print_photopeak_results
from anode_fit_test import extract_photopeak_info
from anode_fit_test import check_continuity
import numpy as np

def diagnose_histogram(bin_centers, hist, lower_bound=None, upper_bound=None):
    """
    Visualize histogram to understand issues.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.step(bin_centers, hist, where='mid', linewidth=1.5)
    plt.xlabel('ADC Value')
    plt.ylabel('Counts')
    plt.title('Full Histogram')
    plt.ylim(bottom=0.5)
    plt.grid(alpha=0.3)
    
    if lower_bound and upper_bound:
        plt.axvline(lower_bound, color='red', linestyle='--', label='Fit bounds')
        plt.axvline(upper_bound, color='red', linestyle='--')
        plt.legend()
    
    plt.subplot(1, 2, 2)
    if lower_bound and upper_bound:
        mask = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
        plt.step(bin_centers[mask], hist[mask], where='mid', linewidth=1.5)
        plt.xlabel('ADC Value')
        plt.ylabel('Counts')
        plt.title(f'Zoomed: [{lower_bound}, {upper_bound}]')
    else:
        plt.step(bin_centers, hist, where='mid', linewidth=1.5)
        plt.xlabel('ADC Value')
        plt.ylabel('Counts')
        plt.title('Data for Fitting')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"\nHistogram Statistics:")
    print(f"  Total counts: {hist.sum():.0f}")
    print(f"  Max count: {hist.max():.0f} at ADC={bin_centers[np.argmax(hist)]:.0f}")
    print(f"  Non-zero bins: {np.sum(hist > 0)}")
    if lower_bound and upper_bound:
        mask = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)
        print(f"  Counts in range [{lower_bound},{upper_bound}]: {hist[mask].sum():.0f}")
        print(f"  Bins in range: {np.sum(mask)}")
        print(f"  Non-zero bins in range: {np.sum(hist[mask] > 0)}")

# ============================================================
# MAIN TESTING SCRIPT
# ============================================================

# Path to test data file
path_to_file1 = "Cs-137new.txt"
path_to_file2 = "Ge-68.txt"

# Build histograms for both files
print("Building histograms...")
histograms1, bin_centers1 = build_anode_histograms(path_to_file1, verbose=True)
histograms2, bin_centers2 = build_anode_histograms(path_to_file2, verbose=True)

# Pick a channel to inspect
print("\n" + "="*60)
print("CHANNEL SELECTION")
print("="*60)
key = next(iter(set(histograms1.keys()) & set(histograms2.keys())))
hist1 = histograms1[key]
hist2 = histograms2[key]

print(f"Selected channel: {key}")
print(f"  node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
print(f"  Total counts: {hist1.sum():.0f}")
print(f"  Max count: {hist1.max():.0f} at ADC={bin_centers1[np.argmax(hist1)]:.0f}")
print(f"  Non-zero bins: {np.sum(hist1 > 0)}")

# Plot full histogram first
print("\nPlotting full histogram...")
plt.figure(figsize=(10, 6))
plt.step(bin_centers1, hist1, where='mid', linewidth=1.5)
plt.xlabel("ADC value", fontsize=12)
plt.ylabel("Counts", fontsize=12)
plt.title(f"Full Histogram: node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
plt.ylim(bottom=0.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Run diagnostic on default fit region BEFORE fitting
print("\n" + "="*60)
print("DIAGNOSTIC: Checking fit region")
print("="*60)
diagnose_histogram(bin_centers1, hist1, lower_bound=2400, upper_bound=2800)

# Ask user if they want to proceed or adjust
print(f"\nProposed fit range: [2400, 2800]")
proceed = input("Proceed with this range? (y/n) or enter new range as 'min,max': ").strip().lower()

if proceed == 'y':
    lower_bound, upper_bound = 2400, 2800
elif ',' in proceed:
    try:
        lower_bound, upper_bound = map(float, proceed.split(','))
        print(f"Using custom range: [{lower_bound}, {upper_bound}]")
    except:
        print("Invalid input, using defaults")
        lower_bound, upper_bound = 2400, 2800
else:
    print("Trying auto-region detection...")
    lower_bound, upper_bound = None, None

# Perform fit
print("\n" + "="*60)
print("FITTING")
print("="*60)

if lower_bound is None:
    # Auto-detect
    result = fit_one_channel2(bin_centers1, hist1, auto_region=True)
else:
    # Manual bounds
    result = fit_one_channel2(bin_centers1, hist1, auto_region=False, 
                             lower_bound=lower_bound, upper_bound=upper_bound)

# Analyze results
if result[0] is not None:
    popt, pcov, bins_fit, hist_fit = result
    
    print("\n" + "="*60)
    print("FIT RESULTS")
    print("="*60)
    
    # Show fitted parameters
    param_names = ['H', 'x0', 'sigma', 'sigma_e', 't', 'h', 'B']
    print("\nFitted parameters:")
    for name, val in zip(param_names, popt):
        print(f"  {name:10s} = {val:12.4f}")
    
    # Check continuity
    print("\n" + "-"*60)
    is_continuous = check_continuity(popt)
    
    # Calculate chi-squared
    print("\n" + "-"*60)
    chi2_reduced = calculate_chi_squared(bins_fit, hist_fit, popt, reduced=True)
    print(f"χ²/dof = {chi2_reduced:.4f}")
    
    if chi2_reduced < 2:
        print("Good fit!")
    elif chi2_reduced < 5:
        print("Acceptable fit")
    else:
        print("Poor fit - consider adjusting range or initial guesses")
    
    # Extract photopeak
    print("\n" + "-"*60)
    peak_info = extract_photopeak_info(popt, pcov)
    adc2 = peak_info['photopeak_position']
    print_photopeak_results(peak_info, source_energy_kev=662)
    
    # Plot final result with residuals
    print("\nGenerating final plot...")
    plot_fit_with_residuals(bins_fit, hist_fit, popt, key)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
else:
    print("\n" + "="*60)
    print("FIT FAILED")
    print("="*60)
    print("Suggestions:")
    print("  1. Try a wider fit range (e.g., 1800-3000)")
    print("  2. Use auto_region=True")
    print("  3. Check if histogram has enough statistics")
    print("  4. Try a different channel with more counts")

hist2 = histograms2[key]

print(f"Selected channel: {key}")
print(f"  node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
print(f"  Total counts: {hist2.sum():.0f}")
print(f"  Max count: {hist2.max():.0f} at ADC={bin_centers2[np.argmax(hist2)]:.0f}")
print(f"  Non-zero bins: {np.sum(hist2 > 0)}")

# Plot full histogram first
print("\nPlotting full histogram...")
plt.figure(figsize=(10, 6))
plt.step(bin_centers2, hist2, where='mid', linewidth=1.5)
plt.xlabel("ADC value", fontsize=12)
plt.ylabel("Counts", fontsize=12)
plt.title(f"Full Histogram: node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
plt.ylim(bottom=0.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Run diagnostic on default fit region BEFORE fitting
print("\n" + "="*60)
print("DIAGNOSTIC: Checking fit region")
print("="*60)
diagnose_histogram(bin_centers2, hist2, lower_bound=2000, upper_bound=2600)

# Ask user if they want to proceed or adjust
print(f"\nProposed fit range: [2100, 2500]")
proceed = input("Proceed with this range? (y/n) or enter new range as 'min,max': ").strip().lower()

if proceed == 'y':
    lower_bound, upper_bound = 2100, 2500
elif ',' in proceed:
    try:
        lower_bound, upper_bound = map(float, proceed.split(','))
        print(f"Using custom range: [{lower_bound}, {upper_bound}]")
    except:
        print("Invalid input, using defaults")
        lower_bound, upper_bound = 2100, 2500
else:
    print("Trying auto-region detection...")
    lower_bound, upper_bound = None, None

# Perform fit
print("\n" + "="*60)
print("FITTING")
print("="*60)

if lower_bound is None:
    # Auto-detect
    result = fit_one_channel2(bin_centers2, hist2, auto_region=True)
else:
    # Manual bounds
    result = fit_one_channel2(bin_centers2, hist2, auto_region=False, 
                             lower_bound=lower_bound, upper_bound=upper_bound)

# Analyze results
if result[0] is not None:
    popt, pcov, bins_fit, hist_fit = result
    
    print("\n" + "="*60)
    print("FIT RESULTS")
    print("="*60)
    
    # Show fitted parameters
    param_names = ['H', 'x0', 'sigma', 'sigma_e', 't', 'h', 'B']
    print("\nFitted parameters:")
    for name, val in zip(param_names, popt):
        print(f"  {name:10s} = {val:12.4f}")
    
    # Check continuity
    print("\n" + "-"*60)
    is_continuous = check_continuity(popt)
    
    # Calculate chi-squared
    print("\n" + "-"*60)
    chi2_reduced = calculate_chi_squared(bins_fit, hist_fit, popt, reduced=True)
    print(f"χ²/dof = {chi2_reduced:.4f}")
    
    if chi2_reduced < 2:
        print("Good fit!")
    elif chi2_reduced < 5:
        print("Acceptable fit")
    else:
        print("Poor fit - consider adjusting range or initial guesses")
    
    # Extract photopeak
    print("\n" + "-"*60)
    peak_info = extract_photopeak_info(popt, pcov)
    adc1 = peak_info['photopeak_position']
    print_photopeak_results(peak_info, source_energy_kev=662)
    
    # Plot final result with residuals
    print("\nGenerating final plot...")
    plot_fit_with_residuals(bins_fit, hist_fit, popt, key)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
else:
    print("\n" + "="*60)
    print("FIT FAILED")
    print("="*60)
    print("Suggestions:")
    print("  1. Try a wider fit range (e.g., 1800-3000)")
    print("  2. Use auto_region=True")
    print("  3. Check if histogram has enough statistics")
    print("  4. Try a different channel with more counts")

slope = (662 - 511) / (adc2 - adc1) 
#adc1 = Ge68 adc
#adc2 = Cs137 adc

intercept = 511 - slope * adc1
print(f"Slope = {slope}")
print(f"Intercept = {intercept}")

print(F"Kev = {slope}(adc) + {intercept}")

print("\n" + "="*60)
print("Printing Cs-137 histogram with kev x-axis")
print("="*60)

kev_bin_centers1 = slope * bin_centers1 + intercept
kev_bin_centers2 = slope * bin_centers2 + intercept

plt.figure(figsize=(10, 6))
plt.step(kev_bin_centers1, hist1, where='mid', linewidth=1.5)
plt.xlabel("KeV value", fontsize=12)
plt.ylabel("Counts", fontsize=12)
plt.title(f"Full Histogram: node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
plt.ylim(bottom=0.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()