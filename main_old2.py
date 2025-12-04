import numpy as np
import matplotlib.pyplot as plt
import argparse
from build_hists_opt import build_anode_histograms2, build_cathode_histograms
from anode_fit_test import extract_photopeak_info, fit_one_channel2, calibrate_anodes_in_parallel
from anode_fit import calculate_chi_squared, photopeak_model
from cathode_fit import fit_one_cathode
import time
import json
import os

# Create output directory for plots
os.makedirs("presentation_plots", exist_ok=True)

def diagnose_anode_histograms(anode_hists, bin_centers):
    """Diagnose histogram quality issues."""
    print("\n" + "="*70)
    print("ANODE HISTOGRAM QUALITY DIAGNOSTICS")
    print("="*70)
    
    total = len(anode_hists)
    empty = sum(1 for h in anode_hists.values() if np.sum(h) == 0)
    single_spike = sum(1 for h in anode_hists.values() if np.count_nonzero(h) == 1)
    low_stats = sum(1 for h in anode_hists.values() if np.sum(h) < 100)
    good = total - empty - single_spike - low_stats
    
    print(f"\nTotal channels: {total}")
    print(f"  ✓ Good histograms (>100 counts, >1 bin): {good} ({good/total*100:.1f}%)")
    print(f"  ⚠️  Low statistics (<100 counts): {low_stats} ({low_stats/total*100:.1f}%)")
    print(f"  ✗ Single-spike histograms: {single_spike} ({single_spike/total*100:.1f}%)")
    print(f"  ✗ Empty histograms: {empty} ({empty/total*100:.1f}%)")
    
    if single_spike > total * 0.1:
        print(f"\n⚠️  WARNING: >10% single-spike histograms detected!")
        print("    This indicates a histogram building bug.")
    
    # Show examples
    if single_spike > 0:
        print("\nExample single-spike channels:")
        count = 0
        for key, hist in anode_hists.items():
            if np.count_nonzero(hist) == 1:
                spike_bin = np.argmax(hist)
                spike_adc = bin_centers[spike_bin]
                spike_count = hist[spike_bin]
                print(f"  {key}: spike at ADC={spike_adc:.0f}, count={spike_count}")
                count += 1
                if count >= 5:
                    break
    
    return good, single_spike, low_stats, empty


def plot_calibration_validation(anode_hists, bin_centers, calibrations, kev, n_samples=6):
    """Plot histograms with keV x-axis to validate calibrations."""
    print("\n📊 Generating calibration validation plots (keV x-axis)...")
    
    # Find successfully calibrated channels
    calibrated_channels = [(k, h) for k, h in anode_hists.items() 
                           if k in calibrations and np.sum(h) > 100]
    
    if len(calibrated_channels) == 0:
        print("⚠️  No calibrated channels with sufficient statistics")
        return
    
    # Take samples
    samples = calibrated_channels[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (channel, hist) in enumerate(samples):
        ax = axes[idx]
        
        # Get calibration
        slope, intercept = calibrations[channel]
        
        # Convert ADC bins to keV
        kev_bins = slope * bin_centers + intercept
        
        # Plot histogram with keV x-axis
        ax.step(kev_bins, hist, where='mid', color='blue', 
                linewidth=1.5, label='Data', alpha=0.7)
        
        # Try to fit
        popt, pcov, bins_fit, hist_fit = fit_one_channel2(
            bin_centers, hist, auto_region=False, 
            lower_bound=1800, upper_bound=3200
        )
        
        if popt is not None:
            # Get peak ADC
            peak_info = extract_photopeak_info(popt, pcov)
            peak_adc = peak_info['photopeak_position']
            
            # Convert peak to keV
            peak_kev = slope * peak_adc + intercept
            
            # Plot fit in keV space
            kev_fit = slope * bins_fit + intercept
            model_fit = photopeak_model(bins_fit, *popt)
            ax.plot(kev_fit, model_fit, 'r-', linewidth=2, label='Fit')
            
            # Mark expected and measured peak
            ax.axvline(kev, color='green', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Expected: {kev:.0f} keV')
            ax.axvline(peak_kev, color='orange', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Measured: {peak_kev:.1f} keV')
            
            # Calculate error
            error = abs(peak_kev - kev)
            error_pct = error / kev * 100
            
            if error_pct < 5:
                status = '✓'
                color = 'green'
            elif error_pct < 10:
                status = '⚠️'
                color = 'orange'
            else:
                status = '✗'
                color = 'red'
            
            title = f'{status} Channel {channel}\nError: {error:.1f} keV ({error_pct:.1f}%)'
        else:
            title = f'✗ Channel {channel}\nFit Failed'
            color = 'red'
        
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel('Energy (keV)', fontsize=9)
        ax.set_ylabel('Counts', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Calibration Validation (Expected: {kev} keV)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('presentation_plots/calibration_validation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: presentation_plots/calibration_validation.png")
    plt.close()


def plot_fit_failure_analysis(anode_hists, bin_centers, n_samples=6):
    """Analyze why fits are failing."""
    print("\n📊 Analyzing fit failures...")
    
    # Find channels where fits might fail
    problematic = []
    
    for key, hist in anode_hists.items():
        total_counts = np.sum(hist)
        max_count = np.max(hist)
        nonzero_bins = np.count_nonzero(hist)
        
        # Categorize
        if nonzero_bins == 1:
            category = "single_spike"
        elif total_counts < 100:
            category = "low_stats"
        elif max_count < 20:
            category = "low_peak"
        elif nonzero_bins < 10:
            category = "too_narrow"
        else:
            continue
        
        problematic.append((key, hist, category))
    
    if len(problematic) == 0:
        print("  No problematic histograms found!")
        return
    
    print(f"Found {len(problematic)} problematic histograms")
    
    # Show examples of each category
    samples = problematic[6:6+n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (channel, hist, category) in enumerate(samples):
        ax = axes[idx]
        
        ax.step(bin_centers, hist, where='mid', color='red', 
                linewidth=1.5, alpha=0.7)
        
        total = np.sum(hist)
        max_val = np.max(hist)
        nonzero = np.count_nonzero(hist)
        
        title = f'Channel {channel}\n{category.replace("_", " ").title()}'
        subtitle = f'Total: {total:.0f}, Max: {max_val:.0f}, Non-zero bins: {nonzero}'
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=9, fontweight='bold', color='red')
        ax.set_xlabel('ADC Value', fontsize=9)
        ax.set_ylabel('Counts', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Fit Failure Analysis - Problematic Histograms', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('presentation_plots/fit_failures.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: presentation_plots/fit_failures.png")
    plt.close()


def plot_sample_anode_fits(anode_hists, bin_centers, calibrations, kev, n_samples=6):
    """Generate plots of successful anode fits for presentation."""
    print("\n📊 Generating anode fit plots for presentation...")
    
    # Find channels with good statistics
    good_channels = [(k, h) for k, h in anode_hists.items() 
                     if np.sum(h) > 500 and np.max(h) > 20 and np.count_nonzero(h) > 10]
    
    if len(good_channels) == 0:
        print("⚠️  No good anode channels found")
        return
    
    samples = good_channels[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (channel, hist) in enumerate(samples):
        ax = axes[idx]
        
        ax.step(bin_centers, hist, where='mid', color='blue', 
                linewidth=1.5, label='Data', alpha=0.7)
        
        popt, pcov, bins_fit, hist_fit = fit_one_channel2(
            bin_centers, hist, auto_region=False, 
            lower_bound=1800, upper_bound=3200
        )
        
        if popt is not None:
            model_fit = photopeak_model(bins_fit, *popt)
            ax.plot(bins_fit, model_fit, 'r-', linewidth=2, label='Fit')
            
            peak_info = extract_photopeak_info(popt, pcov)
            peak_adc = peak_info['photopeak_position']
            
            ax.axvline(peak_adc, color='green', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Peak={peak_adc:.0f} ADC')
            
            if channel in calibrations:
                slope, intercept = calibrations[channel]
                energy = slope * peak_adc + intercept
                title = f'Channel {channel}\nPeak: {peak_adc:.0f} ADC → {energy:.1f} keV'
            else:
                title = f'Channel {channel}\nPeak: {peak_adc:.0f} ADC'
        else:
            title = f'Channel {channel}\nFit Failed'
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('ADC Value', fontsize=9)
        ax.set_ylabel('Counts', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Anode Photopeak Fits ({kev} keV Source)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('presentation_plots/anode_fits.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: presentation_plots/anode_fits.png")
    plt.close()


def plot_sample_cathode_fits(cathode_hists, bin_centers, kev, n_samples=6):
    """Generate plots of cathode fits for presentation."""
    print("\n📊 Generating cathode fit plots for presentation...")
    
    good_channels = [(k, h) for k, h in cathode_hists.items() 
                     if np.sum(h) > 100]
    
    if len(good_channels) == 0:
        print("⚠️  No good cathode channels found")
        return
    
    print(f"Found {len(good_channels)} cathode channels with data")
    
    samples = good_channels[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (channel, hist) in enumerate(samples):
        ax = axes[idx]
        
        ax.step(bin_centers, hist, where='mid', color='blue', 
                linewidth=1.5, label='Data', alpha=0.7)
        
        result = fit_one_cathode(hist, bin_centers)
        
        if result is not None:
            slope, intercept, photopeak_adc, (range_lo, range_hi) = result
            
            x_fit = np.linspace(range_lo, range_hi, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Linear Fit')
            
            ax.axvline(photopeak_adc, color='green', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Peak={photopeak_adc:.0f}')
            
            ax.axvline(range_lo, color='orange', linestyle=':', alpha=0.5)
            ax.axvline(range_hi, color='orange', linestyle=':', alpha=0.5)
            
            title = f'✓ Channel {channel}\nCompton Edge: {photopeak_adc:.0f} ADC'
        else:
            title = f'✗ Channel {channel}\nFit Failed'
        
        total = np.sum(hist)
        ax.set_title(f'{title}\nTotal: {total:.0f} counts', 
                    fontsize=9, fontweight='bold')
        ax.set_xlabel('ADC Value', fontsize=9)
        ax.set_ylabel('Counts', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Cathode Compton Edge Fits ({kev} keV Source)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('presentation_plots/cathode_fits.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: presentation_plots/cathode_fits.png")
    plt.close()


def main():
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="ADC to keV Calibration Pipeline")
    parser.add_argument("--file1", required=True, help=".txt file for isotope 1")
    parser.add_argument("--kev1", default=662, type=float, help="Known keV for file1")
    parser.add_argument("--file2", required=True, help=".txt file for isotope 2")
    parser.add_argument("--kev2", default=511, type=float, help="Known keV for file2")
    parser.add_argument("--out", default="calibration.json", help="Output calibration file")
    args = parser.parse_args()
    
    print("="*70)
    print("ADC TO KEV ENERGY CALIBRATION PIPELINE")
    print("="*70)
    
    calibrations = {}
    
    # ===== STEP 1: Build Anode Histograms =====
    print(f"\n{'='*70}")
    print("STEP 1: BUILDING ANODE HISTOGRAMS")
    print(f"{'='*70}")
    
    t0 = time.perf_counter()
    anode_hists1, anode_bin_centers1, cathode_anode_pairs1 = build_anode_histograms2(
        args.file1, verbose=True
    )
    anode_hists2, anode_bin_centers2, cathode_anode_pairs2 = build_anode_histograms2(
        args.file2, verbose=True
    )
    t1 = time.perf_counter()
    
    print(f"\n✓ Anode histograms built in {t1-t0:.2f}s")
    
    # Diagnose histogram quality
    print("\nFile 1 diagnostics:")
    
    # ====== ADD BACK AFTER TESTING ====
    ##  good1, spike1, low1, empty1 = diagnose_anode_histograms(anode_hists1, anode_bin_centers1)
    print("\nFile 2 diagnostics:")
    ##  good2, spike2, low2, empty2 = diagnose_anode_histograms(anode_hists2, anode_bin_centers2)
    
    # ===================================
    
    # ===== STEP 2: Fit Anode Channels =====
    print(f"\n{'='*70}")
    print("STEP 2: FITTING ANODE PHOTOPEAKS")
    print(f"{'='*70}")
    
    common_anode_channels = set(anode_hists1.keys()).intersection(set(anode_hists2.keys()))
    print(f"\nCommon anode channels: {len(common_anode_channels)}")
    
    t0 = time.perf_counter()
    failed_fits = 0
    
    for channel in common_anode_channels:
        hist1 = anode_hists1[channel]
        hist2 = anode_hists2[channel]
        
        # Skip bad histograms
        if np.count_nonzero(hist1) <= 1 or np.count_nonzero(hist2) <= 1:
            failed_fits += 1
            continue
        
        if np.sum(hist1) < 100 or np.sum(hist2) < 100:
            failed_fits += 1
            continue
        
        # Fit with wider bounds
        popt1, pcov1, bins_fit1, hist_fit1 = fit_one_channel2(
            anode_bin_centers1, hist1, auto_region=False, 
            lower_bound=1800, upper_bound=3200
        )
        
        popt2, pcov2, bins_fit2, hist_fit2 = fit_one_channel2(
            anode_bin_centers2, hist2, auto_region=False,
            lower_bound=1500, upper_bound=2800
        )
        
        if popt1 is None or popt2 is None:
            failed_fits += 1
            continue
        
        peak_info1 = extract_photopeak_info(popt1, pcov1)
        peak_info2 = extract_photopeak_info(popt2, pcov2)
        adc1 = peak_info1['photopeak_position']
        adc2 = peak_info2['photopeak_position']
        
        if adc1 <= 0 or adc2 <= 0 or abs(adc1 - adc2) < 50:
            failed_fits += 1
            continue
        
        slope = (args.kev1 - args.kev2) / (adc1 - adc2)
        intercept = args.kev2 - slope * adc2
        
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0 or slope > 2:
            failed_fits += 1
            continue
        
        calibrations[channel] = (slope, intercept)
    
    t1 = time.perf_counter()
    anode_hist_time = t1-t0
    success_rate = 100 * (1 - failed_fits / len(common_anode_channels))
    print(f"\n✓ Anode fitting complete in {t1-t0:.2f}s")
    print(f"  Success: {len(calibrations)}/{len(common_anode_channels)} ({success_rate:.1f}%)")
    
    # ===== STEP 3: Build Cathode Histograms =====
    print(f"\n{'='*70}")
    print("STEP 3: BUILDING CATHODE HISTOGRAMS")
    print(f"{'='*70}")
    
    t0 = time.perf_counter()
    cathode_hists1, cathode_bin_centers1 = build_cathode_histograms(
        cathode_anode_pairs1, calibrations, args.kev1
    )
    cathode_hists2, cathode_bin_centers2 = build_cathode_histograms(
        cathode_anode_pairs2, calibrations, args.kev2
    )
    t1 = time.perf_counter()
    print(f"✓ Cathode histograms built in {t1-t0:.2f}s")
    
    # ===== STEP 4: Fit Cathode Channels =====
    print(f"\n{'='*70}")
    print("STEP 4: FITTING CATHODE COMPTON EDGES")
    print(f"{'='*70}")
    
    common_cathode_channels = set(cathode_hists1.keys()).intersection(set(cathode_hists2.keys()))
    failed_cathode = 0
    
    t0 = time.perf_counter()
    for channel in common_cathode_channels:
        if np.sum(cathode_hists1[channel]) < 10 or np.sum(cathode_hists2[channel]) < 10:
            failed_cathode += 1
            continue
        
        r1 = fit_one_cathode(cathode_hists1[channel], cathode_bin_centers1)
        r2 = fit_one_cathode(cathode_hists2[channel], cathode_bin_centers2)
        
        if r1 is None or r2 is None:
            failed_cathode += 1
            continue
        
        slope = (args.kev1 - args.kev2) / (r1[2] - r2[2])
        intercept = args.kev2 - slope * r2[2]
        
        if np.isfinite(slope) and np.isfinite(intercept):
            calibrations[channel] = (slope, intercept)
        else:
            failed_cathode += 1
    
    t1 = time.perf_counter()
    cathode_success = 100 * (1 - failed_cathode / max(1, len(common_cathode_channels)))
    print(f"✓ Cathode fitting complete in {t1-t0:.2f}s")
    print(f"  Success: {cathode_success:.1f}%")
    
    # ===== SAVE & VISUALIZE =====
    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in calibrations.items()}, f, indent=2)
    print(f"\n✓ Saved: {args.out}")
    
    print(f"\n{'='*70}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print(f"{'='*70}")
    
    plot_fit_failure_analysis(anode_hists1, anode_bin_centers1)
    plot_sample_anode_fits(anode_hists1, anode_bin_centers1, calibrations, args.kev1)
    plot_calibration_validation(anode_hists1, anode_bin_centers1, calibrations, args.kev1)
    plot_sample_cathode_fits(cathode_hists1, cathode_bin_centers1, args.kev1)
    
    t_end = time.perf_counter()
    print()
    print(f"total run time: {t_end-t_start:0.2f} seconds")
    print(f"time to build anode hists: {anode_hist_time} seconds")
    print()
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    

if __name__ == "__main__":
    main()