import numpy as np
import matplotlib.pyplot as plt
from cathode_fit import fit_one_cathode

def diagnose_cathode_histograms(cathode_hists, bin_centers, sample_size=10):
    """
    Diagnose why cathode fits are failing.
    Shows statistics and plots sample histograms.
    """
    print("\n" + "="*70)
    print("CATHODE HISTOGRAM DIAGNOSTICS")
    print("="*70)
    
    # Collect statistics
    total_counts = []
    max_counts = []
    nonzero_bins = []
    empty_hists = 0
    
    for hist in cathode_hists.values():
        tc = np.sum(hist)
        total_counts.append(tc)
        max_counts.append(np.max(hist))
        nonzero_bins.append(np.count_nonzero(hist))
        if tc < 10:
            empty_hists += 1
    
    print(f"\nTotal channels: {len(cathode_hists)}")
    print(f"Empty histograms (< 10 counts): {empty_hists} ({empty_hists/len(cathode_hists)*100:.1f}%)")
    
    print(f"\nTotal counts per histogram:")
    print(f"  Min: {np.min(total_counts):.0f}")
    print(f"  Median: {np.median(total_counts):.0f}")
    print(f"  Mean: {np.mean(total_counts):.0f}")
    print(f"  Max: {np.max(total_counts):.0f}")
    
    print(f"\nMax bin counts per histogram:")
    print(f"  Min: {np.min(max_counts):.0f}")
    print(f"  Median: {np.median(max_counts):.0f}")
    print(f"  Mean: {np.mean(max_counts):.0f}")
    print(f"  Max: {np.max(max_counts):.0f}")
    
    print(f"\nNon-zero bins per histogram:")
    print(f"  Min: {np.min(nonzero_bins)}")
    print(f"  Median: {np.median(nonzero_bins):.0f}")
    print(f"  Mean: {np.mean(nonzero_bins):.0f}")
    print(f"  Max: {np.max(nonzero_bins)}")
    
    # Plot sample histograms
    print(f"\n" + "="*70)
    print(f"Plotting {sample_size} sample histograms...")
    print("="*70)
    
    # Get diverse samples: some with high counts, some with low
    sorted_channels = sorted(cathode_hists.items(), 
                            key=lambda x: np.sum(x[1]), 
                            reverse=True)
    
    # Top 5 (most counts) and bottom 5 (least counts, but > 0)
    non_empty = [(k, h) for k, h in sorted_channels if np.sum(h) > 0]
    
    if len(non_empty) == 0:
        print("⚠️  ALL CATHODE HISTOGRAMS ARE EMPTY!")
        return
    
    samples = []
    # Top 5
    samples.extend(non_empty[:min(5, len(non_empty))])
    # Bottom 5
    samples.extend(non_empty[-min(5, len(non_empty)):])
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (channel, hist) in enumerate(samples[:sample_size]):
        ax = axes[idx]
        
        # Plot histogram
        ax.step(bin_centers, hist, where='mid', color='blue', linewidth=1.5)
        
        # Try to fit
        result = fit_one_cathode(hist, bin_centers)
        
        total = np.sum(hist)
        max_val = np.max(hist)
        
        if result is not None:
            slope, intercept, photopeak_adc, (range_lo, range_hi) = result
            
            # Plot fit
            x_fit = np.linspace(range_lo, range_hi, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fit')
            
            # Mark photopeak
            ax.axvline(photopeak_adc, color='green', linestyle='--', 
                      label=f'Peak={photopeak_adc:.0f}')
            
            # Mark fitting region
            ax.axvline(range_lo, color='orange', linestyle=':', alpha=0.5)
            ax.axvline(range_hi, color='orange', linestyle=':', alpha=0.5)
            
            title = f'{channel}\n✓ FIT: peak={photopeak_adc:.0f}'
        else:
            title = f'{channel}\n✗ FIT FAILED'
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('ADC', fontsize=8)
        ax.set_ylabel('Counts', fontsize=8)
        ax.text(0.02, 0.98, f'Total: {total:.0f}\nMax: {max_val:.0f}', 
               transform=ax.transAxes, va='top', fontsize=7,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('cathode_diagnostics.png', dpi=150)
    print("\n✓ Saved diagnostic plot: cathode_diagnostics.png")
    plt.show()


def test_cathode_fit_logic(hist, bin_centers, show_plot=True):
    """
    Test the cathode fitting logic step by step to understand failures.
    """
    print("\n" + "="*70)
    print("STEP-BY-STEP CATHODE FIT ANALYSIS")
    print("="*70)
    
    # Step 1: Find maximum
    if len(hist) == 0 or np.max(hist) == 0:
        print("✗ FAIL: Empty histogram or all zeros")
        return None
    
    max_bin_index = np.argmax(hist)
    max_height = hist[max_bin_index]
    max_adc = bin_centers[max_bin_index]
    
    print(f"\n1. Maximum bin:")
    print(f"   Index: {max_bin_index}, ADC: {max_adc:.1f}, Height: {max_height:.0f}")
    
    # Step 2: Calculate thresholds (using 10%-90%)
    lo_height = max_height * 0.10
    hi_height = max_height * 0.90
    
    print(f"\n2. Height thresholds (10%-90%):")
    print(f"   Lo (10%): {lo_height:.2f}")
    print(f"   Hi (90%): {hi_height:.2f}")
    
    # Step 3: Find indices
    lo_height_index = 0
    hi_height_index = 0
    
    for i in range(len(hist)):
        if hist[i] <= lo_height:
            lo_height_index = i
        if hist[i] > hi_height:
            hi_height_index = i
            break
    
    range_lo = bin_centers[lo_height_index]
    range_hi = bin_centers[hi_height_index]
    
    print(f"\n3. Fitting range indices:")
    print(f"   Lo index: {lo_height_index} (ADC: {range_lo:.1f})")
    print(f"   Hi index: {hi_height_index} (ADC: {range_hi:.1f})")
    print(f"   Range width: {range_hi - range_lo:.1f} ADC units")
    
    # Step 4: Extract fitting data
    mask = (bin_centers >= range_lo) & (bin_centers <= range_hi)
    x_fit = bin_centers[mask]
    y_fit = hist[mask]
    
    print(f"\n4. Fitting data:")
    print(f"   Number of points: {len(x_fit)}")
    
    if len(x_fit) < 3:
        print("   ✗ FAIL: Need at least 3 points to fit")
        return None
    
    # Step 5: Linear fit
    try:
        coeffs = np.polyfit(x_fit, y_fit, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        print(f"\n5. Linear fit results:")
        print(f"   Slope: {slope:.4f}")
        print(f"   Intercept: {intercept:.2f}")
        
        # Step 6: Calculate photopeak
        if slope == 0:
            print("   ✗ FAIL: Slope is zero, cannot calculate photopeak")
            return None
        
        photopeak_adc = -intercept / slope
        
        print(f"\n6. Photopeak position:")
        print(f"   ADC: {photopeak_adc:.1f}")
        
        # Sanity check
        if photopeak_adc < 0 or photopeak_adc > 4095:
            print(f"   ✗ FAIL: Photopeak outside valid range [0, 4095]")
            return None
        
        print(f"   ✓ SUCCESS: Valid photopeak position")
        
        # Plot if requested
        if show_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot full histogram
            ax.step(bin_centers, hist, where='mid', color='blue', 
                   linewidth=1.5, label='Cathode Histogram')
            
            # Plot fit
            y_fit_line = slope * x_fit + intercept
            ax.plot(x_fit, y_fit_line, 'r-', linewidth=2, label='Linear Fit')
            
            # Mark key points
            ax.axvline(max_adc, color='purple', linestyle='--', 
                      alpha=0.7, label=f'Max at {max_adc:.0f}')
            ax.axvline(photopeak_adc, color='green', linestyle='--', 
                      linewidth=2, label=f'Photopeak at {photopeak_adc:.0f}')
            ax.axvline(range_lo, color='orange', linestyle=':', 
                      alpha=0.5, label=f'Fit range [{range_lo:.0f}, {range_hi:.0f}]')
            ax.axvline(range_hi, color='orange', linestyle=':', alpha=0.5)
            
            # Mark thresholds
            ax.axhline(lo_height, color='cyan', linestyle='--', 
                      alpha=0.3, label=f'10% threshold')
            ax.axhline(hi_height, color='magenta', linestyle='--', 
                      alpha=0.3, label=f'90% threshold')
            
            ax.set_xlabel('ADC', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title('Cathode Fit Analysis', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('cathode_fit_analysis.png', dpi=150)
            print(f"\n✓ Saved plot: cathode_fit_analysis.png")
            plt.show()
        
        return slope, intercept, photopeak_adc, (range_lo, range_hi)
        
    except Exception as e:
        print(f"\n✗ FAIL: Fitting error: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Cathode Fitting Diagnostic Tool")
    print("\nTo use this tool, import your histograms and run:")
    print("  from cathode_diagnostic import diagnose_cathode_histograms")
    print("  diagnose_cathode_histograms(cathode_hists1, bin_centers)")
    print("\nOr test a single histogram:")
    print("  from cathode_diagnostic import test_cathode_fit_logic")
    print("  test_cathode_fit_logic(hist, bin_centers)")