import numpy as np
import argparse
from build_hists_opt import build_anode_histograms, build_cathode_histograms
from anode_fit_test import extract_photopeak_info
from anode_fit_test import fit_one_channel2
from anode_fit import calculate_chi_squared
from cathode_fit import fit_one_cathode
import time
import json


def main():
    # we expect file1 = CS137, kev1 = 662, file2 = Ge68, kev2 = 511
    parser = argparse.ArgumentParser(description="Calibrate ADC->keV from two .txt files (streaming).")
    parser.add_argument("--file1", required=True, help=".txt file for isotope 1 (whitespace-delimited)")
    parser.add_argument("--kev1", default=662, type=float, help="Known keV for file1")
    parser.add_argument("--file2", required=True, help=".txt file for isotope 2 (whitespace-delimited)")
    parser.add_argument("--kev2", default=511, type=float, help="Known keV for file2")
    parser.add_argument("--out", default="calibration.json", help="Output calibration file")
    parser.add_argument("--verbose", action='store_true', default=True, help="Print progress messages")
    args = parser.parse_args()
    
    # creating dictionary that stores calibration parameters in the form:
    #   (node,board,rena,channel) : (slope, intercept)
    calibrations = {}

    # ----- step1: build anode histograms -----
    t0 = time.perf_counter()
    
    # anode_hists1 and anode_hists2 are default dictionaries with the format:
    # key = (node,board,rena,channel) : value = numpy array (per-channel histogram)
    
    if args.verbose:
        print(f"Building anode histograms for file 1: {args.file1}")
    anode_hists1, anode_bin_centers1, cathode_anode_pairs1 = build_anode_histograms(
        args.file1, 
        verbose=args.verbose
    )

    if args.verbose:
        print(f"\nBuilding anode histograms for file 2: {args.file2}")
    anode_hists2, anode_bin_centers2, cathode_anode_pairs2 = build_anode_histograms(
        args.file2, 
        verbose=args.verbose
    )
    
    t1 = time.perf_counter()
    
    anode_histograms_build_time = t1 - t0
    
    # ----- Step 2: calibrate anode histograms ----
    # setting default upper and lower bounds for anode histogram fitting
    #To-Do: find good default bound values | create auto find for bounds
    lower_bound1 = 2400
    upper_bound1 = 2800
    
    lower_bound2 = 2100
    upper_bound2 = 2500

    # variable to keep track of failed fits
    failed_fits = 0
    
    # We'll only use channels that exist in both histograms
    common_anode_channels = set(anode_hists1.keys()).intersection(set(anode_hists2.keys()))
    
    if args.verbose:
        # print length of symmetric difference between set of channels == number of channels that don't exist in both .txt files
        print(f"\nCommon anode channels: {len(common_anode_channels)}")
        print(f"Dropped: {len(set(anode_hists1.keys()) ^ set(anode_hists2.keys()))} uncommon channels") # ^ is the symmetric difference operator
    
    # loop over all anode channel histograms, fit each histogram, compute ADC photopeak, and compute calibration parameters.
    if args.verbose:
        print("\nFitting anode channels...")
    
    t0 = time.perf_counter()
    
    for channel in common_anode_channels:
        fit_fail = False
        hist1 = anode_hists1[channel]
        hist2 = anode_hists2[channel]

        # ===== TRY AUTO-REGION FIRST =====
        popt1, pcov1, bins_fit1, hist_fit1 = fit_one_channel2(anode_bin_centers1, hist1, auto_region=True)
    
        # If auto failed, try manual bounds as fallback
        if popt1 is None:
            if args.verbose:
                print(f"  → Channel {channel}: Retrying with manual bounds [{lower_bound1}, {upper_bound1}]")
            popt1, pcov1, bins_fit1, hist_fit1 = fit_one_channel2(
                anode_bin_centers1, hist1, 
                auto_region=False,
                lower_bound=lower_bound1, 
                upper_bound=upper_bound1
            )
    
        # ===== SAME FOR SECOND ISOTOPE =====
        
        # try auto-bounds first
        popt2, pcov2, bins_fit2, hist_fit2 = fit_one_channel2(anode_bin_centers2, hist2, auto_region=True)
    
        # If auto failed, try manual bounds as fallback
        if popt2 is None:
            if args.verbose:
                print(f"  → Channel {channel}: Retrying with manual bounds [{lower_bound2}, {upper_bound2}]")
            popt2, pcov2, bins_fit2, hist_fit2 = fit_one_channel2(
                anode_bin_centers2, hist2,
                auto_region=False,
                lower_bound=lower_bound2,
                upper_bound=upper_bound2
            )
    
        # ===== CHECK IF BOTH FITS COMPLETELY FAILED =====
        if popt1 is None or popt2 is None:
            if args.verbose:
                print(f"Channel {channel}: Both auto and manual fits failed, using defaults")
            fit_fail = True
            failed_fits += 1
        
        if not fit_fail:
            # extracting ADC values for current channel from both isotopes
            peak_info1 = extract_photopeak_info(popt1, pcov1)
            adc1 = peak_info1['photopeak_position']
            
            peak_info2 = extract_photopeak_info(popt2, pcov2)
            adc2 = peak_info2['photopeak_position']
            
            # check GOF(Goodness of Fit) and determine if fit failed or not
            chi2_reduced1 = calculate_chi_squared(bins_fit1, hist_fit1, popt1, reduced=True)
            chi2_reduced2 = calculate_chi_squared(bins_fit2, hist_fit2, popt2, reduced=True)
            
            if (chi2_reduced1 > 5 or chi2_reduced2 > 5):
                fit_fail = True
                failed_fits += 1
        
        #  use ADC values to get calibration values for current channel
        if not fit_fail:
            slope = (args.kev1 - args.kev2) / (adc1 - adc2) 
            #adc1 = Cs137 adc
            #adc2 = Ge68 adc

            intercept = args.kev2 - slope * adc2
        else:
            #if fit fails, use default calibration values
            #To-Do: find good default calibration values
            slope = 0.4
            intercept = -415
        
        # store current anode channel's calibration parameters
        calibrations[channel] = (slope, intercept)
        
    t1 = time.perf_counter()
    anode_fit_time = t1 - t0
    
    if args.verbose:
        print(f"\nAnode fitting complete:")
        print(f"  - Failed fits: {failed_fits}/{len(common_anode_channels)}")
        print(f"  - Success rate: {100*(1-failed_fits/len(common_anode_channels)):.1f}%")
        
    
    # ----- Step 3: create cathode histograms ----
    if args.verbose:
        print("\nBuilding cathode histograms...")
    
    t0 = time.perf_counter()
    
    cathode_hists1, cathode_bin_centers1 = build_cathode_histograms(
        cathode_anode_pairs1, 
        calibrations, 
        args.kev1,
        verbose=args.verbose
    )
    cathode_hists2, cathode_bin_centers2 = build_cathode_histograms(
        cathode_anode_pairs2, 
        calibrations, 
        args.kev2,
        verbose=args.verbose
    )
    
    t1 = time.perf_counter()
    cathode_hist_build_time = t1 - t0
    
    if args.verbose:
        print(f"\nCathode histogram summary:")
        print(f"  - File1 cathode channels: {len(cathode_hists1)}")
        print(f"  - File2 cathode channels: {len(cathode_hists2)}")
    
    # ----- Step 4: calibrate cathode histograms ----
    if args.verbose:
        print("\nFitting cathode channels...")
    
    t0 = time.perf_counter()
    
    common_cathode_channels = set(cathode_hists1.keys()).intersection(set(cathode_hists2.keys()))
    
    if args.verbose:
        print(f"  - Common cathode channels: {len(common_cathode_channels)}")
    
    failed_cathode_fits = 0
    for channel in common_cathode_channels:
        hist1 = cathode_hists1[channel]
        hist2 = cathode_hists2[channel]
        
        # fitting current channel's histograms from both isotopes
        result1 = fit_one_cathode(hist1, cathode_bin_centers1)
        if result1 is None:
            failed_cathode_fits += 1
            calibrations[channel] = (0.5, -550)
            continue
        fit1_slope, fit1_intercept, fit1_adc, (fit1_range_lo, fit1_range_hi) = result1
        
        result2 = fit_one_cathode(hist2, cathode_bin_centers2)
        if result2 is None:
            #if fit fails, apply default calibration vals
            failed_cathode_fits += 1
            calibrations[channel] = (0.5, -550)
            continue
        
        fit2_slope, fit2_intercept, fit2_adc, (fit2_range_lo, fit2_range_hi) = result2
        
        #  use ADC values to get calibration values for current channel
        slope = (args.kev1 - args.kev2) / (fit1_adc - fit2_adc) 
        #adc1 = Cs137 adc
        #adc2 = Ge68 adc
        intercept = args.kev2 - slope * fit2_adc  
        
        # store current cathode channel's calibration parameters
        calibrations[channel] = (slope, intercept)
        
    t1 = time.perf_counter()
    cathode_fit_time = t1 - t0
    
    if args.verbose:
        print(f"\nCathode fitting complete:")
        print(f"  - Failed fits: {failed_cathode_fits}/{len(common_cathode_channels)}")
        print(f"  - Success rate: {100*(1-failed_cathode_fits/len(common_cathode_channels)):.1f}%")
    
    # ----- Step 5: save calibration parameters to output file ----
    if args.verbose:
        print(f"\nWriting calibration values to: '{args.out}'")
    
    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in calibrations.items()}, f, indent=2)
    
    # ------ (optional) Step 6: print runtime benchmarks ------
    if args.verbose:
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Anode histograms built in:   {anode_histograms_build_time:.3f} seconds")
        print(f"Anode histograms fitted in:  {anode_fit_time:.3f} seconds")
        print(f"  → Avg per channel:          {(anode_fit_time)/len(common_anode_channels):.4f} seconds")
        print(f"Cathode histograms built in: {cathode_hist_build_time:.3f} seconds")
        print(f"Cathode histograms fitted in: {cathode_fit_time:.3f} seconds")
        print(f"  → Avg per channel:          {(cathode_fit_time)/len(common_cathode_channels):.4f} seconds")
        print("=" * 60)
        
if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"\nTotal runtime: {end - start:.3f} seconds")