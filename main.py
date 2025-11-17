import numpy as np
import matplotlib as plt
import argparse
from anode_fit import anode_fit
from build_hists import build_histogram

def main():
    parser = argparse.ArgumentParser(description="Calibrate ADC->keV from two large txt files (streaming).")
    parser.add_argument("--file1", required=True, help=".txt file for isotope 1 (whitespace-delimited)")
    parser.add_argument("--kev1", required=True, type=int, help="Known keV for file1")
    parser.add_argument("--file2", required=True, help=".txt file for isotope 2 (whitespace-delimited)")
    parser.add_argument("--kev2", required=True, type=int, help="Known keV for file2")
    parser.add_argument("--out", default="calibration.kev", help="Output .kev calibration file")
    parser.add_argument("--verbose", default=True, help="Print progress messages")
    args = parser.parse_args()

    # ----- step1: build histograms -----
    
    # hists1 and hists2 are default dictionaries with the format:
    # key = (node,board,rena,channel) : value = numpy array (per-channel histogram
    if args.verbose:
        print(f"Building anode histograms for file 1: {args.file1}")
    anode_hists1 = build_histogram(args.file1, verbose=args.verbose)

    if args.verbose:
        print(f"Building anode histograms for file 2: {args.file2}")
    anode_hists2 = build_histogram(args.file2, verbose=args.verbose)

    # ----- Step 2: compute ADC values from each histogram -----
    
    # We'll iterate over channels that exist in both files
    common_channels = set(hists1.keys()).intersection(set(hists2.keys()))

    # Placeholder for results: linear mapping ADC->keV
    # For now we'll just store the ADC "peak" placeholder (to be computed later)
    channel_peaks = {}

    for channel in common_channels:
        hist1 = hists1.get(channel, None)
        hist2 = hists2.get(channel, None)

        # Skip channels missing in either file (could also handle differently)
        if hist1 is None or hist2 is None:
            continue

        # Just store the histograms for now
        channel_peaks[channel] = {
            "hist1": hist1,
            "hist2": hist2
        }

    if args.verbose:
        print("Histograms for all channels prepared. Ready for peak detection and ADC->keV mapping.")
    

    
    adc1 = anode_fit(anode_hists1)
    adc2 = anode_fit(anode_hists2)
    
    
if __name__ == "__main__":
    main()