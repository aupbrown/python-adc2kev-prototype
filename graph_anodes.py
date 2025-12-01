import numpy as np
import argparse
from build_hists_opt import build_anode_histograms, build_cathode_histograms
import matplotlib.pyplot as plt

# we expect file1 = CS137, kev1 = 662, file2 = Ge68, kev2 = 511
parser = argparse.ArgumentParser(description="Calibrate ADC->keV from two .txt files (streaming).")
parser.add_argument("--file1", required=True, help=".txt file for isotope 1 (whitespace-delimited)")
parser.add_argument("--kev1", default=662, type=float, help="Known keV for file1")
parser.add_argument("--file2", required=True, help=".txt file for isotope 2 (whitespace-delimited)")
parser.add_argument("--kev2", default=511, type=float, help="Known keV for file2")
parser.add_argument("--out", default="calibration.json", help="Output calibration file")
parser.add_argument("--verbose", action='store_true', default=True, help="Print progress messages")
args = parser.parse_args()
    
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

# graphing 10 histograms from both isotopes 
keys = iter(set(anode_hists1.keys()) & set(anode_hists2.keys()))

for index, key in enumerate(keys):
    if index > 10:
        hist = anode_hists1[key]
        plt.figure(figsize=(10, 6))
        plt.step(anode_bin_centers1, hist, where='mid', linewidth=1.5)
        plt.xlabel("ADC value", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.title(f"Full Histogram: node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
        plt.ylim(bottom=0.5)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        if index == 20:
            break

for index, key in enumerate(keys):
    if index > 10:
        hist = anode_hists2[key]
        plt.figure(figsize=(10, 6))
        plt.step(anode_bin_centers2, hist, where='mid', linewidth=1.5)
        plt.xlabel("ADC value", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.title(f"Full Histogram: node={key[0]}, board={key[1]}, rena={key[2]}, channel={key[3]}")
        plt.ylim(bottom=0.5)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        if index == 20:
            break