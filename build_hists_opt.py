import polars as pl
import numpy as np
from collections import defaultdict
import numba
from numba import njit, prange
from multiprocessing import Pool, cpu_count

@njit(parallel=True, fastmath=True)
def _fast_histogram_update(adc_values, bin_edges, histogram):
    """Numba-optimized histogram update using parallel processing."""
    num_bins = len(histogram)
    for i in prange(len(adc_values)):
        adc = adc_values[i]
        # Binary search for bin index
        bin_idx = np.searchsorted(bin_edges, adc) - 1
        if 0 <= bin_idx < num_bins:
            histogram[bin_idx] += 1

@njit(parallel=True, fastmath=True)
def _filter_and_bin_cathodes(adc_array, slope, intercept, target_kev, bin_edges, histogram):
    """Numba-optimized filtering and binning for cathode histograms."""
    num_bins = len(histogram)
    lower_bound = 0.9 * target_kev
    upper_bound = 1.1 * target_kev
    
    for i in prange(len(adc_array)):
        adc = adc_array[i]
        current_kev = slope * adc + intercept
        
        if lower_bound <= current_kev <= upper_bound:
            bin_idx = np.searchsorted(bin_edges, adc) - 1
            if 0 <= bin_idx < num_bins:
                histogram[bin_idx] += 1

def build_anode_histograms(path, verbose=True, n_workers=None):
    """
    Read the whitespace-delimited .txt file and build histograms per channel using Polars.
    Returns: dict mapping (node, board, rena, channel) -> numpy array
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Histogram parameters
    NUM_BINS = 500
    ADC_MIN = 0
    ADC_MAX = 4095
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    histograms = defaultdict(lambda: np.zeros(NUM_BINS, dtype=np.int32))
    cathode_anode_pairs = defaultdict(lambda: defaultdict(list))
    
    # Define schema for faster parsing
    schema = {
        "node": pl.Int8,
        "board": pl.Int8,
        "rena": pl.Int8,
        "channel": pl.Int8,
        "polarity": pl.Int8,
        "adc": pl.Int16,
        "u": pl.Int16,
        "v": pl.Int16,
        "timestamp": pl.Int64
    }
    
    if verbose:
        print("Reading file with Polars...")
    
    # Read entire file with Polars (much faster than pandas chunking)
    df = pl.read_csv(
        path,
        separator=' ',
        has_header=False,
        new_columns=list(schema.keys()),
        schema=schema
    )
    
    if verbose:
        print(f"Loaded {len(df):,} rows. Processing...")
    
    # Split into cathode and anode dataframes
    cathode_df = df.filter(pl.col('polarity') == 0)
    anode_df = df.filter(pl.col('polarity') == 1)
    
    # ================== PAIRING LOGIC ==================
    if verbose:
        print("Pairing cathodes and anodes...")
    
    # Efficient join using Polars
    merged = cathode_df.join(
        anode_df,
        on=['node', 'board', 'timestamp'],
        suffix='_anode'
    )
    
    # Build cathode-anode pairs dictionary using numpy arrays
    # First collect all pairs efficiently
    temp_pairs = defaultdict(lambda: defaultdict(list))
    for row in merged.iter_rows(named=True):
        cathode_key = (row['node'], row['board'], row['rena'], row['channel'])
        anode_key = (row['node'], row['board'], row['rena_anode'], row['channel_anode'])
        temp_pairs[cathode_key][anode_key].append(row['adc_anode'])
    
    # Convert lists to numpy arrays for much faster processing
    for cathode_key in temp_pairs:
        for anode_key in temp_pairs[cathode_key]:
            cathode_anode_pairs[cathode_key][anode_key] = np.array(
                temp_pairs[cathode_key][anode_key], dtype=np.int16
            )
    
    # ============ Building Anode Histograms ===============
    if verbose:
        print("Building anode histograms...")
    
    # Group by channel for efficient histogram building
    anode_groups = anode_df.group_by(['node', 'board', 'rena', 'channel'])
    
    for (node, board, rena, channel), group_df in anode_groups:
        key = (node, board, rena, channel)
        adc_values = group_df['adc'].to_numpy()
        
        # Use Numba-optimized histogram update
        _fast_histogram_update(adc_values, bin_edges, histograms[key])
    
    if verbose:
        print(f"Built histograms for {len(histograms)} anode channels")
    
    return histograms, bin_centers, cathode_anode_pairs


def build_cathode_histograms(cathode_anode_pairs, calibrations, target_kev, verbose=True):
    """
    Build cathode histograms using filtered anode ADC values.
    Optimized with Numba for speed.
    """
    # Histogram parameters
    NUM_BINS = 500
    ADC_MIN = 0
    ADC_MAX = 4095
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    histograms = defaultdict(lambda: np.zeros(NUM_BINS, dtype=np.int32))
    
    if verbose:
        print(f"Building cathode histograms for {len(cathode_anode_pairs)} channels...")
    
    # Process each cathode channel
    for cathode_channel in cathode_anode_pairs.keys():
        # Collect all filtered ADCs for this cathode
        for anode_channel, adc_array in cathode_anode_pairs[cathode_channel].items():
            if anode_channel not in calibrations:
                continue
            
            slope, intercept = calibrations[anode_channel]
            if slope is None or intercept is None:
                continue
            
            # adc_array is already numpy, just ensure correct dtype for Numba
            adc_array = adc_array.astype(np.float64)
            
            # Use Numba-optimized filtering and binning
            _filter_and_bin_cathodes(
                adc_array, 
                slope, 
                intercept, 
                target_kev, 
                bin_edges, 
                histograms[cathode_channel]
            )
    
    if verbose:
        print(f"Built histograms for {len(histograms)} cathode channels")
    
    return histograms, bin_centers


# Optional: Parallel version for very large datasets
def _process_cathode_chunk(args):
    """Helper function for parallel cathode processing."""
    cathode_channels, cathode_anode_pairs, calibrations, target_kev, bin_edges, NUM_BINS = args
    
    local_histograms = {}
    
    for cathode_channel in cathode_channels:
        histogram = np.zeros(NUM_BINS, dtype=np.int32)
        
        for anode_channel, adc_array in cathode_anode_pairs[cathode_channel].items():
            if anode_channel not in calibrations:
                continue
            
            slope, intercept = calibrations[anode_channel]
            if slope is None or intercept is None:
                continue
            
            # adc_array is already numpy
            adc_array = adc_array.astype(np.float64)
            _filter_and_bin_cathodes(adc_array, slope, intercept, target_kev, bin_edges, histogram)
        
        local_histograms[cathode_channel] = histogram
    
    return local_histograms


def build_cathode_histograms_parallel(cathode_anode_pairs, calibrations, target_kev, 
                                      verbose=True, n_workers=None):
    """
    Parallel version of cathode histogram building for very large datasets.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    NUM_BINS = 500
    ADC_MIN = 0
    ADC_MAX = 4095
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Split cathode channels into chunks for parallel processing
    cathode_channels = list(cathode_anode_pairs.keys())
    chunk_size = max(1, len(cathode_channels) // n_workers)
    chunks = [cathode_channels[i:i + chunk_size] 
              for i in range(0, len(cathode_channels), chunk_size)]
    
    if verbose:
        print(f"Processing {len(cathode_channels)} cathode channels using {n_workers} workers...")
    
    # Prepare arguments for each worker
    args_list = [
        (chunk, cathode_anode_pairs, calibrations, target_kev, bin_edges, NUM_BINS)
        for chunk in chunks
    ]
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(_process_cathode_chunk, args_list)
    
    # Merge results
    histograms = {}
    for result_dict in results:
        histograms.update(result_dict)
    
    if verbose:
        print(f"Built histograms for {len(histograms)} cathode channels")
    
    return histograms, bin_centers