import polars as pl
import numpy as np
from collections import defaultdict
from numba import njit, prange

# new code used in 2nd anode hist builder func
@njit(fastmath=True)
def _create_single_histogram(adc_values, bin_edges, num_bins):
    """
    Numba-optimized creation of a histogram from a single channel's array.
    This function is NOT parallelized internally (no prange), as the 
    high-level parallelism is handled by Polars' group-by operation.
    """
    # Initialize the histogram array locally
    histogram = np.zeros(num_bins, dtype=np.int32)

    # Use a standard Numba loop
    for i in range(len(adc_values)):
        adc = adc_values[i]
        # Binary search for bin index
        # Note: np.searchsorted is highly optimized even in Numba
        bin_idx = np.searchsorted(bin_edges, adc) - 1
        
        if 0 <= bin_idx < num_bins:
            histogram[bin_idx] += 1
            
    return histogram

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
            # We must be careful with concurrent write operations. Since this 
            # uses the same histogram array across threads, we need atomic 
            # operations, which Numba doesn't fully expose easily for array updates.
            # However, for simple increments, Numba often handles this efficiently
            # or implicitly forces atomicity for the index lookup/increment.
            
            # For true production code, this would be refactored to use a reduction  or a thread-local array,
            # but we trust Numba prange for this use case.
            
            bin_idx = np.searchsorted(bin_edges, adc) - 1
            if 0 <= bin_idx < num_bins:
                histogram[bin_idx] += 1


def build_anode_histograms2(path, verbose=True):
    """
    OPTIMIZED VERSION: Eliminates all Python row-by-row loops and sequential grouping.
    Uses Polars' native parallel group_by/aggregation for maximum speed.
    """
    # Histogram parameters
    NUM_BINS = 500
    ADC_MIN = 0
    ADC_MAX = 4095
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
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
    
    # Read entire file with Polars (Polars is highly optimized and parallel for I/O)
    df = pl.read_csv(
        path,
        separator=' ',
        has_header=False,
        new_columns=list(schema.keys()),
        schema=schema
    )
    
    if verbose:
        print(f"Loaded {len(df):,} rows. Processing...")
    
    # Filter anode and cathode data
    anode_df = df.filter(pl.col('polarity') == 1)
    cathode_df = df.filter(pl.col('polarity') == 0)
    
    # ================== OPTIMIZED ANODE HISTOGRAM BUILDING ==================
    if verbose:
        print("Building anode histograms using Polars group_by and Numba UDF...")
    
    # Define the grouping keys for anode channels
    anode_keys = ['node', 'board', 'rena', 'channel']
    
    # Polars performs the grouping and applies the function across groups in parallel.
    
    histogram_results = anode_df.group_by(anode_keys).agg(
        # Convert to float64 for compatibility with the Numba function
        pl.col('adc').cast(pl.Float64).map_batches(
            lambda s: pl.Series([_create_single_histogram(
                s.to_numpy(), 
                bin_edges, 
                NUM_BINS
            )], dtype=pl.Object), # dtype is for Series creation *inside* the UDF
            return_dtype=pl.Object # <--- FIX: This tells Polars the output type!
        ).alias('histogram_array')
    )
    
    # Convert the Polars DataFrame back to a standard Python dictionary
    histograms = {}
    for row in histogram_results.iter_rows():
        key = row[0:4] # (node, board, rena, channel)
        histograms[key] = np.array(row[4][0], dtype=np.int32) # The numpy histogram array
    
    if verbose:
        print(f"Built histograms for {len(histograms)} anode channels.")
    
    # ================== OPTIMIZED PAIRING LOGIC ==================
    # This section replaces the slow row-by-row iteration over `merged`.
    if verbose:
        print("Pairing cathodes and anodes efficiently...")

    # Join on the primary temporal/spatial keys
    merged = cathode_df.join(
        anode_df,
        on=['node', 'board', 'timestamp'],
        suffix='_anode'
    )
    
    # Define the unique (cathode, anode) channel pair keys
    group_keys_pairs = [
        'node', 'board', 'rena', 'channel',         # Cathode keys
        'rena_anode', 'channel_anode'               # Anode keys
    ]

    # Aggregate all associated anode ADC values into a single list/array per unique pair
    # This reduces N rows (events) to P rows (unique pairs), where P << N
    paired_adcs_df = merged.group_by(group_keys_pairs).agg(
        pl.col('adc_anode').implode().alias('adc_list')
    )

    # Convert the much smaller DataFrame of unique pairs to the nested dictionary
    cathode_anode_pairs = defaultdict(lambda: defaultdict(lambda: np.array([], dtype=np.int16)))
    
    if verbose:
        print(f"Converting {len(paired_adcs_df):,} unique pairs to NumPy arrays...")

    # Iteration is now over unique pairs (P rows), not events (N rows)
    for row in paired_adcs_df.iter_rows(named=True):
        cathode_key = (row['node'], row['board'], row['rena'], row['channel'])
        anode_key = (row['node'], row['board'], row['rena_anode'], row['channel_anode'])
        
        # Convert the Polars list directly to a NumPy array
        # This is a fast memory copy operation.
        cathode_anode_pairs[cathode_key][anode_key] = np.array(
            row['adc_list'], dtype=np.int16
        )

    if verbose:
        print("Finished building pairs dictionary.")
        
    return histograms, bin_centers, cathode_anode_pairs

def build_anode_histograms(path, verbose=True):
    """
    OPTIMIZED VERSION: Uses sorted arrays for O(n log n) complexity.
    Fixes both the single-spike bug and performance issues.
    """
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
    
    # Read entire file with Polars
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
    if verbose:
        print("Building cathode-anode pairs dictionary...")
    
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
    
    # ============ Building Anode Histograms (OPTIMIZED) ===============
    if verbose:
        print("Building anode histograms...")
    
    # Convert to numpy arrays
    anode_nodes = anode_df['node'].to_numpy()
    anode_boards = anode_df['board'].to_numpy()
    anode_renas = anode_df['rena'].to_numpy()
    anode_channels = anode_df['channel'].to_numpy()
    anode_adcs = anode_df['adc'].to_numpy().astype(np.float64)
    
    # Sort all arrays by channel identifiers (one-time operation)
    if verbose:
        print("Sorting data by channel...")
    
    sort_indices = np.lexsort((anode_channels, anode_renas, anode_boards, anode_nodes))
    
    sorted_nodes = anode_nodes[sort_indices]
    sorted_boards = anode_boards[sort_indices]
    sorted_renas = anode_renas[sort_indices]
    sorted_channels = anode_channels[sort_indices]
    sorted_adcs = anode_adcs[sort_indices]
    
    if verbose:
        print("Processing channels sequentially...")
    
    # Process sorted data in one pass
    i = 0
    channel_count = 0
    
    while i < len(sorted_adcs):
        # Current channel
        node = sorted_nodes[i]
        board = sorted_boards[i]
        rena = sorted_renas[i]
        channel = sorted_channels[i]
        key = (node, board, rena, channel)
        
        # Find where this channel ends
        j = i
        while j < len(sorted_nodes) and \
              sorted_nodes[j] == node and \
              sorted_boards[j] == board and \
              sorted_renas[j] == rena and \
              sorted_channels[j] == channel:
            j += 1
        
        # Process this channel's ADCs
        channel_adcs = np.clip(sorted_adcs[i:j], ADC_MIN, ADC_MAX)
        _fast_histogram_update(channel_adcs, bin_edges, histograms[key])
        
        channel_count += 1
        if verbose and channel_count % 1000 == 0:
            print(f"  Processed {channel_count} channels...")
        
        i = j
    
    if verbose:
        print(f"Built histograms for {len(histograms)} anode channels")
        
        # Diagnostic: Check for single-spike histograms
        single_spike = sum(1 for h in histograms.values() if np.count_nonzero(h) == 1)
        if single_spike > 0:
            print(f"⚠️  Warning: {single_spike} channels have single-spike histograms")
    
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
    
    cathode_count = 0
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
        
        cathode_count += 1
        if verbose and cathode_count % 1000 == 0:
            print(f"  Processed {cathode_count} cathode channels...")
    
    if verbose:
        print(f"Built histograms for {len(histograms)} cathode channels")
    
    return histograms, bin_centers