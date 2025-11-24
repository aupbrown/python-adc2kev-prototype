import pandas as pd
import numpy as np
from collections import defaultdict

def build_anode_histograms(path, verbose=True):
    """
    Read the whitespace-delimited .txt file in chunks and build histograms per channel.
    Columns expected per row: node board rena channel polarity energy u v cts
    Returns: dict mapping (node, board, rena, channel) -> numpy array
    """
    CHUNK_SIZE = 1_000_000 #1 million rows read at a time from .txt
    
    #defining histogram parameters
    NUM_BINS = 500
    ADC_MIN = 0
    ADC_MAX = 4095
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS+1)
    bin_width = bin_edges[1]-bin_edges[0]
    
    # histograms[channel_index, bin_index] will accumulate counts
    # histograms is a defaultdict with keys as (node,board,rena,channel) and numpy arrays that represent channel histograms as values.
    histograms = defaultdict(lambda: np.zeros(NUM_BINS, dtype=np.int32))
    
    # dictionary for holding cathode anode pairs
    cathode_anode_pairs = defaultdict(lambda: defaultdict(list))
    
    colnames = ["node","board","rena","channel","polarity","adc","u","v","timestamp"]
    
    dtypes = {
    "node": np.int8,
    "board": np.int8,
    "rena": np.int8,
    "channel": np.int8,
    "polarity": np.int8,
    "adc": np.int16,
    "u": np.int16,
    "v": np.int16,
    "timestamp": np.int64
    }
    
    # Read .txt file in chunks
    # we can iterate through reader to get our chunks
    reader = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        names=colnames,
        dtype=dtypes,
        chunksize=CHUNK_SIZE
    )
    
    for chunk_idx, chunk in enumerate(reader):
        # chunk is a pandas dataframe
        # each row of chunk is an event, with same columns as .txt file
        if verbose:
            print(f"Processing chunk #{chunk_idx}...")
        
        # ================== PAIRING LOGIC ==================

        cathode_chunk = chunk[chunk['polarity'] == 0]
        anode_chunk = chunk[chunk['polarity'] == 1] #polarity == 1 gives us anode channels only

        # Group anodes by (node, board, timestamp)
        merged = cathode_chunk.merge(
            anode_chunk,
            on=['node', 'board', 'timestamp'],
            suffixes=('_cathode', '_anode')
        )

        for row in merged.itertuples(index=False):
            cathode_key = (row.node, row.board, row.rena_cathode, row.channel_cathode)
            anode_key   = (row.node, row.board, row.rena_anode, row.channel_anode)
            cathode_anode_pairs[cathode_key][anode_key].append(row.adc_anode)
        
        # ============ Building Anode Histograms ===============
        
        # adc_values is a NumPy array of ADC readings (fast to work with).
        adc_values = anode_chunk['adc'].to_numpy()
        bin_indices = np.searchsorted(bin_edges, adc_values, side='right') - 1
        valid_mask = (bin_indices >= 0) & (bin_indices < NUM_BINS)

        # Get all channels as tuples
        channel_keys = list(zip(anode_chunk['node'], anode_chunk['board'], anode_chunk['rena'], anode_chunk['channel']))

        # Only keep valid indices
        bin_indices = bin_indices[valid_mask]
        channel_keys = np.array(channel_keys)[valid_mask]

        # Increment histograms in a vectorized way
        # For each event, we:
        # 1) Turn its [node,board,rena,channel] vector into a tuple key.
        # 2) Increment the right bin in that channel’s histogram.
        for key, bin_idx in zip(channel_keys, bin_indices):
            histograms[tuple(key)][bin_idx] += 1

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return histograms, bin_centers, cathode_anode_pairs

def build_cathode_histograms(cathode_anode_pairs, calibrations, target_kev):
    
    # defining histogram parameters
    NUM_BINS = 500
    ADC_MIN = 0
    ADC_MAX = 4095
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS+1)
    bin_width = bin_edges[1]-bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    histograms = defaultdict(lambda: np.zeros(NUM_BINS, dtype=np.int32))
    
    # iterate over each cathode channel and create a histogram using filtered, paired anode event ADC values.
    for cathode_channel in cathode_anode_pairs.keys():
        filtered_cathode_adcs = [] # list of filtered ADC values for current cathode channel
        # for each cathode channel, iterate over all of its paired anodes'
        for anode_channel in cathode_anode_pairs[cathode_channel].keys():
            if anode_channel not in calibrations:
                continue
            # get calibration values for current anode channel
            slope, intercept = calibrations[anode_channel]
            if slope is None or intercept is None:
                continue
            
            # iterate through all of each paired anode's events to filter non-target kev events
            for adc in cathode_anode_pairs[cathode_channel][anode_channel]:
                #calculate current event's kev using calibration values for current channel
                current_kev = slope * adc + intercept
                # only add anode events that are known to be within +/- 10% of target kev
                if ( current_kev >= 0.9*target_kev and current_kev <= 1.1*target_kev):
                    filtered_cathode_adcs.append(adc)

        cathode_adc_array = np.array(filtered_cathode_adcs)
        
        bin_indices = np.searchsorted(bin_edges, cathode_adc_array, side='right') - 1
        valid = (bin_indices >= 0) & (bin_indices < NUM_BINS)

        for idx in bin_indices[valid]:
            histograms[cathode_channel][idx] += 1
        
    return histograms, bin_centers