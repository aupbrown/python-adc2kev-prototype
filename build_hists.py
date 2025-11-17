import pandas as pd
import numpy as np
from collections import defaultdict

def build_histograms(path, verbose=True):
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
    
    # used for initializing histograms and converting (n,b,r,c) tuples to flat index
    NUM_NODES = 10
    NUM_BOARDS = 48
    NUM_RENAS = 2
    NUM_CHANNELS_PER_RENA = 36
    TOTAL_CHANNELS = NUM_NODES * NUM_BOARDS * NUM_RENAS * NUM_CHANNELS_PER_RENA
    
    bin_edges = np.linspace(ADC_MIN, ADC_MAX, NUM_BINS+1)
    bin_width = bin_edges[1]-bin_edges[0]
    
    # histograms[channel_index, bin_index] will accumulate counts
    histograms = defaultdict(lambda: np.zeros(NUM_BINS, dtype=np.int32))
    
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
            
        # Vectorized binning
        anode_chunk = chunk[chunk['polarity'] == 1]
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
        
    return histograms, bin_centers