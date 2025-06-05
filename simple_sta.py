import numpy as np

def simple_sta(stimulus, spike_train, start = 0, end = 1):
    """
    Calculates the spike-triggered average (STA) of a 2D stimulus of shape [time, x, y]
    
    Args:
        stimulus (np.ndarray): 3D array of shape [time, height, width], where each 2D frame is the stimulus at a given time.
        spike_train (np.ndarray): 1D array of shape with the same length as stimulus in time, consisting of 0s and 1s (0: no spike, 1: spike)
        start (int): Start offset relative to each spike time.
        end (int): End offset relative to each spike time.

    Returns:
        sta: The average stimulus that caused a spike with respect to the given window (start/end).
    
    Note:
        By default start and end are set to take only the stimulus that was present during each spike.
    """
    
    spikes, = np.where(spike_train == 1)
    
    if not spikes:
        raise ValueError('Found no spikes in the given spike train.')
    
    sta = np.zeros((stimulus.shape[1],stimulus.shape[2]))
    pattern_count = 0

    for spike in spikes:
        window_start = spike + start
        window_end = spike + end
        if window_start < 0 or window_end > len(stimulus):
            print(f'Exceeding: {window_start} - {window_end}')
            continue

        for i in range(window_start, window_end + 1):
            sta += stimulus[i]
            pattern_count += 1

    if pattern_count == 0:
        raise ValueError('No patterns found. The heck?')

    sta /= pattern_count
    sta -= sta.mean()
    
    return sta