import numpy as np
import matplotlib.pyplot as plt

def plot_mean_sta_with_filter(mean_sta, filter):
    """
    Helper function to plot the mean sta (mean across all lags) together with the ground truth filter.

    Args:
        mean_sta (np.ndarray): 2 dimensional array containing the average stimulus that preceded spikes
        filter (np.ndarray): 2 dimensional array containing the filter used for the simulation of spikes

    Returns:
        None
    
    Notes:
        This function is used for illustration and plotting.
    """
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(mean_sta, vmin = -2, vmax = 2, cmap = "seismic")
    a = axs[1].imshow(filter, vmin = -2, vmax = 2, cmap = "seismic")
    plt.colorbar(a, label="Intensity")
    plt.title(f'STA + Filter')
    plt.show()

def plot_simple_sta_lags(sta, filter, start, end):
    """
    Helper function to plot all lags of an sta together with the ground truth filter.

    Args:
        sta (np.ndarray): 2 dimensional array containing the average stimulus that preceded spikes
        filter (np.ndarray): 2 dimensional array containing the filter used for the simulation
        start (int): Offset for the time indices within which a stimulus is considered relevant for a spike (start). 
        end (int): Offset for the time indices within which a stimulus is considered relevant for a spike (end).

    Returns:
        None
    
    Notes:
        This function is used for illustration and plotting.
    """
    n_lags = end - start
    
    fig, axs = plt.subplots(1, n_lags + 1, figsize=(3*(n_lags + 1), 4))
    for i_lag in range(n_lags): 
        im = axs[i_lag].imshow(sta[:, :, i_lag], vmin=-2, vmax=2, cmap="seismic") 
        axs[i_lag].set_title(f"lag {i_lag + start}") 

    filt_im = axs[-1].imshow(filter, vmin=-2, vmax=2, cmap="seismic") 
    axs[-1].set_title("Filter") 

    fig.colorbar(filt_im, ax=axs.ravel().tolist(), label="Intensity")