import numpy as np
import matplotlib.pyplot as plt

def plot_avg_sta_with_filter(sta, filter):
    sta = sta.mean(axis = 2)  # axis 2 is the time axis
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sta, vmin = -2, vmax = 2, cmap = "seismic")
    a = axs[1].imshow(filter, vmin = -2, vmax = 2, cmap = "seismic")
    plt.colorbar(a, label="Intensity")
    plt.title(f'STA + Filter')
    plt.show()

def plot_simple_sta_lags(sta, filter, start, end):
    n_lags = end - start
    
    fig, axs = plt.subplots(1, n_lags + 1, figsize=(3*(n_lags + 1), 4))
    for i_lag in range(n_lags): 
        im = axs[i_lag].imshow(sta[:, :, i_lag], vmin=-2, vmax=2, cmap="seismic") 
        axs[i_lag].set_title(f"lag {i_lag + start}") 

    filt_im = axs[-1].imshow(filter, vmin=-2, vmax=2, cmap="seismic") 
    axs[-1].set_title("Filter") 

    fig.colorbar(filt_im, ax=axs.ravel().tolist(), label="Intensity")