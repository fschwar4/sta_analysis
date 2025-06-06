import toml
import numpy as np
from utils.plotter import *

class SpikeAnalysis:
    def __init__(self, stimulus, spike_train, filter, config_file:str='config.toml'):
        with open(config_file, "r") as f:
            config = toml.load(f)

        self.stimulus = stimulus
        self.spike_train = spike_train
        self.filter = filter
        self.start = config["sta_parameters"]["start"]
        self.end = config["sta_parameters"]["end"]
        
        self._sanity_checks()
    
    def _sanity_checks(self) -> None:
        """
        Sanity check for the given stimulus, spike_train and filter
        """
        if self.stimulus.ndim != 3:
            raise ValueError("Stimulus must be a 3D array: [height, width, time].")
        if len(self.spike_train) != self.stimulus.shape[2]:
            raise ValueError(f"spike_train length must match the third dimension (time) of stimulus: {len(self.spike_train)} vs {self.stimulus.shape[2]}")
        if self.end < self.start:
            raise ValueError("End must be after start.")
        
    def calc_sta(self) -> np.ndarray:
        """
        Calculates the simple spike triggered average (STA) in terms of lags. 
        """
        spikes, = np.where(self.spike_train == 1)
        if spikes.size == 0:
            raise ValueError('Found no spikes in the given spike train.')
        
        n_lags = self.end - self.start
        sta = np.zeros((self.stimulus.shape[0], self.stimulus.shape[1], n_lags))
        spike_count = 0

        for spike in spikes:
            window_start = spike + self.start
            window_end = spike + self.end
            if window_start < 0 or window_end > self.stimulus.shape[2]:
                print(f'Exceeding: {window_start} - {window_end}')
                continue

            for i_lag, time_point in enumerate(range(window_start, window_end)):
                sta[:, :, i_lag] += self.stimulus[:, :, time_point]
                
            spike_count += 1

        sta /= spike_count
        # sta -= sta.mean()  # Skipping this has actually given less noisy results
        # TODO: Get some metrics about noise and simulation length vs clarity of the sta
        
        for i_lag in range(n_lags): # And I think I have to do this
            sta[i_lag] -= sta[i_lag].mean()

        self.sta = sta
        self.mean_sta = self.sta.mean(axis = 2)
        return sta
        