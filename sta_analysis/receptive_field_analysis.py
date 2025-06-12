import numpy as np
import toml

class RFAnalysis:
    def __init__(self, stimulus, spike_train, filter, config_file:str='config.toml'):
        """
        Initialize the RFAnalysis class with stimulus, spike_train, filter, and configuration file.

        Args:
            stimulus (np.ndarray): 3D array representing the stimulus with shape [height, width, time].
            spike_train (np.ndarray): 1D array representing the spike train.
            filter (np.ndarray): 2D array representing the filter with shape [height, width].
            config_file (str): Path to the configuration file in TOML format. Default is 'config.toml'.
        """
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
        Sanity check for the given stimulus, spike_train and filter to ensure they are compatible.

        Args:
            None
        Raises:
            ValueError: If the stimulus is not a 3D array,
                        if the spike_train length does not match the time dimension of the stimulus,
                        or if the end time is before the start time.
        """

        if self.stimulus.ndim != 3:
            raise ValueError("Stimulus must be a 3D array: [height, width, time].")
        if len(self.spike_train) != self.stimulus.shape[2]:
            raise ValueError(f"spike_train length must match the third dimension (time) of stimulus: {len(self.spike_train)} vs {self.stimulus.shape[2]}")
        if self.end < self.start:
            raise ValueError("End must be after start.")


class STA(RFAnalysis):
    def __init__(self):
        self.sta = None

    def calc_sta(self, rf_analysis: RFAnalysis) -> np.ndarray:
        """
        Calculate the Spike-Triggered Average (STA) from the given RFAnalysis object.

        Args:
            rf_analysis (RFAnalysis): An instance of RFAnalysis containing stimulus and spike train.

        Returns:
            np.ndarray: The calculated STA.
        """
        stimulus = rf_analysis.stimulus
        spike_train = rf_analysis.spike_train
        start = rf_analysis.start
        end = rf_analysis.end

        # Extract the relevant time window from the stimulus
        relevant_stimulus = stimulus[:, :, start:end]

        # Calculate the STA
        sta = np.mean(relevant_stimulus[:, :, spike_train == 1], axis=2)
        
        return sta

class STC(RFAnalysis):
    def __init__(self):
