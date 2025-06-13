import numpy as np
import toml
import numpy.matlib as ml
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import scipy.linalg
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
        self.n_time_lags = config["sta_parameters"]["n_time_lags"]
        # results
        self.sta = None   # (H, W, L)
        self.stc = None   # (H*W*L, H*W*L)
        self.raw_mu  = None
        self.raw_cov = None
        self.stc_minus_C = None
        self.eigen_results = None

        self._sanity_checks()
        self._get_spikes()
    
    

    def calc_sta(self):
        self._get_spikes()

        n_lags = self.end - self.start
        height, width, _ = self.stimulus.shape
        sta = np.zeros((height, width, n_lags))

        for spike in self.spikes:  # spike is the index of the spike in the spike train
            window_start = self.start + spike
            window_end = self.end + spike

            for i_lag, time_point in enumerate(range(window_start, window_end)):
                sta[:, :, i_lag] += self.stimulus[:, :, time_point]

        sta /= len(self.spikes)
        
        # TODO: Subtract mean?

        self.sta = sta
        self.mean_sta = self.sta.mean(axis = 2)
        return sta
    
    def calc_stc(self):
        # Get Stack of Stimuli

        stack = self.stimulus.reshape(-1, self.stimulus.shape[2])
        RawMu = stack.mean(axis=1, keepdims=True)
        RawCov = np.cov(stack.T, rowvar=False)

        
        
        spike_stimulus = stack[:, self.spikes]

        STC = np.cov(spike_stimulus.T, rowvar=False)

        stc_minus_C = STC - RawCov

        eigenvals, eigenvecs = np.linalg.eigh(stc_minus_C)
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        self.stc_minus_C = stc_minus_C
        self.raw_cov = RawCov
        self.eigen_results = (eigenvals, eigenvecs)

    def _make_stim_rows(self):
        """
        Internal method to create time-embedded stimulus rows.
        Converts a 3D stimulus array [height, width, time] into a time-embedded format.
        Each row contains nkt video frames flattened and concatenated.
        Returns an array of shape [time - nkt + 1, nkt * height * width]
        """
        h, w, total_time = self.stimulus.shape
        stim_rows = []
        for t in range(self.n_time_lags - 1, total_time):
            # Stack nkt frames (across time), flatten, and concatenate
            window = self.stimulus[:, :, t - self.n_time_lags + 1: t + 1]  # shape: [h, w, n_time_lags]
            stim_rows.append(window.flatten())
        return np.array(stim_rows)

    def calc_stc_full(self):
        """
        Computes STA, STC, RawMu, RawCov for a 3D stimulus [height, width, time] using time-embedding.

        Args
            stimulus : array, shape [height, width, time]
                The 3D stimulus array (2D led patterns over time)
            spikes : array, shape [time]
                Binned spike counts or events for each time bin (should match time dimension)
            n_time_lags : int
                Number of time bins to include in time embedding (before the spike)

        Returns
        
        """
        h, w, total_time = self.stimulus.shape
        S = self._make_stim_rows()  # [samples, n_time_lags * h * w]
        valid_spikes = self.spike_train[self.n_time_lags - 1:]  # Discard spikes during first n_time_lags-1 pre-window

        nspikes = np.sum(valid_spikes)
        if nspikes == 0:
            raise ValueError("No spikes in aligned window! Cannot compute STC.")

  
        RawMu = np.mean(S, axis=0, keepdims=True).T        # [n_time_lags*h*w, 1]
        RawCov = np.cov(S, rowvar=False)                   # [n_time_lags*h*w, n_time_lags*h*w]

        STA = np.dot(valid_spikes, S) / nspikes            # [n_time_lags*h*w]
        STA = STA[:, np.newaxis]

        # STC (weighted)
        S_sp = S[valid_spikes > 0]
        weights = valid_spikes[valid_spikes > 0]
        S_sp_repeat = np.repeat(S_sp, weights.astype(int), axis=0)
        STC = np.cov(S_sp_repeat, rowvar=False)

        stc_minus_C = STC - RawCov
        eigenvals, eigenvecs = np.linalg.eigh(stc_minus_C)
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        self.sta = STA.reshape(h, w, self.n_time_lags)
        self.eigen_results = (eigenvals, eigenvecs)
        self.raw_mu = RawMu.reshape(h, w, self.n_time_lags)
        self.raw_cov = RawCov
        self.stc = STC
        self.stc_minus_C = stc_minus_C

    def _get_spikes(self):
        """
        Get the indices of spikes in the spike train.

        Returns:
            np.ndarray: Indices of spikes in the spike train.
        """
        all_spikes, = np.where(self.spike_train == 1)
        self.all_spikes = all_spikes

        mask = (all_spikes + self.start >= 0) & (all_spikes + self.end < len(self.spike_train))
        spikes = all_spikes[mask]
        if spikes.size == 0:
            raise ValueError('No spikes found in the valid time window defined by start and end.')

        self.spikes = spikes

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
    
    def plot_sta_lags(self):
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
        fig, axs = plt.subplots(1, self.n_time_lags + 1, figsize=(3*(self.n_time_lags + 1), 4))
        for i_lag in range(self.n_time_lags): 
            im = axs[i_lag].imshow(self.sta[:, :, i_lag], vmin=-2, vmax=2, cmap="seismic") 
            axs[i_lag].set_title(f"lag {i_lag - self.n_time_lags + 1}") 

        filt_im = axs[-1].imshow(self.filter, vmin=-1, vmax=1, cmap="seismic") 
        axs[-1].set_title("Filter") 

        fig.colorbar(filt_im, ax=axs.ravel().tolist(), label="Intensity")

    def plot_eigenvals_stc(self):
        """
        Plots the eigenvalues of the STC matrix.
        This function assumes that the STC matrix has already been computed and stored in `self.eigen_results`.
        """
        eigenvals, eigenvecs = self.eigen_results
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        plt.figure(figsize=(6, 4))
        plt.plot(eigenvals, marker='.', linestyle='None')
        plt.title('Eigenvalues of STC Matrix')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()

    def plot_eigenvecs_stc(self, top=3, vmin=-1, vmax=1):
        """
        Plots the top `top` eigenvectors of the STC matrix.
        Handles multiple nkt (time-embedding) by showing each time lag.
        """
        eigenvals, eigenvecs = self.eigen_results
        h, w, t = self.sta.shape

        for n in range(top):
            vec = eigenvecs[:, n] 
            vec3d = vec.reshape(h, w, t)

            fig, axes = plt.subplots(1, t, figsize=(4*t, 3))
            if t == 1:
                axes = [axes]
            for lag in range(t):
                ax = axes[lag]
                im = ax.imshow(vec3d[:, :, lag], cmap='seismic', vmin=vmin, vmax=vmax)
                ax.set_title(f'Lag {lag - self.n_time_lags + 1}')
                ax.axis('off')
            fig.suptitle(f'Eigenvector {n+1} (Eigenvalue: {eigenvals[n]:.2f})')
            fig.colorbar(im, ax=axes, fraction=0.045)
            plt.show()



