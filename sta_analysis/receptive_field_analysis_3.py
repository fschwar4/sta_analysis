import toml
import numpy as np
import matplotlib.pyplot as plt

class STA:
    def __init__(self, stimulus, spike_train, filter, config_file:str='config.toml'):
        with open(config_file, "r") as f:
            config = toml.load(f)
        self.stimulus = stimulus.astype(np.float32).copy()
        # self.stimulus -= self.stimulus.mean()
        self.spike_train = spike_train
        self.filter = filter
        self.start = config["sta_parameters"]["start"]
        self.end = config["sta_parameters"]["end"]
        
        self.sta = None
        self.stc = None
        self.stc_minus_C = None
        self.eigen_results = None  

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
        height, width, _ = self.stimulus.shape
        sta = np.zeros((height, width, n_lags))
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
        
        # for i_lag in range(n_lags): # And I think I have to do this
        #     sta[:, :, i_lag] -= sta[:, :, i_lag].mean()
        
        self.sta = sta
        self.mean_sta = self.sta.mean(axis = 2)
        return sta
    
    # comments are kinda just for myself rn
    def calc_stc(self) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
        # first: take all spikes
        spikes, = np.where(self.spike_train == 1)
        if spikes.size == 0:
            raise ValueError('No spikes found.')

        # get some constants
        n_lags = self.end - self.start
        height, width, n_times = self.stimulus.shape
        n_pixels = height * width

        # since stc is based on sta we need that.
        if self.sta is None:
            self.calc_sta()

        # preallocation similar to sta
        stimuli = [np.zeros((spikes.size, n_pixels)) for _ in range(n_lags)]
        spike_count = 0

        # go throgh all spikes and...
        for spike in spikes:
            window_start = spike + self.start
            window_end = spike + self.end  # get index dimensions (start/end)
            if window_start < 0 or window_end > n_times:  # continue if oor
                continue

            for i_lag, time_point in enumerate(range(window_start, window_end)):
                stimuli[i_lag][spike_count] = self.stimulus[:, :, time_point].flatten()  # save the stimulus as flattened (16x16 -> 256x1) vector into the stimuli entry for each lag.
            spike_count += 1

        # trim excess rows, im not sure if i need this
        for i_lag in range(n_lags):
            stimuli[i_lag] = stimuli[i_lag][:spike_count]

        # now get stc for each lag
        stc_list = []
        for i_lag in range(n_lags):
            sta_flat = self.sta[:, :, i_lag].flatten()
            centered = stimuli[i_lag] - sta_flat
            cov_matrix = np.cov(centered, rowvar=False, bias=False)
            stc_list.append(cov_matrix)

        # get overall stimulus covariance (C)
        all_stimuli = np.reshape(self.stimulus, (n_pixels, n_times)).T
        s0 = all_stimuli.mean(0, keepdims=True)
        C  = np.cov(all_stimuli - s0, rowvar=False, bias=False)
        # C = np.cov(all_stimuli, rowvar=False, bias=False)

        # STC - C
        stc_minus_C_list = [stc - C for stc in stc_list]

        # compute eigenvalues and eigenvectors
        eigen_results = []
        for i_lag in range(n_lags):
            matrix = stc_minus_C_list[i_lag] 
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigen_results.append((eigenvalues, eigenvectors))

        self.stc = stc_list
        self.stc_minus_C = stc_minus_C_list
        self.eigen_results = eigen_results

        return stc_list, stc_minus_C_list, eigen_results


    def calc_stc_simplified(self):
        spikes, = np.where(self.spike_train == 1)
        if spikes.size == 0:
            raise ValueError('No spikes found.')
        
        n_lags = self.end - self.start
        height, width, n_times = self.stimulus.shape
        n_pixels = height * width
        n_spikes = len(spikes)

        if self.sta is None:
            self.calc_sta()

        stc_list = []
        for i_lag in range(n_lags):
            stimuli = np.zeros((n_spikes, n_pixels))
            valid_spikes = 0

            for spike in spikes:
                window_start = spike + self.start
                window_end = spike + self.end
                if window_start < 0 or window_end > n_times:
                    continue

                time_point = window_start + i_lag
                stimuli[valid_spikes] = self.stimulus[:, :, time_point].flatten()
                valid_spikes += 1

            if valid_spikes == 0:
                raise ValueError(f'No valid spikes found for lag {i_lag + self.start}.')

            stimuli = stimuli[:valid_spikes]

            sta_flat = self.sta[:, :, i_lag].flatten()
            centered = stimuli - sta_flat
            
            stc = np.dot(centered.T, centered) / (valid_spikes - 1)
            stc_list.append(stc)

        all_stimuli = np.reshape(self.stimulus, (n_pixels, n_times)).T
        all_stimuli_centered = all_stimuli - all_stimuli.mean(axis=0)
        C = np.dot(all_stimuli_centered.T, all_stimuli_centered) / (n_times - 1)

        # stc_minus_C_list = [np.abs(stc - C) for stc in stc_list]  # Somehow abs gives better results
        stc_minus_C_list = [stc - C for stc in stc_list]

        eigen_results = []
        for i_lag in range(n_lags):
            matrix = stc_minus_C_list[i_lag]
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            idx = np.abs(eigenvalues).argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # eigenvectors = eigenvectors[:, idx] * np.sign(eigenvalues[idx])
            # eigenvectors = eigenvectors[:, idx] * eigenvalues[idx]
            # eigenvectors = eigenvectors[:, idx] * np.sign(eigenvectors[0, idx])
            eigen_results.append((eigenvalues, eigenvectors))

        self.stc = stc_list
        self.stc_minus_C = stc_minus_C_list
        self.eigen_results = eigen_results

        return stc_list, stc_minus_C_list, eigen_results




    # quick chatgpt plotting fn
    def plot_eigenvectors_all_lags(self, top_n=3, cmap='seismic'):

        if self.eigen_results is None:
            raise ValueError("Run calc_stc() first to compute eigen_results.")

        n_lags   = len(self.eigen_results)          # number of separate Σ–C blocks
        height   = self.stimulus.shape[0]
        width    = self.stimulus.shape[1]
        n_cols   = n_lags + 1                       # +1 for the “average” column
        top_n    = min(top_n, self.eigen_results[0][1].shape[1])

        # Figure canvas ----------------------------------------------------------------
        fig, axes = plt.subplots(top_n, n_cols,
                                figsize=(2 * n_cols, 2 * top_n),
                                squeeze=False)

        # place-holders for accumulating the average over lags -------------------------
        avg_sums = [np.zeros((height, width)) for _ in range(top_n)]

        # go through the lags -----------------------------------------------------------
        for lag_idx, (eigenvalues, eigenvectors) in enumerate(self.eigen_results):
            # sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]   # |λ| descending
            # order = np.argsort(np.abs(eigenvalues))[::-1]
            for rank in range(top_n):
                ev2d = eigenvectors[:, rank].reshape(height, width)
                val   = eigenvalues[rank]
                # accumulate for later average
                avg_sums[rank] += ev2d * val
                
                # plot this lag
                ax = axes[rank, lag_idx]
                im = ax.imshow(ev2d * val, cmap=cmap)
                if rank == 0:
                    ax.set_title(f"Lag {lag_idx + self.start}\n"
                            f"Idx: {rank}: {val:.2f}")
                else:
                    ax.set_title(f"Idx: {rank}: {val:.2f}")
                
                ax.axis('off')


        # add the average column --------------------------------------------------------
        for rank in range(top_n):
            avg_ev2d = avg_sums[rank] / n_lags

            ax = axes[rank, -1]                      # last column
            im = ax.imshow(avg_ev2d, cmap=cmap)
            if rank == 0:
                ax.set_title("Avg\n(all lags)")
            ax.axis('off')

        # annotate rows with eigenvalue (from first lag just for reference)
        for rank in range(top_n):
            axes[rank, 0].set_ylabel(f"EV {rank+1}", rotation=0,
                                    ha='right', va='center')

        # single colour-bar -------------------------------------------------------------
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cbar_ax)

        plt.suptitle("Top Eigenvectors per Lag + Average Across Lags")
        plt.show()

    def plot_eigenvalue_spectrum(self, i_lag=0, abs_values=True, log_y=True):
        """
        Plot eigen-value versus index for the specified lag.

        Parameters
        ----------
        lag : int
            Index into self.eigen_results (default = 0).
        """
        if self.eigen_results is None:
            raise RuntimeError("Run calc_stc() first")

        # pick the eigen-values for the chosen lag
        eigenvalues = self.eigen_results[i_lag][0]   # tuple = (eigenvalues, eigenvectors)

        x = np.arange(1, len(eigenvalues) + 1)     # 1, 2, 3, ...
        y = eigenvalues                            # plotted as-is

        plt.figure(figsize=(4, 3))
        plt.plot(x, y, marker='o')
        plt.xlabel("Eigen-index")
        plt.ylabel("Eigen-value")
        plt.title(f"Eigen-value spectrum (lag {i_lag + self.start})")
        plt.tight_layout()
        plt.show()