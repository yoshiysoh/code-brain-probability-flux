"""
Probability flux analysis of HCP fMRI data (the PFA class).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from scipy import interpolate, io, stats
from scipy.sparse import dok_array, find, triu
from sklearn.decomposition import PCA
from tqdm.notebook import trange

from ..spin_states import binary_codes_to_ising, generate_binary_codes

SCALE = 1 / 1000
DEFAULT_HCP_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "HCP_data_task"
)
LABEL_POS = [1.05, 1.5, 1.225, 1.4]
# HCP runs in task order; each is loaded for both phase encodings (LR, RL).
HCP_RUNS = [
    "REST1",
    "EMOTION",
    "WM",
    "SOCIAL",
    "LANGUAGE",
    "RELATIONAL",
    "GAMBLING",
    "MOTOR",
]


def _get_X_data(data, num_time_steps):
    # fMRI data
    X_data = data[0]["task_data"].T
    for i in range(len(data) - 1):
        X_data = np.vstack([X_data, data[i + 1]["task_data"].T])
    # Personal ID
    ID_data = data[0]["task_IDs"]
    for i in range(len(data) - 1):
        ID_data = np.hstack(
            (
                ID_data,
                ID_data.max() + data[i + 1]["task_IDs"],
            )
        )
    ID_data = ID_data[0]
    # Unique personal ID
    ID_unique = np.unique(ID_data)
    num_IDs = len(ID_unique)
    # Task ID
    task_data = np.zeros(
        len(data[0]["task_data"].T) + len(data[0]["task_data"].T)
    )
    for i in range(len(data) // 2 - 1):
        task_data = np.hstack(
            (
                task_data,
                (i + 1)
                * np.ones(
                    len(data[2 * (i + 1)]["task_data"].T)
                    + len(data[2 * (i + 1) + 1]["task_data"].T)
                ),
            )
        )

    # Restructure fMRI data based on personal ID
    @njit(cache=True, nogil=True, parallel=True)
    def restructure(num_time_steps, X_data, ID_data, task_data, num_IDs):
        """Stack the recordings into (time, parcel) data with recording and task labels."""
        X_data_temp = np.zeros((num_time_steps * num_IDs, X_data.shape[1]))
        ID_data_temp = np.zeros(num_time_steps * num_IDs)
        task_data_temp = np.zeros(num_time_steps * num_IDs)
        for i in prange(num_IDs):
            inds = np.where(ID_data == ID_unique[i])[0]
            X_data_temp[num_time_steps * i : num_time_steps * (i + 1), :] = (
                X_data[inds[:num_time_steps], :]
            )
            ID_data_temp[num_time_steps * i : num_time_steps * (i + 1)] = (
                ID_data[inds[:num_time_steps]]
            )
            task_data_temp[num_time_steps * i : num_time_steps * (i + 1)] = (
                task_data[inds[:num_time_steps]]
            )
        X_data = X_data_temp
        ID_data = ID_data_temp
        task_data = task_data_temp
        return X_data, ID_data, task_data

    X_data, ID_data, task_data = restructure(
        num_time_steps, X_data, ID_data, task_data, num_IDs
    )
    return X_data, ID_data, task_data, ID_unique, num_IDs


def normalized_hist2d(
    fig,
    ax,
    x_data,
    y_data,
    x_bins,
    y_bins,
    weight=None,
    cmap=plt.cm.viridis,
    plot_cbar=True,
    vmax=None,
    H=None,
):
    """Draw the normalized 2-D histogram of (x_data, y_data); pass H to
    draw a precomputed one (rows = y bins) instead."""
    if H is None:
        H, xedges, yedges = np.histogram2d(
            x_data, y_data, bins=(x_bins, y_bins), weights=weight
        )
        H = H.T
        H = H / H.sum()
    else:
        xedges, yedges = x_bins, y_bins
    xgrid, ygrid = np.meshgrid(xedges, yedges)
    if len(x_bins) != len(y_bins):
        cbar = ax.pcolormesh(xgrid, ygrid, H, cmap=cmap, vmin=0, vmax=vmax)
    else:
        cbar = ax.imshow(
            H,
            interpolation="nearest",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )

    if plot_cbar:
        fig.colorbar(cbar, ax=ax, label="Probability")
    return ax, cbar


def set_aspect_ratio(ax, ratio=1.0):
    """Set the axes aspect so the data box has the given ratio."""
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(np.abs((x_right - x_left) / (y_low - y_high)) * ratio)
    return ax


def as_sci(x, ndp):
    """LaTeX scientific notation of x with ndp decimals."""
    s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
    m, e = s.split("e")
    return rf"{m:s}\times 10^{{{int(e):d}}}"


@njit(cache=True, nogil=True)
def _grid_flux(transition_matrix, num_bins):
    """Net flux through each cell of a num_bins x num_bins grid.

    transition_matrix[source, target] holds net transition counts between
    cells (cell index x + y * num_bins). Returns (num_bins, num_bins, 2):
    the x and y components at each cell, each the sum of the net flux in
    from the lower neighbour and out to the upper neighbour.
    """
    flux = np.zeros((num_bins, num_bins, 2))
    for flux_y in range(num_bins):
        for flux_x in range(num_bins):
            target = flux_x + flux_y * num_bins
            if 0 <= flux_x - 1:
                left = flux_x - 1 + flux_y * num_bins
                flux[flux_y, flux_x, 0] += transition_matrix[left, target]
            if flux_x + 1 < num_bins:
                right = flux_x + 1 + flux_y * num_bins
                flux[flux_y, flux_x, 0] += transition_matrix[target, right]
            if 0 <= flux_y - 1:
                bottom = flux_x + (flux_y - 1) * num_bins
                flux[flux_y, flux_x, 1] += transition_matrix[bottom, target]
            if flux_y + 1 < num_bins:
                top = flux_x + (flux_y + 1) * num_bins
                flux[flux_y, flux_x, 1] += transition_matrix[target, top]
    return flux


@njit(cache=True, nogil=True)
def _bootstrapping(transition_list, t_tot, num_bins, rng, num_realization=100):
    """Resample PCA-grid transitions; return the flux of each realization."""
    vec_strapped = np.zeros((num_realization, num_bins**2, 2))
    for realization in range(num_realization):
        random_indices = rng.integers(
            0, transition_list.shape[0], size=transition_list.shape[0]
        )
        transition_list_strapped = transition_list[random_indices, :]
        transition_sample = np.zeros(
            (num_bins * num_bins, num_bins * num_bins), np.int64
        )
        for t in range(len(transition_list_strapped)):
            source = (
                transition_list_strapped[t, 0]
                + transition_list_strapped[t, 1] * num_bins
            )
            target = (
                transition_list_strapped[t, 2]
                + transition_list_strapped[t, 3] * num_bins
            )
            transition_sample[source, target] += 1
        net = (transition_sample - transition_sample.T).astype(np.float64)
        flux_sample = _grid_flux(net, num_bins)
        vec_strapped[realization, :, 0] = (
            0.5 * flux_sample[:, :, 0].flatten() / t_tot
        )
        vec_strapped[realization, :, 1] = (
            0.5 * flux_sample[:, :, 1].flatten() / t_tot
        )
    return vec_strapped


def draw_ellipses(ax, x, y, u, v, angle, std0, std1):
    """Draw a confidence ellipse centered at the tip of (u, v)."""
    X = x + u
    Y = y + v
    e = mpl.patches.Ellipse(
        xy=(X, Y),
        width=2 * std0,
        height=2 * std1,
        angle=np.rad2deg(angle),
        alpha=0.4,
        color="w",
        lw=0,
    )
    ax.add_patch(e)


@njit(cache=True, nogil=True, parallel=True)
def _sort_zscore_derivative_event(
    event_step_time_spin,
    ID_data_event,
    ID_unique,
    progress_proxy,
):
    event_step_time_spin_sorted = np.zeros_like(event_step_time_spin)
    for i in prange(len(ID_unique)):
        unique = ID_unique[i]
        inds_id = np.where(ID_data_event == unique)[0]
        inds_sort = np.argsort(event_step_time_spin[inds_id, 1])
        event_step_time_spin_sorted[inds_id] = event_step_time_spin[inds_id][
            inds_sort
        ]
        progress_proxy.update(1)
    event_step_time_spin = event_step_time_spin_sorted
    return event_step_time_spin


@njit(cache=True, nogil=True)
def _event_to_trajectory(
    time_between_steps,
    num_time_steps,
    zscore,
    event_step_time_spin,
    task_data_event,
    ID_data_event,
    initial_states,
    progress_proxy,
):
    num_initial_states = len(initial_states)
    final_time = time_between_steps * num_time_steps

    trajectories = np.zeros(
        (len(event_step_time_spin) + num_initial_states, zscore.shape[1])
    )
    ID_data_trajectory = np.zeros(
        len(event_step_time_spin) + num_initial_states
    )
    task_data_trajectory = np.zeros(
        len(event_step_time_spin) + num_initial_states
    )
    dts = np.zeros(len(event_step_time_spin) + num_initial_states)

    trajectories[0] = initial_states[0]
    ID_data_trajectory[0] = ID_data_event[0]
    task_data_trajectory[0] = task_data_event[0]
    dts[0] = event_step_time_spin[0, 1] - 0.0

    id_counter = 1
    for i in range(len(event_step_time_spin) - 1):
        changed_spin = np.int64(event_step_time_spin[i, 2])
        trajectories[i + id_counter] = trajectories[i + id_counter - 1].copy()
        trajectories[i + id_counter, changed_spin] *= -1
        ID_data_trajectory[i + id_counter] = ID_data_event[i]
        task_data_trajectory[i + id_counter] = task_data_event[i]
        time_difference = (
            event_step_time_spin[i + 1, 1] - event_step_time_spin[i, 1]
        )
        # if next event time i+1 is smaller than event time i, the step is final step
        if time_difference < 0.0:
            time_difference = final_time - event_step_time_spin[i, 1]
            dts[i + id_counter] = time_difference
            # setting next initial state
            trajectories[i + 1 + id_counter] = initial_states[id_counter]
            dts[i + 1 + id_counter] = event_step_time_spin[i + 1, 1] - 0.0
            ID_data_trajectory[i + 1 + id_counter] = ID_data_event[i + 1]
            task_data_trajectory[i + 1 + id_counter] = task_data_event[i + 1]
            id_counter += 1
        else:
            dts[i + id_counter] = time_difference
        progress_proxy.update(1)

    changed_spin = np.int64(event_step_time_spin[-1, 2])
    trajectories[-1] = trajectories[-2].copy()
    trajectories[-1, changed_spin] *= -1
    ID_data_trajectory[-1] = ID_data_trajectory[-2]
    task_data_trajectory[-1] = task_data_trajectory[-2]
    dts[-1] = final_time - event_step_time_spin[-1, 1]

    return trajectories, dts, ID_data_trajectory, task_data_trajectory


@njit(cache=True, nogil=True, parallel=True)
def _unique_probability(indices, dts):
    probability = np.bincount(indices, weights=dts) / dts.sum()
    return probability


@njit(cache=True, nogil=True)
def _entropy_production_rate(joint_transition, total_time):
    """Entropy production rate (k_B = 1, natural log) from transition counts.

    joint_transition[i, j] counts transitions i -> j observed over
    total_time. A transition whose reverse was never observed is assigned
    the reverse rate 1 / total_time.
    """
    n = joint_transition.shape[0]
    epr = 0.0
    for i in range(n):
        for j in range(n):
            forward = joint_transition[i, j]
            if forward == 0.0:
                continue
            backward = joint_transition[j, i]
            if backward != 0.0:
                log_ratio = np.log(forward / backward)
            else:
                log_ratio = np.log(forward * total_time)
            epr += (forward - backward) * log_ratio
    return 0.5 * epr / total_time


@njit(cache=True, nogil=True, parallel=True)
def _hypercubic_bootstrapping(
    transition_count,
    time_threshold,
    seeds,
    num_unique_states,
    num_dims,
    progress_proxy,
):
    """Resample observed transitions until their total dwell time reaches
    time_threshold; return the entropy production rate and the fraction of
    observed single-spin-flip transitions of each realization.

    Each realization seeds its own random stream, so the result does not
    depend on how the realizations are scheduled across threads.
    """
    num_realization = len(seeds)
    len_transitions = len(transition_count)
    entropy_production_rates = np.zeros(num_realization)
    fracs_observed_transitions = np.zeros(num_realization)

    for r in prange(num_realization):
        np.random.seed(seeds[r])
        joint_transition = np.zeros((num_unique_states, num_unique_states))
        total_time = 0.0
        while True:
            choice = np.random.randint(0, len_transitions)
            source = np.int64(transition_count[choice, 0])
            target = np.int64(transition_count[choice, 1])
            joint_transition[source, target] += 1
            total_time += transition_count[choice, 2]
            if total_time >= time_threshold:
                break

        entropy_production_rates[r] = _entropy_production_rate(
            joint_transition, total_time
        )
        num_observed = np.count_nonzero(joint_transition)
        fracs_observed_transitions[r] = num_observed / (num_dims * 2**num_dims)
        progress_proxy.update(1)
    return entropy_production_rates, fracs_observed_transitions


class WPCA:
    """PCA weighted by the probability of each observed state."""

    def fit(self, zscores, weights):
        """Fit the weighted principal components of zscores."""
        if weights is None:
            weights = np.ones(len(zscores))
        p = weights / weights.sum()
        cov = zscores.T * p @ zscores
        values, vectors = np.linalg.eigh(cov)
        values = values[::-1]
        vectors = vectors.T[::-1]
        vectors = vectors * (2 * (vectors.sum(axis=1, keepdims=True) >= 0) - 1)
        self.components_ = vectors
        self.explained_variance_ratio_ = values / values.sum()
        return self

    def transform(self, X):
        """Project X onto the principal components."""
        return X @ self.components_.T

    def fit_transform(self, X, weights):
        """Fit, then project X."""
        return self.fit(X, weights).transform(X)


class PFA:
    """Probability flux analysis of the HCP recordings.

    Built from the recordings (preprocessing, coarse_grain) or from saved
    group-level results (from_aggregates).
    """

    def __init__(self, file_path: str | Path | None = None):
        if file_path is None:
            file_path = DEFAULT_HCP_DATA_DIR
        self.file_path = file_path
        ## Length of time step in seconds
        self.time_between_steps = 0.72
        ## maximum number of time step
        # 176 is corresponding the time step of shortest task (emotional processing).
        self.num_time_steps = 176
        ## Component list of PCA
        self.components = [0, 1]
        ## number of discrete bins for flux analysis on pca
        self.num_bins = 10
        ## number of standard deviations from component center to cut off flux analysis
        self.num_std = 2
        ## name of tasks
        self.task_name_list = [
            "Rest",
            "Emotion",
            "Working memory",
            "Social",
            "Language",
            "Relational",
            "Gambling",
            "Motor",
        ]
        ## number of tasks
        self.num_tasks = len(self.task_name_list)

    # Preprocessing
    def load_data(self):
        """Load the 16 HCP .mat files (8 runs x 2 phase encodings)."""
        self.data = [
            io.loadmat(f"{self.file_path}/{run}_{phase}.mat")
            for run in HCP_RUNS
            for phase in ("LR", "RL")
        ]

    ########
    # Functions for reproducing Lynn et al. (2021) PNAS
    ########
    def get_X_data(self):
        """Parcel time series of every recording, truncated to num_time_steps."""
        X_data, ID_data, task_data, ID_unique, _ = _get_X_data(
            self.data, self.num_time_steps
        )
        self.X_data = X_data
        self.ID_data = ID_data
        self.task_data = task_data
        self.ID_unique = ID_unique

    def preprocessing(self):
        """Load the recordings and z-score every parcel."""
        print("Loading data...")
        self.load_data()
        print("Processing data...")
        self.get_X_data()
        # Z-score data
        self.zscore = stats.zscore(self.X_data)
        print("Done!")

    def coarse_grain(self, cluster_inds, order_derivative=0):
        """Average the parcels of each cluster into one z-scored signal
        (one spin per cluster) and binarize it into trajectories."""
        num_clusters = len(np.unique(cluster_inds))
        zscore = np.zeros((len(self.X_data), num_clusters))
        for c in range(num_clusters):
            cluster_mean = self.X_data[:, cluster_inds == c].mean(axis=1)
            zscore[:, c] = stats.zscore(cluster_mean)
        self.zscore = zscore
        self.zscore_to_trajectory(order_derivative)

    def perform_PCA(self):
        # pca
        """PCA of the parcel z scores (fig-s02)."""
        pca = PCA(svd_solver="full")
        pca.fit(self.zscore)
        vectors = pca.components_
        vectors = vectors * (2 * (vectors.sum(axis=1, keepdims=True) >= 0) - 1)
        pca.components_ = vectors
        self.X = pca.transform(self.zscore)
        self.pca = pca

    # Discretization
    def digitize_PCA_data(self):
        """Bin the scores of the PCs in `components` on a num_bins grid."""
        bin_edges = np.zeros([np.max(self.components) + 1, self.num_bins + 1])
        bin_pos = np.zeros([np.max(self.components) + 1, self.num_bins])
        X_bin = np.zeros(self.X.shape, int)
        for i in self.components:
            mean_comp = self.X[:, i].mean()
            std_comp = self.X[:, i].std()
            bins_start = mean_comp - self.num_std * std_comp
            bins_end = mean_comp + self.num_std * std_comp
            bin_edges[i, :] = np.linspace(
                bins_start, bins_end, self.num_bins + 1
            )
            bin_pos[i, :] = (bin_edges[i, :-1] + bin_edges[i, 1:]) / 2
            X_bin[:, i] = np.digitize(self.X[:, i], bin_edges[i, :])
        self.bin_edges = bin_edges
        self.bin_pos = bin_pos
        self.X_bin = X_bin

    # Picking task data
    def pick_task_data(self, task_choice):
        """Select the data of one task."""
        inds_task = np.where(self.task_data == task_choice)[0]
        if hasattr(self, "zscore"):
            self.zscore_task = self.zscore[inds_task]
            self.ID_data_task = self.ID_data[inds_task]
            self.ID_unique_task = np.unique(self.ID_data_task)
        if hasattr(self, "X"):
            self.X_task = self.X[inds_task]
        if hasattr(self, "X_bin"):
            self.X_bin_task = self.X_bin[inds_task]
        if hasattr(self, "trajectories"):
            inds_trajectories_task = np.where(
                self.task_data_trajectory == task_choice
            )[0]
            self.trajectories_task = self.trajectories[inds_trajectories_task]
            self.dts_task = self.dts[inds_trajectories_task]
            self.ID_data_trajectory_task = self.ID_data_trajectory[
                inds_trajectories_task
            ]

    # Flux on the PCA grid (Lynn et al.)
    def create_transition_list(self):
        """Count the transitions between neighboring PC grid bins of the current task."""
        bin_edges = self.bin_edges
        ID_data_task = self.ID_data_task
        ID_unique_task = self.ID_unique_task
        X_bin_task = self.X_bin_task
        X_task = self.X_task
        x_component = self.components[0]
        y_component = self.components[1]
        t_tot = self.time_between_steps * X_task.shape[0]

        num_bins = len(bin_edges[0]) - 1
        # state transition matrix
        transition = np.zeros([num_bins**2, num_bins**2])
        transition_list = np.zeros([len(X_bin_task) * num_bins, 5], int)

        tcount = 0

        for i in range(len(ID_unique_task)):
            inds_ID = np.where(ID_data_task == ID_unique_task[i])
            X_bin_temp = X_bin_task[inds_ID]
            X_temp = X_task[inds_ID]

            for t in range(len(X_bin_temp) - 1):
                source_x = X_bin_temp[t, x_component]
                source_y = X_bin_temp[t, y_component]
                target_x = X_bin_temp[t + 1, x_component]
                target_y = X_bin_temp[t + 1, y_component]
                if source_x == 0 or source_x == num_bins + 1:
                    continue
                else:
                    source_x -= 1
                if source_y == 0 or source_y == num_bins + 1:
                    continue
                else:
                    source_y -= 1
                if target_x == 0 or target_x == num_bins + 1:
                    continue
                else:
                    target_x -= 1
                if target_y == 0 or target_y == num_bins + 1:
                    continue
                else:
                    target_y -= 1
                gap_x = abs(target_x - source_x)
                gap_y = abs(target_y - source_y)
                # check if the transition is adjacent or stay
                if (gap_x + gap_y) == 1 or (gap_x + gap_y) == 0:
                    source = source_x + num_bins * source_y
                    target = target_x + num_bins * target_y
                    transition[source, target] += 1
                    transition_list[tcount, :] = [
                        source_x,
                        source_y,
                        target_x,
                        target_y,
                        1,
                    ]
                    tcount += 1
                # non-adjacent state transition
                # do linear interpolation
                else:
                    x_t = X_temp[t, x_component]
                    y_t = X_temp[t, y_component]
                    x_next = X_temp[t + 1, x_component]
                    y_next = X_temp[t + 1, y_component]

                    xvals = np.linspace(x_t, x_next, 2 * num_bins + 1)
                    f = interpolate.interp1d([x_t, x_next], [y_t, y_next])
                    yinterp = f(xvals)
                    x_id = np.digitize(xvals, bin_edges[x_component, :])
                    y_id = np.digitize(yinterp, bin_edges[y_component, :])
                    _, ind = np.unique(
                        np.dstack((x_id, y_id)), axis=1, return_index=True
                    )
                    unique_interp = np.dstack((x_id, y_id))[0][
                        np.sort(ind)
                    ].astype(int)

                    for ti in range(len(unique_interp) - 1):
                        source_x = unique_interp[ti, 0] - 1
                        source_y = unique_interp[ti, 1] - 1
                        target_x = unique_interp[ti + 1, 0] - 1
                        target_y = unique_interp[ti + 1, 1] - 1
                        source = source_x + num_bins * source_y
                        target = target_x + num_bins * target_y
                        transition[source, target] += 1
                        transition_list[tcount, :] = [
                            source_x,
                            source_y,
                            target_x,
                            target_y,
                            1,
                        ]
                        tcount += 1

        transition = transition - transition.T
        transition_list = transition_list[
            np.where(transition_list[:, 4] == 1)[0]
        ]

        flux = _grid_flux(transition, num_bins)
        u = 0.5 * flux[:, :, 0].flatten() / t_tot
        v = 0.5 * flux[:, :, 1].flatten() / t_tot
        self.t_tot = t_tot
        self.u = u
        self.v = v
        self.transition_list = transition_list

    def bootstrapping(self, seed=None, num_realization=100):
        """Bootstrap the PC grid flux of the current task."""
        rng = np.random.default_rng(seed)
        self.vec_strapped = _bootstrapping(
            self.transition_list,
            self.t_tot,
            self.num_bins,
            rng,
            num_realization=num_realization,
        )

    def calc_mean_vec_from_bootstrapping(self):
        """Mean of the bootstrapped grid flux."""
        vec_strapped = self.vec_strapped
        u_mean = []
        v_mean = []
        for t in range(vec_strapped.shape[1]):
            u_temp = vec_strapped[:, t, 0].mean()
            v_temp = vec_strapped[:, t, 1].mean()
            u_mean.append(u_temp)
            v_mean.append(v_temp)
        u_mean = np.array(u_mean)
        v_mean = np.array(v_mean)
        self.u_mean = u_mean
        self.v_mean = v_mean

    def ellipses_radii_angles(self):
        """Principal axes of the bootstrapped grid flux at each bin."""
        vec_strapped = self.vec_strapped
        angles = []
        stds = []
        num_states = vec_strapped.shape[1]
        for state in range(num_states):
            covariance = np.cov(vec_strapped[:, state].T)
            eigenvalues, vectors = np.linalg.eig(covariance)
            # numpy >= 2.5 returns complex dtype even for real spectra
            eigenvalues = np.real_if_close(eigenvalues)
            vectors = np.real_if_close(vectors)
            ## angle of PC0
            angle = np.arctan2(vectors[0, 1], vectors[0, 0])
            std0 = np.sqrt(eigenvalues[0])
            std1 = np.sqrt(eigenvalues[1])
            angles.append(angle)
            stds.append([std0, std1])
        angles = np.array(angles)
        stds = np.array(stds)
        self.angles = angles
        self.stds = stds

    def grid_summary(self):
        """Group-level results of the current task on the PC grid: the
        normalized 2-D histogram and the bootstrap-mean flux and ellipses."""
        x_component, y_component = self.components
        H, _, _ = np.histogram2d(
            self.X_task[:, x_component],
            self.X_task[:, y_component],
            bins=(self.bin_edges[x_component], self.bin_edges[y_component]),
        )
        H = H.T
        return {
            "hist": H / H.sum(),
            "u_mean": self.u_mean,
            "v_mean": self.v_mean,
            "angles": self.angles,
            "stds": self.stds,
        }

    def set_grid_summary(self, bin_edges, bin_pos, summary):
        """Restore a grid_summary (and the grid) for plotting."""
        self.bin_edges = bin_edges
        self.bin_pos = bin_pos
        self.hist_task = summary["hist"]
        self.u_mean = summary["u_mean"]
        self.v_mean = summary["v_mean"]
        self.angles = summary["angles"]
        self.stds = summary["stds"]

    # plotting
    def plot_probability_distribution(
        self,
        fig,
        ax,
        vmax=None,
        plot_cbar=True,
    ):
        """Draw the probability of each PC grid bin."""
        bin_edges = self.bin_edges
        x_component = self.components[0]
        y_component = self.components[1]
        if hasattr(self, "hist_task"):
            x_data = y_data = None
        else:
            X_plot = self.X_task if hasattr(self, "X_task") else self.X
            x_data = X_plot[:, x_component]
            y_data = X_plot[:, y_component]

        ax, cbar = normalized_hist2d(
            fig,
            ax,
            x_data,
            y_data,
            x_bins=bin_edges[x_component],
            y_bins=bin_edges[y_component],
            plot_cbar=plot_cbar,
            vmax=vmax,
            H=getattr(self, "hist_task", None),
        )
        ax = set_aspect_ratio(ax)
        return ax, cbar

    def draw_flux_scale(
        self,
        ax,
        unit=0.001,
        scale=SCALE,
        arrow_width=0.005,
        label_pos=LABEL_POS,
    ):
        """Draw the reference arrows of the PC grid flux."""
        bin_edges = self.bin_edges
        x_component = self.components[0]
        y_component = self.components[1]
        bin_size = np.diff(bin_edges)[:, 0]

        xy = (bin_edges[x_component, -3], bin_edges[y_component, -3])
        r = mpl.patches.Rectangle(
            xy=xy,
            width=2 * bin_size[x_component],
            height=2 * bin_size[y_component],
            color="gray",
            lw=0,
            zorder=3,
        )
        ax.add_patch(r)

        ax.quiver(
            1.05 * xy[0],
            1.05 * xy[1],
            unit,
            0,
            color="white",
            angles="xy",
            scale=scale,
            scale_units="xy",
            width=arrow_width,
            minlength=0,
            zorder=3,
        )
        ax.quiver(
            1.05 * xy[0],
            1.05 * xy[1],
            0,
            unit,
            color="white",
            angles="xy",
            scale=scale,
            scale_units="xy",
            width=arrow_width,
            minlength=0,
            zorder=3,
        )
        ax.text(
            label_pos[0] * xy[0],
            label_pos[1] * xy[1],
            f"${as_sci(unit, 0)}$",
            color="white",
            fontsize=6,
        )
        ax.text(
            label_pos[2] * xy[0],
            label_pos[3] * xy[1],
            "$\\mathrm{{s}}^{{-1}}$",
            color="white",
            fontsize=6,
        )
        return ax

    def plot_flux(
        self,
        fig,
        ax,
        vmax=None,
        unit=0.001,
        scale=SCALE,
        arrow_width=0.005,
        plot_cbar=True,
        label_pos=LABEL_POS,
    ):
        """Draw the PC grid flux over the probability distribution."""
        x_component = self.components[0]
        y_component = self.components[1]
        if hasattr(self, "u_mean"):
            u = self.u_mean
            v = self.v_mean
        else:
            u = self.u
            v = self.v
        x = self.bin_pos[x_component, :]
        y = self.bin_pos[y_component, :]
        xx, yy = np.meshgrid(x, y)
        self.xx = xx
        self.yy = yy

        ax, cbar = self.plot_probability_distribution(
            fig,
            ax,
            vmax=vmax,
            plot_cbar=plot_cbar,
        )
        ax.quiver(
            xx,
            yy,
            u,
            v,
            color="white",
            angles="xy",
            scale=scale,
            scale_units="xy",
            width=arrow_width,
            minlength=0,
            zorder=2,
        )
        ax = self.draw_flux_scale(
            ax,
            unit,
            scale,
            arrow_width,
            label_pos,
        )
        return ax, cbar

    def plot_confidence(self, ax, scale=SCALE):
        """Draw the bootstrap confidence ellipses of the PC grid flux."""
        xx = self.xx
        yy = self.yy
        u_mean = self.u_mean
        v_mean = self.v_mean
        angles = self.angles
        stds = self.stds

        num_states = len(angles)
        for i in range(num_states):
            draw_ellipses(
                ax,
                xx.flatten()[i],
                yy.flatten()[i],
                u_mean[i] / scale,
                v_mean[i] / scale,
                angles[i],
                2 * stds[i][0] / scale,
                2 * stds[i][1] / scale,
            )
        return ax

    def plot_flux_with_confidence(
        self,
        fig,
        ax,
        vmax=None,
        unit=0.001,
        scale=SCALE,
        arrow_width=0.005,
        plot_cbar=True,
        label_pos=LABEL_POS,
        seed=None,
        num_realization=100,
        bootstrap=True,
    ):
        """Grid flux with bootstrap confidence ellipses. With
        bootstrap=False the results set by set_grid_summary are drawn."""
        if bootstrap:
            self.bootstrapping(seed, num_realization)
            self.calc_mean_vec_from_bootstrapping()
            self.ellipses_radii_angles()
        ax, cbar = self.plot_flux(
            fig,
            ax,
            vmax=vmax,
            unit=unit,
            scale=scale,
            arrow_width=arrow_width,
            plot_cbar=plot_cbar,
            label_pos=label_pos,
        )
        ax = self.plot_confidence(ax, scale=scale)
        return ax, cbar

    ########
    # Functions to analyze the data in hypercubic manner
    ########
    def event_to_trajectory(self):
        """Build the binary state trajectories from the sorted events."""
        time_between_steps = self.time_between_steps
        num_time_steps = self.num_time_steps
        zscore = self.zscore
        event_step_time_spin = self.event_step_time_spin
        task_data_event = self.task_data_event
        ID_data_event = self.ID_data_event
        initial_states = self.initial_states

        print("Creating trajectories...")
        with ProgressBar(
            total=len(event_step_time_spin) - 1, dynamic_ncols=False
        ) as progress:
            trajectories, dts, ID_data_trajectory, task_data_trajectory = (
                _event_to_trajectory(
                    time_between_steps,
                    num_time_steps,
                    zscore,
                    event_step_time_spin,
                    task_data_event,
                    ID_data_event,
                    initial_states,
                    progress,
                )
            )
        self.trajectories = trajectories
        self.dts = dts
        self.ID_data_trajectory = ID_data_trajectory
        self.task_data_trajectory = task_data_trajectory
        print("Getting unique states from trajectories...")
        self.unique_states, indices = np.unique(
            trajectories, axis=0, return_inverse=True
        )
        print("Calculating probability...")
        self.probability = _unique_probability(indices, dts)

    def zscore_derivative_to_event(self, order_derivative):
        """Find the zeros of each spin's z-score derivative (state flips)."""
        event_step_time_spin = []
        ID_data_event = []
        task_data_event = []
        initial_states = []
        t = np.arange(self.num_time_steps) * self.time_between_steps

        print("Detecting the state transition by spline interpolation...")
        for task_choice in trange(self.num_tasks):
            self.pick_task_data(task_choice)
            zscore_task = self.zscore_task
            ID_data_task = self.ID_data_task
            ID_unique_task = self.ID_unique_task
            for j in range(len(ID_unique_task)):
                personal_id = ID_unique_task[j]
                inds = np.where(ID_data_task == personal_id)[0]
                zscore_task_personal = zscore_task[inds]
                initial_state = np.zeros(zscore_task_personal.shape[1])
                for i in range(zscore_task_personal.shape[1]):
                    zscore_task_personal_dim = zscore_task_personal[:, i]
                    cs = interpolate.CubicSpline(t, zscore_task_personal_dim)
                    derivative = cs.derivative(nu=order_derivative)
                    event_time_dim = derivative.roots()
                    event_time_dim = event_time_dim[
                        np.logical_and(
                            event_time_dim >= t[0], event_time_dim <= t[-1]
                        )
                    ]
                    event_step = np.searchsorted(t, event_time_dim) - 1
                    event_step_time_spin_np = np.vstack(
                        (
                            task_choice * j * self.num_time_steps + event_step,
                            event_time_dim,
                            i * np.ones_like(event_time_dim),
                        )
                    ).T
                    event_step_time_spin += event_step_time_spin_np.tolist()
                    ID_data_event += (
                        personal_id * np.ones_like(event_time_dim)
                    ).tolist()
                    task_data_event += (
                        task_choice * np.ones_like(event_time_dim)
                    ).tolist()
                    if order_derivative == 0:
                        initial_state[i] = np.sign(zscore_task_personal_dim[0])
                    else:
                        initial_state[i] = -np.sign(
                            cs(event_time_dim[1], order_derivative - 1)
                            - cs(event_time_dim[0], order_derivative - 1)
                        ) * (-1) ** (order_derivative - 1)
                initial_states.append(initial_state.tolist())
        event_step_time_spin = np.array(event_step_time_spin)
        ID_data_event = np.array(ID_data_event)
        task_data_event = np.array(task_data_event)
        initial_states = np.array(initial_states)
        self.event_step_time_spin = event_step_time_spin
        self.ID_data_event = ID_data_event
        self.task_data_event = task_data_event
        self.initial_states = initial_states

    def sort_zscore_derivative_event(self):
        """Sort the events of each recording in time."""
        event_step_time_spin = self.event_step_time_spin
        ID_data_event = self.ID_data_event
        ID_unique = self.ID_unique

        print("Sorting events...")
        with ProgressBar(
            total=len(ID_unique), dynamic_ncols=False
        ) as progress:
            event_step_time_spin = _sort_zscore_derivative_event(
                event_step_time_spin,
                ID_data_event,
                ID_unique,
                progress,
            )
        self.event_step_time_spin = event_step_time_spin

    def zscore_to_trajectory(self, order_derivative=0):
        """Binarize each spin at the zeros of its z-score's derivative of
        order `order_derivative` (0, 1 or 2), located by spline
        interpolation, and build the state trajectories."""
        if order_derivative > 2:
            raise ValueError("Discretization order should be 0, 1, or 2.")
        self.zscore_derivative_to_event(order_derivative=order_derivative)
        self.sort_zscore_derivative_event()
        self.event_to_trajectory()

    def trajectories_to_flux(
        self,
        task_choice,
        progress_message=True,
        heuristic_state_mapping=None,
    ):
        """Count the transitions of task `task_choice` and set its flux,
        traffic, probability and entropy production rate.

        A PFA built by from_aggregates uses the stored counts instead.
        """
        if hasattr(self, "_task_counts"):
            counts = self._task_counts[task_choice]
            n = len(counts["states"])
            joint_transition = dok_array((n, n))
            joint_transition[counts["source"], counts["target"]] = counts[
                "count"
            ]
            self.transition_count = None
            self._set_task_flux(
                joint_transition,
                counts["waiting_time"],
                counts["total_time"],
                counts["states"],
            )
            return

        # pick up task data
        if not hasattr(self, "trajectories"):
            self.zscore_to_trajectory()
        self.pick_task_data(task_choice)
        trajectories_task = self.trajectories_task
        num_dims = trajectories_task.shape[1]
        dts_task = self.dts_task
        ID_data_trajectory_task = self.ID_data_trajectory_task
        total_time = dts_task.sum()

        # Preprocessing of trajectories
        # Convert Ising binary state to binary state.
        if progress_message:
            print("Registering states...")
        trajectories_binary = (trajectories_task + 1) * 0.5
        trajectories_binary = trajectories_binary.astype(int)
        # Create binary id from state
        binary_codes = [
            "".join(str(digit) for digit in sequence)
            for sequence in trajectories_binary
        ]

        binary_code_to_id = {}
        id_to_binary_code = {}
        trajectories_id = []
        if heuristic_state_mapping is None:
            if num_dims >= 10:
                heuristic_state_mapping = True
            else:
                heuristic_state_mapping = False
        if heuristic_state_mapping:
            # register the unique id for observed states
            for code in binary_codes:
                if code not in binary_code_to_id:
                    new_id = len(binary_code_to_id)
                    binary_code_to_id[code] = new_id
                    id_to_binary_code[new_id] = code
                    trajectories_id.append(new_id)
                else:
                    trajectories_id.append(binary_code_to_id[code])
        else:
            # replace binary_codes with integers
            for i, code in enumerate(generate_binary_codes(num_dims)):
                binary_code_to_id[code] = i
                id_to_binary_code[i] = code
            for code in binary_codes:
                trajectories_id.append(binary_code_to_id[code])
        trajectories_id = np.array(trajectories_id).astype(int)

        # Create data storage
        # create joint transition rate matrix
        num_unique_states = len(id_to_binary_code)
        if progress_message:
            print(f"Total number of unique state is {num_unique_states}")
            print("Counting joint transition...")
        ## joint transition rate matrix
        joint_transition = dok_array((num_unique_states, num_unique_states))
        # transition trajectory for bootstrapping
        # 3D array [source, target, amount of time spend in source]
        transition_count = np.zeros((len(trajectories_id), 3))
        # time spend in a source state (will be probability if normalized)
        waiting_time = np.zeros(num_unique_states)
        # unique ID
        unique_ID_trajectory_task = np.unique(ID_data_trajectory_task)
        # data length of each unique data
        num_unique_data = len(unique_ID_trajectory_task)
        # Counting
        counter = 0
        for i in range(num_unique_data):
            ID_data = unique_ID_trajectory_task[i]
            inds_id = np.where(ID_data_trajectory_task == ID_data)[0]
            # pick unique data
            data = trajectories_id[inds_id]
            dts_data = dts_task[inds_id]
            ## count waiting time
            for t in range(len(data) - 1):
                ## count transition
                source = data[t]
                target = data[t + 1]
                ## if transition matrix of master equation is necessary,
                ## transpose the transition matrix
                joint_transition[source, target] += 1
                transition_count[counter] = [
                    source,
                    target,
                    dts_data[t],
                ]
                ## count waiting time
                waiting_time[source] += dts_data[t]
                # count the number of transitions
                counter += 1
            waiting_time[target] += dts_data[t + 1]
        self.transition_count = transition_count[:counter]
        states = binary_codes_to_ising(
            [id_to_binary_code[i] for i in range(num_unique_states)]
        )
        self._set_task_flux(joint_transition, waiting_time, total_time, states)
        if progress_message:
            print("All processes were completed!!")

    def _set_task_flux(
        self, joint_transition, waiting_time, total_time, states
    ):
        """Set the per-task results from the transition counts
        joint_transition[source, target] (scipy dok_array), the dwell time
        in each state, the total observation time, and the +/-1 spin vector
        of each state index."""
        num_unique_states, num_dims = states.shape
        flux_matrix = joint_transition - joint_transition.T
        traffic_matrix = joint_transition + joint_transition.T
        entropy_production_rate = _entropy_production_rate(
            joint_transition.toarray(), total_time
        )
        flux_matrix = flux_matrix / total_time
        traffic_matrix = traffic_matrix / total_time
        probability = waiting_time / total_time
        num_observed_transitions = joint_transition.count_nonzero()
        frac_observed_transitions = num_observed_transitions / (
            num_dims * 2**num_dims
        )

        ## (source, target, flux) of each transition with positive flux
        transition_list = np.array(find(flux_matrix)).T
        transition_list = transition_list[transition_list[:, 2] > 0]
        source_and_target_index = transition_list[:, :2].astype(int)
        flux = transition_list[:, 2]
        traffic_matrix = triu(traffic_matrix, format="dok")
        traffic_source_target_index = np.stack(
            traffic_matrix.nonzero(), axis=1
        )
        traffic = traffic_matrix[
            traffic_source_target_index[:, 0],
            traffic_source_target_index[:, 1],
        ].toarray()

        self.num_unique_states = num_unique_states
        self.joint_counts = joint_transition
        self.joint_transition = joint_transition.T
        self.waiting_time_task = waiting_time
        self.total_time_task = total_time

        self.flux_matrix = flux_matrix.T
        self.flux_source = states[source_and_target_index[:, 0]]
        self.flux_target = states[source_and_target_index[:, 1]]
        self.flux = flux

        self.traffic_matrix = traffic_matrix.T
        # traffic is undirected; source and target are swapped as stored
        self.traffic_source = states[traffic_source_target_index[:, 1]]
        self.traffic_target = states[traffic_source_target_index[:, 0]]
        self.traffic = traffic

        self.unique_states_task = states
        self.probability_task = probability
        self.entropy_production_rate = entropy_production_rate
        self.frac_observed_transitions = frac_observed_transitions

    def save_aggregates(self, group):
        """Write the group-level results of every task to an h5py group:
        the observed states and their probability, and per task the
        transition counts, dwell times and total observation time."""
        group.create_dataset(
            "unique_states", data=self.unique_states.astype(np.int8)
        )
        group.create_dataset("probability", data=self.probability)
        for k in range(self.num_tasks):
            self.trajectories_to_flux(k, progress_message=False)
            counts = self.joint_counts.tocoo()
            task = group.create_group(f"task{k}")
            task.create_dataset(
                "states", data=self.unique_states_task.astype(np.int8)
            )
            task.create_dataset("source", data=counts.row.astype(np.int32))
            task.create_dataset("target", data=counts.col.astype(np.int32))
            task.create_dataset("count", data=counts.data)
            task.create_dataset("waiting_time", data=self.waiting_time_task)
            task.attrs["total_time"] = self.total_time_task

    @classmethod
    def from_aggregates(cls, group):
        """A PFA restored from save_aggregates output (an h5py group): its
        trajectories_to_flux, perform_WPCA and plotting methods work without
        the HCP recordings."""
        pfa = cls()
        pfa.unique_states = group["unique_states"][()].astype(float)
        pfa.probability = group["probability"][()]
        pfa._task_counts = []
        for k in range(pfa.num_tasks):
            task = group[f"task{k}"]
            pfa._task_counts.append(
                {
                    "states": task["states"][()].astype(float),
                    "source": task["source"][()],
                    "target": task["target"][()],
                    "count": task["count"][()],
                    "waiting_time": task["waiting_time"][()],
                    "total_time": float(task.attrs["total_time"]),
                }
            )
        return pfa

    def transition_rate_matrix(self, total_time=None):
        """Conditional transition rates w[target, source] of the last task
        passed to trajectories_to_flux: joint rate / source probability.

        total_time defaults to the task's total observation time.
        """
        if total_time is None:
            total_time = self.total_time_task
        joint_rate = self.joint_transition.toarray() / total_time
        return joint_rate / (self.probability_task + 1e-8)

    def hypercubic_bootstrapping(self, num_realization=100, seed=None):
        """Bootstrap the entropy production rate and the fraction of
        observed transitions of the last task passed to
        trajectories_to_flux."""
        if getattr(self, "transition_count", None) is None:
            raise ValueError(
                "Needs the transitions of a task: run trajectories_to_flux on "
                "a PFA built from the recordings."
            )

        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**32 - 1, size=num_realization)
        with ProgressBar(
            total=num_realization, dynamic_ncols=False
        ) as progress:
            entropy_production_rates, fracs_observed_transitions = (
                _hypercubic_bootstrapping(
                    self.transition_count,
                    self.dts_task.sum(),
                    seeds,
                    self.num_unique_states,
                    self.trajectories.shape[1],
                    progress,
                )
            )
        self.bootstrapped = {
            "entropy_production_rates": entropy_production_rates,
            "fracs_observed_transitions": fracs_observed_transitions,
        }

    def perform_WPCA(self, spin_colors=None):
        """Probability-weighted PCA of the observed states.

        spin_colors (one color per spin) sets the arrow colors used by the
        biplot and the flux and traffic plots.
        """
        self.wpca = WPCA().fit(self.unique_states, self.probability)
        if spin_colors is not None:
            self.biplot_color = np.array(spin_colors)

    def plot_biplot_vector(
        self,
        color=None,
        cmap=plt.cm.turbo,
        width=0.04,
        x_component=0,
        y_component=1,
        fig=None,
        ax=None,
    ):
        """Draw each spin's loading on the two components as an arrow."""
        if not hasattr(self, "wpca"):
            self.perform_WPCA()
        eigenvectors = self.wpca.components_
        # multiply by 2 for Ising data
        biplot_vector = (
            np.stack(
                (eigenvectors[x_component], eigenvectors[y_component]), axis=1
            )
            * 2
        )
        angle = np.arctan2(biplot_vector[:, 1], biplot_vector[:, 0])

        if color is None:
            norm_angle = (angle - angle.min()) / (angle.max() - angle.min())
            self.biplot_color = cmap(norm_angle)
        else:
            self.biplot_color = np.array(color)

        if fig is None and ax is None:
            fig, ax = plt.subplots(constrained_layout=True)

        # to make suitable margins
        ax.scatter(0.0, 0.0, s=0, alpha=0, lw=0)
        ax.scatter(
            biplot_vector[:, 0], biplot_vector[:, 1], s=0, alpha=0, lw=0
        )

        arrows = []
        for i, (x, y) in enumerate(biplot_vector):
            arrows.append(mpl.patches.Arrow(0, 0, x, y, width=5 * width))
            ax.text(
                x,
                y,
                f"{i + 1}",
                rotation=np.rad2deg(angle[i]) - 90,
                rotation_mode="anchor",
                va="bottom",
                ha="center",
            )
        arrows = mpl.collections.PatchCollection(arrows)
        arrows.set_facecolor(self.biplot_color)
        arrows.set_edgecolor(None)
        ax.add_collection(arrows)
        ax.set_aspect("equal")
        ax.set_xlabel(f"PC{x_component + 1}")
        ax.set_ylabel(f"PC{y_component + 1}")
        return fig, ax

    def _plot_hypercubic_arrow_on_pca(
        self,
        arrow_type,
        scale=None,
        x_component=0,
        y_component=1,
        fig=None,
        ax=None,
    ):
        """Draw each flux (or traffic) as an arrow between its source and
        target states in WPCA space, with width proportional to its size
        and the color of the flipped spin."""
        if scale is None:
            scale = 1.0 / len(self.trajectories)
        if not hasattr(self, "biplot_color"):
            self.plot_biplot_vector()
        if fig is None and ax is None:
            fig, ax = plt.subplots(constrained_layout=True)

        X_ = self.wpca.transform(self.unique_states)
        ax.scatter(X_[:, x_component], X_[:, y_component], s=0, alpha=0, lw=0)
        ax.set_xlabel(f"PC{x_component + 1}")
        ax.set_ylabel(f"PC{y_component + 1}")
        ax.set_aspect("equal")
        if len(self.flux_source) == 0:
            return fig, ax

        if arrow_type == "flux":
            source = self.flux_source
            target = self.flux_target
            width = self.flux.copy()
            head_width_ratio = 3.0
            head_length_ratio = 0.2
        else:
            source = self.traffic_source
            target = self.traffic_target
            width = self.traffic.copy()
            head_width_ratio = 1.0
            head_length_ratio = 0.0
        change = np.argwhere(source * target == -1)[:, 1]
        vec_color = self.biplot_color[change]
        source_pos = self.wpca.transform(source)
        arrow_vec = self.wpca.transform(target) - source_pos
        width *= scale

        # widest arrows first
        indices = np.argsort(-width)
        arrows = []
        for x, y, u, v, w in zip(
            source_pos[indices, x_component],
            source_pos[indices, y_component],
            arrow_vec[indices, x_component],
            arrow_vec[indices, y_component],
            width[indices],
        ):
            arrows.append(
                mpl.patches.FancyArrow(
                    x,
                    y,
                    u,
                    v,
                    width=w,
                    length_includes_head=True,
                    head_width=head_width_ratio * w,
                    head_length=np.sqrt(u**2 + v**2) * head_length_ratio,
                )
            )
        arrows = mpl.collections.PatchCollection(arrows)
        arrows.set_facecolor(vec_color[indices])
        arrows.set_edgecolor(None)
        ax.add_collection(arrows)
        return fig, ax

    def plot_hypercubic_flux_on_pca(self, scale=None, **kwargs):
        """Draw the flux of each hypercube edge in WPCA space."""
        return self._plot_hypercubic_arrow_on_pca("flux", scale, **kwargs)

    def plot_hypercubic_traffic_on_pca(self, scale=None, **kwargs):
        """Draw the traffic of each hypercube edge in WPCA space."""
        return self._plot_hypercubic_arrow_on_pca("traffic", scale, **kwargs)
