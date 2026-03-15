from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

from PGAM.GAM_library import general_additive_model
from PGAM.gam_data_handlers import smooths_handler


ARENA_X_M = 3.0
ARENA_Y_M = 2.0
SPEED_THRESHOLD_MPS = 0.1
CYLINDER_RADIUS_M = ARENA_X_M / (2.0 * np.pi)
DEFAULT_DURATION_S = 1800.0
DEFAULT_FS = 10
SUMMARY_FIG = "simulate_social_place_cell_pgam_summary.png"


def wrap_x(x):
    return np.mod(x, ARENA_X_M)


def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def circular_dx(x1, x2):
    dx = x1 - x2
    return (dx + ARENA_X_M / 2.0) % ARENA_X_M - ARENA_X_M / 2.0


def reflect_y(y):
    bounces = 0
    while y < 0.0 or y > ARENA_Y_M:
        if y < 0.0:
            y = -y
            bounces += 1
        elif y > ARENA_Y_M:
            y = 2.0 * ARENA_Y_M - y
            bounces += 1
    return y, bounces


def normalize_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def circular_gaussian(x, y, center, sd, amplitude):
    dx = circular_dx(x, center[0])
    dy = y - center[1]
    return amplitude * np.exp(-0.5 * (dx / sd[0]) ** 2 - 0.5 * (dy / sd[1]) ** 2)


def unfolded_to_cylinder_xyz(pos):
    phi = 2.0 * np.pi * pos[:, 0] / ARENA_X_M
    xyz = np.zeros((pos.shape[0], 3), dtype=float)
    xyz[:, 0] = CYLINDER_RADIUS_M * np.cos(phi)
    xyz[:, 1] = CYLINDER_RADIUS_M * np.sin(phi)
    xyz[:, 2] = pos[:, 1]
    return xyz


def cylinder_frame(pos):
    phi = 2.0 * np.pi * pos[:, 0] / ARENA_X_M
    tangent = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(pos.shape[0])])
    vertical = np.tile(np.array([[0.0, 0.0, 1.0]]), (pos.shape[0], 1))
    inward = np.column_stack([-np.cos(phi), -np.sin(phi), np.zeros(pos.shape[0])])
    return tangent, vertical, inward


def maybe_show():
    if matplotlib.get_backend().lower() != "agg":
        plt.show(block=False)
        plt.pause(0.001)


def simulate_social_place_cell_data(seed=4, duration_s=DEFAULT_DURATION_S, fs=DEFAULT_FS):
    """
    Major step 1.
    Simulate trajectories, 3D heading, self-only spatial view, and Poisson spikes.
    Return every variable needed for fitting, statistics, and plotting.
    """

    dt = 1.0 / fs
    n = int(duration_s * fs)
    rng = np.random.default_rng(seed)

    def spatial_speed_field(x, y, speed_params):
        phase_x, phase_y, phase_mix, mean_speed, speed_amp = speed_params
        x_norm = x / ARENA_X_M
        y_norm = y / ARENA_Y_M
        field = (
            0.60 * np.sin(2.0 * np.pi * x_norm + phase_x)
            + 0.50 * np.cos(2.0 * np.pi * y_norm + phase_y)
            + 0.35 * np.sin(2.0 * np.pi * (x_norm + 0.7 * y_norm) + phase_mix)
        )
        field = np.clip(field / 1.45, -1.0, 1.0)
        return np.clip(mean_speed + speed_amp * field, 0.05, 0.145)

    def simulate_single_trajectory(local_seed, start, speed_params):
        local_rng = np.random.default_rng(local_seed)
        pos = np.zeros((n, 2), dtype=float)
        vel = np.zeros((n, 2), dtype=float)
        occupancy = np.zeros((5, 4), dtype=float)
        pos[0] = np.array(start, dtype=float)
        heading_2d = local_rng.uniform(-np.pi, np.pi)
        speed_state = spatial_speed_field(np.array([pos[0, 0]]), np.array([pos[0, 1]]), speed_params)[0]

        def pick_goal():
            flat = occupancy.ravel()
            pool = np.where(flat <= flat.min() + 0.5)[0]
            choice = pool[local_rng.integers(0, pool.size)]
            iy = choice % occupancy.shape[1]
            ix = choice // occupancy.shape[1]
            gx = (ix + local_rng.uniform(0.08, 0.92)) * (ARENA_X_M / occupancy.shape[0])
            gy = (iy + local_rng.uniform(0.08, 0.92)) * (ARENA_Y_M / occupancy.shape[1])
            return np.array([gx, gy], dtype=float)

        goal = pick_goal()
        for t in range(1, n):
            ix = min(occupancy.shape[0] - 1, int(pos[t - 1, 0] / ARENA_X_M * occupancy.shape[0]))
            iy = min(occupancy.shape[1] - 1, int(pos[t - 1, 1] / ARENA_Y_M * occupancy.shape[1]))
            occupancy[ix, iy] += 1.0

            goal_dx = circular_dx(np.array([goal[0]]), np.array([pos[t - 1, 0]]))[0]
            goal_dy = goal[1] - pos[t - 1, 1]
            if np.hypot(goal_dx, goal_dy) < 0.06 or t % 4 == 0:
                goal = pick_goal()
                goal_dx = circular_dx(np.array([goal[0]]), np.array([pos[t - 1, 0]]))[0]
                goal_dy = goal[1] - pos[t - 1, 1]

            goal_heading = np.arctan2(goal_dy, goal_dx)
            heading_2d = wrap_angle(0.05 * heading_2d + 0.95 * goal_heading + local_rng.normal(0.0, 0.12))
            target_speed = spatial_speed_field(np.array([pos[t - 1, 0]]), np.array([pos[t - 1, 1]]), speed_params)[0]
            speed_state = 0.94 * speed_state + 0.06 * target_speed + local_rng.normal(0.0, 0.0025)
            speed_state = np.clip(speed_state, 0.04, 0.16)

            dx = speed_state * dt * np.cos(heading_2d)
            dy = speed_state * dt * np.sin(heading_2d)
            new_x = wrap_x(pos[t - 1, 0] + dx)
            new_y, n_bounces = reflect_y(pos[t - 1, 1] + dy)
            if n_bounces % 2 == 1:
                heading_2d = wrap_angle(-heading_2d)

            pos[t] = np.array([new_x, new_y])
            vel[t, 0] = circular_dx(np.array([pos[t, 0]]), np.array([pos[t - 1, 0]]))[0] / dt
            vel[t, 1] = (pos[t, 1] - pos[t - 1, 1]) / dt
            ix = min(occupancy.shape[0] - 1, int(pos[t, 0] / ARENA_X_M * occupancy.shape[0]))
            iy = min(occupancy.shape[1] - 1, int(pos[t, 1] / ARENA_Y_M * occupancy.shape[1]))
            occupancy[ix, iy] += 1.0

        pos3d = unfolded_to_cylinder_xyz(pos)
        vel3d = np.zeros_like(pos3d)
        vel3d[1:] = np.diff(pos3d, axis=0) / dt
        speed = np.linalg.norm(vel3d, axis=1)

        tangent, vertical, inward = cylinder_frame(pos)
        head_dir3d = np.zeros_like(vel3d)
        head_dir3d[0] = normalize_rows(0.85 * tangent[:1] + 0.55 * inward[:1])[0]
        for t in range(1, n):
            if speed[t] > SPEED_THRESHOLD_MPS:
                move_dir = vel3d[t] / max(speed[t], 1e-12)
                candidate = move_dir + 0.22 * inward[t]
            else:
                base_dir = normalize_rows(0.85 * tangent[t : t + 1] + 0.55 * inward[t : t + 1])[0]
                candidate = 0.96 * head_dir3d[t - 1] + 0.04 * base_dir
            head_dir3d[t] = candidate / max(np.linalg.norm(candidate), 1e-12)

        heading = np.arctan2(np.sum(head_dir3d * vertical, axis=1), np.sum(head_dir3d * tangent, axis=1))
        origin_xy = pos3d[:, :2]
        dir_xy = head_dir3d[:, :2]
        ray_t = np.maximum(
            -2.0 * np.sum(origin_xy * dir_xy, axis=1) / np.maximum(np.sum(dir_xy**2, axis=1), 1e-12),
            1e-6,
        )
        hit_xy = origin_xy + ray_t[:, None] * dir_xy
        hit_phi = np.mod(np.arctan2(hit_xy[:, 1], hit_xy[:, 0]), 2.0 * np.pi)
        view_pos = np.column_stack(
            [
                ARENA_X_M * hit_phi / (2.0 * np.pi),
                np.clip(pos3d[:, 2] + ray_t * head_dir3d[:, 2], 0.0, ARENA_Y_M),
            ]
        )

        return {
            "pos": pos,
            "vel": vel,
            "pos3d": pos3d,
            "vel3d": vel3d,
            "speed": speed,
            "heading": heading,
            "head_dir3d": head_dir3d,
            "view_pos": view_pos,
            "fast_mask": speed > SPEED_THRESHOLD_MPS,
            "fast_fraction": float(np.mean(speed > SPEED_THRESHOLD_MPS)),
            "coverage_fraction": float(np.mean(occupancy > 0)),
        }

    def generate_constrained_trajectory(base_seed, start, speed_params):
        best = None
        best_score = np.inf
        for attempt in range(24):
            traj = simulate_single_trajectory(base_seed + attempt, start, speed_params)
            score = 3.0 * abs(traj["fast_fraction"] - 0.80) + 2.0 * max(0.0, 0.95 - traj["coverage_fraction"])
            if score < best_score:
                best = traj
                best_score = score
            if 0.74 <= traj["fast_fraction"] <= 0.86 and traj["coverage_fraction"] >= 0.95:
                return traj
        return best

    self_traj = generate_constrained_trajectory(11, (0.3, 0.4), (0.25, 1.10, -0.55, 0.120, 0.024))
    other_traj = generate_constrained_trajectory(31, (2.2, 1.5), (0.90, -0.60, 0.20, 0.116, 0.025))

    move_fast = self_traj["fast_mask"].astype(float)
    f_self = circular_gaussian(self_traj["pos"][:, 0], self_traj["pos"][:, 1], (0.55, 1.42), (0.30, 0.26), 1.75)
    f_other = circular_gaussian(other_traj["pos"][:, 0], other_traj["pos"][:, 1], (2.12, 0.55), (0.34, 0.24), 1.15)
    f_theta = 0.08 * np.cos(wrap_angle(self_traj["heading"] - 0.65))
    f_view = circular_gaussian(self_traj["view_pos"][:, 0], self_traj["view_pos"][:, 1], (2.35, 1.15), (0.30, 0.22), 0.40)
    log_rate_hz = np.log(1.2) + 0.20 * move_fast + move_fast * f_self + f_other + f_theta + f_view
    rate_hz = np.exp(log_rate_hz)
    y = rng.poisson(rate_hz * dt)

    return {
        "seed": seed,
        "duration_s": duration_s,
        "fs": fs,
        "dt": dt,
        "n": n,
        "self_traj": self_traj,
        "other_traj": other_traj,
        "move_fast": move_fast,
        "y": y,
        "rate_hz": rate_hz,
    }


def fit_social_place_cell_pgam(sim_data):
    """
    Major step 2.
    Take the simulated dataset and fit the PGAM model.
    """

    y = sim_data["y"]
    self_traj = sim_data["self_traj"]
    other_traj = sim_data["other_traj"]

    move_fast = self_traj["fast_mask"].astype(float)
    self_x_fast = self_traj["pos"][:, 0].copy()
    self_y_fast = self_traj["pos"][:, 1].copy()
    self_x_fast[move_fast < 0.5] = np.nan
    self_y_fast[move_fast < 0.5] = np.nan

    def make_test_mask():
        rng = np.random.default_rng(7)
        nbx, nby = 6, 4
        sx = np.clip((self_traj["pos"][:, 0] / ARENA_X_M * nbx).astype(int), 0, nbx - 1)
        sy = np.clip((self_traj["pos"][:, 1] / ARENA_Y_M * nby).astype(int), 0, nby - 1)
        ox = np.clip((other_traj["pos"][:, 0] / ARENA_X_M * nbx).astype(int), 0, nbx - 1)
        oy = np.clip((other_traj["pos"][:, 1] / ARENA_Y_M * nby).astype(int), 0, nby - 1)
        ids = sx + nbx * sy + nbx * nby * ox + nbx * nby * nbx * oy + (nbx * nby * nbx * nby) * move_fast.astype(int)
        test_mask = np.zeros(self_traj["pos"].shape[0], dtype=bool)
        for bin_id in np.unique(ids):
            idx = np.where(ids == bin_id)[0]
            if idx.size <= 1:
                continue
            n_test = min(max(1, int(round(0.2 * idx.size))), idx.size - 1)
            test_mask[rng.choice(idx, size=n_test, replace=False)] = True
        return test_mask

    test_mask = make_test_mask()
    train_mask = ~test_mask

    handler = smooths_handler()
    handler.add_smooth(
        "move_fast",
        [move_fast],
        knots=[np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=float)],
        ord=4,
        penalty_type="der",
        der=1,
        is_cyclic=[False],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "self_xy_fast",
        [self_x_fast, self_y_fast],
        knots_num=7,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True, False],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "theta",
        [self_traj["heading"]],
        knots_num=9,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "other_xy",
        [other_traj["pos"][:, 0], other_traj["pos"][:, 1]],
        knots_num=7,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True, False],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "view_xy",
        [self_traj["view_pos"][:, 0], self_traj["view_pos"][:, 1]],
        knots_num=7,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True, False],
        knots_percentiles=(0, 100),
    )

    family = sm.genmod.families.family.Poisson(link=sm.genmod.families.links.log())
    gam = general_additive_model(handler, handler.smooths_var, y, family)
    fit_start_s = perf_counter()
    model, reduced_model = gam.fit_full_and_reduced(
        handler.smooths_var,
        th_pval=0.01,
        max_iter=80,
        use_dgcv=True,
        filter_trials=train_mask,
        fit_initial_beta=True,
    )
    fit_elapsed_s = perf_counter() - fit_start_s

    x_lookup = {
        "move_fast": [move_fast],
        "self_xy_fast": [self_x_fast, self_y_fast],
        "theta": [self_traj["heading"]],
        "other_xy": [other_traj["pos"][:, 0], other_traj["pos"][:, 1]],
        "view_xy": [self_traj["view_pos"][:, 0], self_traj["view_pos"][:, 1]],
    }
    mu_all = np.clip(model.predict([x_lookup[var] for var in model.var_list], var_list=model.var_list), 1e-12, None)

    return {
        "model": model,
        "reduced_model": reduced_model,
        "fit_elapsed_s": fit_elapsed_s,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "mu_all": mu_all,
    }


def compute_social_place_cell_statistics(sim_data, fit_data):
    """
    Major step 3.
    Compute held-out fit metrics, map summaries, and tuning statistics.
    """

    y = sim_data["y"]
    dt = sim_data["dt"]
    self_traj = sim_data["self_traj"]
    other_traj = sim_data["other_traj"]
    mu_all = fit_data["mu_all"]
    test_mask = fit_data["test_mask"]
    model = fit_data["model"]

    def smooth_map(x, y_pos, weights, mask=None, x_bins=18, y_bins=12):
        if mask is None:
            mask = np.ones(x.shape[0], dtype=bool)
        x_edges = np.linspace(0.0, ARENA_X_M, x_bins + 1)
        y_edges = np.linspace(0.0, ARENA_Y_M, y_bins + 1)
        occ, _, _ = np.histogram2d(x[mask], y_pos[mask], bins=[x_edges, y_edges])
        val, _, _ = np.histogram2d(x[mask], y_pos[mask], bins=[x_edges, y_edges], weights=weights[mask])
        occ = gaussian_filter(occ, sigma=1.0, mode=("wrap", "reflect"))
        val = gaussian_filter(val, sigma=1.0, mode=("wrap", "reflect"))
        rate = val / np.maximum(occ * dt, 1e-12)
        rate[occ < 1.0] = np.nan
        return rate, x_edges, y_edges

    def smooth_theta_curve(theta, weights, n_bins=18):
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        occ, _ = np.histogram(theta, bins=edges)
        val, _ = np.histogram(theta, bins=edges, weights=weights)
        occ = gaussian_filter1d(occ.astype(float), sigma=1.0, mode="wrap")
        val = gaussian_filter1d(val.astype(float), sigma=1.0, mode="wrap")
        centers = wrap_angle((edges[:-1] + edges[1:]) / 2.0)
        return centers, val / np.maximum(occ * dt, 1e-12)

    def binned_rate_traces(observed_counts, predicted_counts, bin_sec=1.0):
        bin_size = max(1, int(round(bin_sec / dt)))
        n_bins = observed_counts.shape[0] // bin_size
        keep = n_bins * bin_size
        obs = observed_counts[:keep].reshape(n_bins, bin_size).sum(axis=1) / (bin_size * dt)
        pred = predicted_counts[:keep].reshape(n_bins, bin_size).sum(axis=1) / (bin_size * dt)
        t = (np.arange(n_bins) + 0.5) * bin_size * dt
        return t, obs, pred

    def poisson_loglik(obs, mu):
        mu = np.clip(mu, 1e-12, None)
        return float(np.sum(obs * np.log(mu) - mu))

    y_test = y[test_mask]
    mu_test = mu_all[test_mask]
    mu_null = np.full(y_test.shape[0], max(y[~test_mask].mean(), 1e-12), dtype=float)
    heldout_pseudo_r2 = 1.0 - poisson_loglik(y_test, mu_test) / poisson_loglik(y_test, mu_null)
    cov_pvals = {row["covariate"]: row["p-val"] for row in model.covariate_significance}

    self_map_obs, self_x_edges, self_y_edges = smooth_map(
        self_traj["pos"][:, 0], self_traj["pos"][:, 1], y, mask=self_traj["fast_mask"]
    )
    self_map_pred, _, _ = smooth_map(
        self_traj["pos"][:, 0], self_traj["pos"][:, 1], mu_all, mask=self_traj["fast_mask"]
    )
    other_map_obs, other_x_edges, other_y_edges = smooth_map(other_traj["pos"][:, 0], other_traj["pos"][:, 1], y)
    other_map_pred, _, _ = smooth_map(other_traj["pos"][:, 0], other_traj["pos"][:, 1], mu_all)
    view_map_obs, view_x_edges, view_y_edges = smooth_map(
        self_traj["view_pos"][:, 0], self_traj["view_pos"][:, 1], y
    )
    theta_centers, theta_obs = smooth_theta_curve(self_traj["heading"], y)
    _, theta_pred = smooth_theta_curve(self_traj["heading"], mu_all)
    t_axis_rate, observed_rate_hz, predicted_rate_hz = binned_rate_traces(y, mu_all)
    rho_s, p_s = spearmanr(observed_rate_hz, predicted_rate_hz)
    rho_p, p_p = pearsonr(observed_rate_hz, predicted_rate_hz)

    return {
        "heldout_pseudo_r2": heldout_pseudo_r2,
        "cov_pvals": cov_pvals,
        "self_map_obs": self_map_obs,
        "self_map_pred": self_map_pred,
        "self_x_edges": self_x_edges,
        "self_y_edges": self_y_edges,
        "other_map_obs": other_map_obs,
        "other_map_pred": other_map_pred,
        "other_x_edges": other_x_edges,
        "other_y_edges": other_y_edges,
        "view_map_obs": view_map_obs,
        "view_x_edges": view_x_edges,
        "view_y_edges": view_y_edges,
        "theta_centers": theta_centers,
        "theta_obs": theta_obs,
        "theta_pred": theta_pred,
        "t_axis_rate": t_axis_rate,
        "observed_rate_hz": observed_rate_hz,
        "predicted_rate_hz": predicted_rate_hz,
        "rho_s": rho_s,
        "p_s": p_s,
        "rho_p": rho_p,
        "p_p": p_p,
    }


def plot_social_place_cell_results(sim_data, fit_data, stats, out_path=SUMMARY_FIG, segment_s=120.0):
    """
    Major step 4.
    Make the main summary figure and the 120-second speed-segment figure.
    """

    self_traj = sim_data["self_traj"]
    other_traj = sim_data["other_traj"]
    y = sim_data["y"]
    dt = sim_data["dt"]
    n = sim_data["n"]
    fs = sim_data["fs"]
    cov_pvals = stats["cov_pvals"]

    def plot_wrapped_trajectory(ax, pos, color="0.80", lw=0.55):
        jumps = np.abs(np.diff(pos[:, 0])) > (ARENA_X_M / 2.0)
        start = 0
        for idx in np.where(jumps)[0]:
            stop = idx + 1
            if stop - start >= 2:
                ax.plot(pos[start:stop, 0], pos[start:stop, 1], color=color, lw=lw)
            start = stop
        if pos.shape[0] - start >= 2:
            ax.plot(pos[start:, 0], pos[start:, 1], color=color, lw=lw)

    def add_map_panel(ax, rate_map, x_edges, y_edges, title, xlabel, ylabel, cmap_name):
        vmax = float(np.nanmax(rate_map)) if np.any(np.isfinite(rate_map)) else 1.0
        vmax = max(vmax, 1e-6)
        ax.imshow(
            rate_map.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
            cmap=cmap_name,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.98,
            0.98,
            f"max={vmax:.2f} Hz",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], height_ratios=[1.0, 1.0, 0.75])
    cmap = "jet"

    ax = fig.add_subplot(gs[0, 0])
    plot_wrapped_trajectory(ax, self_traj["pos"])
    ax.scatter(self_traj["pos"][y > 0, 0], self_traj["pos"][y > 0, 1], s=7, color="#d62728", alpha=0.72)
    ax.set_title("Self trajectory and spikes")
    ax.set_xlabel("Self x (m)")
    ax.set_ylabel("Self y (m)")
    ax.set_xlim(0.0, ARENA_X_M)
    ax.set_ylim(0.0, ARENA_Y_M)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[0, 1])
    add_map_panel(ax, stats["self_map_obs"], stats["self_x_edges"], stats["self_y_edges"], "Observed self map | fast only", "Self x (m)", "Self y (m)", cmap)

    ax = fig.add_subplot(gs[0, 2])
    add_map_panel(
        ax,
        stats["self_map_pred"],
        stats["self_x_edges"],
        stats["self_y_edges"],
        f"Model self map | fast only\np={cov_pvals.get('self_xy_fast', np.nan):.1e}",
        "Self x (m)",
        "Self y (m)",
        cmap,
    )

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(np.arange(n) * dt, self_traj["speed"], color="#1f77b4", lw=0.8, label="Self speed")
    ax.plot(np.arange(n) * dt, other_traj["speed"], color="#2ca02c", lw=0.8, alpha=0.8, label="Other speed")
    ax.axhline(SPEED_THRESHOLD_MPS, color="black", lw=1.0, ls="--", label="0.1 m/s")
    ax.set_title("Speed traces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    plot_wrapped_trajectory(ax, other_traj["pos"])
    ax.scatter(other_traj["pos"][y > 0, 0], other_traj["pos"][y > 0, 1], s=7, color="#d62728", alpha=0.72)
    ax.set_title("Other trajectory and spikes")
    ax.set_xlabel("Other x (m)")
    ax.set_ylabel("Other y (m)")
    ax.set_xlim(0.0, ARENA_X_M)
    ax.set_ylim(0.0, ARENA_Y_M)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[1, 1])
    add_map_panel(ax, stats["other_map_obs"], stats["other_x_edges"], stats["other_y_edges"], "Observed other map", "Other x (m)", "Other y (m)", cmap)

    ax = fig.add_subplot(gs[1, 2])
    add_map_panel(
        ax,
        stats["other_map_pred"],
        stats["other_x_edges"],
        stats["other_y_edges"],
        f"Model other map\np={cov_pvals.get('other_xy', np.nan):.1e}",
        "Other x (m)",
        "Other y (m)",
        cmap,
    )

    ax = fig.add_subplot(gs[1, 3], projection="polar")
    ax.plot(np.r_[stats["theta_centers"], stats["theta_centers"][0]], np.r_[stats["theta_obs"], stats["theta_obs"][0]], color="0.45", lw=1.4, label="Observed")
    ax.plot(np.r_[stats["theta_centers"], stats["theta_centers"][0]], np.r_[stats["theta_pred"], stats["theta_pred"][0]], color="#d62728", lw=1.4, label="Model")
    ax.set_title(f"Theta tuning\np={cov_pvals.get('theta', np.nan):.1e}", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15), fontsize=8, frameon=True)

    ax = fig.add_subplot(gs[2, 0])
    add_map_panel(
        ax,
        stats["view_map_obs"],
        stats["view_x_edges"],
        stats["view_y_edges"],
        f"Observed spatial view map\np={cov_pvals.get('view_xy', np.nan):.1e}",
        "View x (m)",
        "View y (m)",
        cmap,
    )

    ax = fig.add_subplot(gs[2, 1:4])
    ax.plot(stats["t_axis_rate"], stats["observed_rate_hz"], color="0.45", lw=1.0, label="Observed")
    ax.plot(stats["t_axis_rate"], stats["predicted_rate_hz"], color="#d62728", lw=1.0, alpha=0.85, label="PGAM")
    ax.text(
        0.01,
        0.98,
        "\n".join(
            [
                f"Spearman r={stats['rho_s']:.3f}, p={stats['p_s']:.2e}",
                f"Pearson r={stats['rho_p']:.3f}, p={stats['p_p']:.2e}",
                f"Held-out pseudo-R2={stats['heldout_pseudo_r2']:.3f}",
                f"self fast={self_traj['fast_fraction']:.2f}, other fast={other_traj['fast_fraction']:.2f}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )
    ax.set_title("Observed vs predicted firing rate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (Hz)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Social place cell simulation with speed-gated self tuning and PGAM recovery", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    maybe_show()

    n_seg = min(int(round(segment_s * fs)), n)
    start = max(0, n // 2 - n_seg // 2)
    stop = start + n_seg
    t_seg = np.arange(start, stop) * dt
    speed_segment_path = str(Path(out_path).with_name(Path(out_path).stem + "_speed_segment.png"))

    fig_seg, ax_seg = plt.subplots(figsize=(12, 4.2))
    ax_seg.plot(t_seg, self_traj["speed"][start:stop], color="#1f77b4", lw=1.0, label="Self speed")
    ax_seg.plot(t_seg, other_traj["speed"][start:stop], color="#2ca02c", lw=1.0, alpha=0.9, label="Other speed")
    ax_seg.axhline(SPEED_THRESHOLD_MPS, color="black", lw=1.0, ls="--", label="0.1 m/s")
    ax_seg.set_title("Speed traces over 120 s segment")
    ax_seg.set_xlabel("Time (s)")
    ax_seg.set_ylabel("Speed (m/s)")
    ax_seg.legend(loc="upper right", fontsize=9, frameon=True)
    ax_seg.spines["top"].set_visible(False)
    ax_seg.spines["right"].set_visible(False)
    fig_seg.tight_layout()
    fig_seg.savefig(speed_segment_path, dpi=180)

    return {"summary_path": out_path, "speed_segment_path": speed_segment_path}


def main():
    sim_data = simulate_social_place_cell_data()
    fit_data = fit_social_place_cell_pgam(sim_data)
    stats = compute_social_place_cell_statistics(sim_data, fit_data)
    plot_paths = plot_social_place_cell_results(sim_data, fit_data, stats, out_path=SUMMARY_FIG)

    print("Reduced model kept:", list(fit_data["reduced_model"].var_list) if fit_data["reduced_model"] is not None else list(fit_data["model"].var_list))
    print(f"Self fast fraction: {sim_data['self_traj']['fast_fraction']:.3f}")
    print(f"Other fast fraction: {sim_data['other_traj']['fast_fraction']:.3f}")
    print(f"Self coverage fraction: {sim_data['self_traj']['coverage_fraction']:.3f}")
    print(f"Other coverage fraction: {sim_data['other_traj']['coverage_fraction']:.3f}")
    print(f"Train pseudo-R2: {fit_data['model'].pseudo_r2:.3f}")
    print(f"Held-out pseudo-R2: {stats['heldout_pseudo_r2']:.3f}")
    print(f"PGAM fit time: {fit_data['fit_elapsed_s']:.2f} s")
    for key in ["move_fast", "self_xy_fast", "theta", "other_xy", "view_xy"]:
        print(f"{key} p-value: {stats['cov_pvals'].get(key, np.nan):.3e}")
    print(f"Saved: {plot_paths['summary_path']}")
    print(f"Saved: {plot_paths['speed_segment_path']}")


if __name__ == "__main__":
    main()
