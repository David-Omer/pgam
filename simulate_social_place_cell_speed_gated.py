import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

PGAM_SRC = Path("/Users/davidomer/Applications/pgam/src")
if PGAM_SRC.exists():
    sys.path.append(str(PGAM_SRC))

from PGAM.GAM_library import general_additive_model
from PGAM.gam_data_handlers import smooths_handler


def run_social_place_cell_speed_gated_simulation(
    seed: int = 4,
    duration_s: float = 1800.0,
    fs: int = 10,
    out_path: str = "simulate_social_place_cell_pgam_summary.png",
):
    """
    Run the full social-place-cell simulation, fit the PGAM, and save a summary figure.

    The environment is an unfolded cylindrical cage wall:
    - unfolded x is circular and spans 3 m
    - unfolded y spans wall height and spans 2 m
    - spatial view is defined only by the self animal's 3D head direction
      intersecting the 3D cage wall, then mapped back to unfolded wall x/y
    """

    # Step 1: define the arena, timing, and simulation targets.
    arena_x_m = 3.0
    arena_y_m = 2.0
    speed_threshold_mps = 0.1
    cylinder_radius_m = arena_x_m / (2.0 * np.pi)
    dt = 1.0 / fs
    n = int(duration_s * fs)

    # Step 2: define local helper functions so the full simulation stays in one place.
    def maybe_show():
        if matplotlib.get_backend().lower() != "agg":
            plt.show(block=False)
            plt.pause(0.001)

    def wrap_x(x):
        return np.mod(x, arena_x_m)

    def wrap_angle(theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi

    def circular_dx(x1, x2):
        dx = x1 - x2
        return (dx + arena_x_m / 2.0) % arena_x_m - arena_x_m / 2.0

    def reflect_y(y):
        bounces = 0
        while y < 0.0 or y > arena_y_m:
            if y < 0.0:
                y = -y
                bounces += 1
            elif y > arena_y_m:
                y = 2.0 * arena_y_m - y
                bounces += 1
        return y, bounces

    def normalize_rows(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def circular_gaussian(x, y, center, sd, amplitude):
        dx = circular_dx(x, center[0])
        dy = y - center[1]
        return amplitude * np.exp(-0.5 * (dx / sd[0]) ** 2 - 0.5 * (dy / sd[1]) ** 2)

    def unfolded_to_cylinder_xyz(pos):
        phi = 2.0 * np.pi * pos[:, 0] / arena_x_m
        xyz = np.zeros((pos.shape[0], 3), dtype=float)
        xyz[:, 0] = cylinder_radius_m * np.cos(phi)
        xyz[:, 1] = cylinder_radius_m * np.sin(phi)
        xyz[:, 2] = pos[:, 1]
        return xyz

    def cylinder_frame(pos):
        phi = 2.0 * np.pi * pos[:, 0] / arena_x_m
        tangent = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(pos.shape[0])])
        vertical = np.tile(np.array([[0.0, 0.0, 1.0]]), (pos.shape[0], 1))
        inward = np.column_stack([-np.cos(phi), -np.sin(phi), np.zeros(pos.shape[0])])
        return tangent, vertical, inward

    def smooth_map(x, y, weights, mask=None, x_bins=18, y_bins=12):
        if mask is None:
            mask = np.ones(x.shape[0], dtype=bool)
        x_edges = np.linspace(0.0, arena_x_m, x_bins + 1)
        y_edges = np.linspace(0.0, arena_y_m, y_bins + 1)
        occ, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
        val, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges], weights=weights[mask])
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

    def poisson_loglik(y, mu):
        mu = np.clip(mu, 1e-12, None)
        return float(np.sum(y * np.log(mu) - mu))

    def masked_self_xy(traj):
        move_fast = traj["fast_mask"].astype(float)
        x_fast = traj["pos"][:, 0].copy()
        y_fast = traj["pos"][:, 1].copy()
        x_fast[move_fast < 0.5] = np.nan
        y_fast[move_fast < 0.5] = np.nan
        return move_fast, x_fast, y_fast

    def plot_wrapped_trajectory(ax, pos, color="0.80", lw=0.55):
        jumps = np.abs(np.diff(pos[:, 0])) > (arena_x_m / 2.0)
        start = 0
        for idx in np.where(jumps)[0]:
            stop = idx + 1
            if stop - start >= 2:
                ax.plot(pos[start:stop, 0], pos[start:stop, 1], color=color, lw=lw)
            start = stop
        if pos.shape[0] - start >= 2:
            ax.plot(pos[start:, 0], pos[start:, 1], color=color, lw=lw)

    def add_map_panel(ax, rate_map, x_edges, y_edges, title, xlabel, ylabel, cmap_name):
        im = ax.imshow(
            rate_map.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
            cmap=cmap_name,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hz")

    def print_summary(model, reduced_model, self_traj, other_traj, heldout_pseudo_r2, cov_pvals):
        print("Reduced model kept:", list(reduced_model.var_list) if reduced_model is not None else list(model.var_list))
        print(f"Self fast fraction: {self_traj['fast_fraction']:.3f}")
        print(f"Other fast fraction: {other_traj['fast_fraction']:.3f}")
        print(f"Self coverage fraction: {self_traj['coverage_fraction']:.3f}")
        print(f"Other coverage fraction: {other_traj['coverage_fraction']:.3f}")
        print(f"Train pseudo-R2: {model.pseudo_r2:.3f}")
        print(f"Held-out pseudo-R2: {heldout_pseudo_r2:.3f}")
        for key in ["move_fast", "self_xy_fast", "theta", "other_xy", "view_xy"]:
            print(f"{key} p-value: {cov_pvals.get(key, np.nan):.3e}")

    # Step 3: define the smooth spatial speed field that drives the animals.
    def spatial_speed_field(x, y, speed_params):
        phase_x, phase_y, phase_mix, mean_speed, speed_amp = speed_params
        x_norm = x / arena_x_m
        y_norm = y / arena_y_m
        field = (
            0.60 * np.sin(2.0 * np.pi * x_norm + phase_x)
            + 0.50 * np.cos(2.0 * np.pi * y_norm + phase_y)
            + 0.35 * np.sin(2.0 * np.pi * (x_norm + 0.7 * y_norm) + phase_mix)
        )
        field = np.clip(field / 1.45, -1.0, 1.0)
        return np.clip(mean_speed + speed_amp * field, 0.05, 0.145)

    # Step 4: simulate one exploratory trajectory with full-wall coverage and smooth speed.
    def simulate_trajectory(local_seed, start, speed_params):
        rng = np.random.default_rng(local_seed)
        pos = np.zeros((n, 2), dtype=float)
        vel = np.zeros((n, 2), dtype=float)
        pos[0] = np.array(start, dtype=float)
        occupancy = np.zeros((5, 4), dtype=float)
        heading_2d = rng.uniform(-np.pi, np.pi)
        speed_state = spatial_speed_field(
            np.array([pos[0, 0]]),
            np.array([pos[0, 1]]),
            speed_params,
        )[0]

        def pick_goal():
            flat = occupancy.ravel()
            pool = np.where(flat <= flat.min() + 0.5)[0]
            choice = pool[rng.integers(0, pool.size)]
            iy = choice % occupancy.shape[1]
            ix = choice // occupancy.shape[1]
            gx = (ix + rng.uniform(0.08, 0.92)) * (arena_x_m / occupancy.shape[0])
            gy = (iy + rng.uniform(0.08, 0.92)) * (arena_y_m / occupancy.shape[1])
            return np.array([gx, gy], dtype=float)

        goal = pick_goal()
        for t in range(1, n):
            ix = min(occupancy.shape[0] - 1, int(pos[t - 1, 0] / arena_x_m * occupancy.shape[0]))
            iy = min(occupancy.shape[1] - 1, int(pos[t - 1, 1] / arena_y_m * occupancy.shape[1]))
            occupancy[ix, iy] += 1.0

            goal_dx = circular_dx(np.array([goal[0]]), np.array([pos[t - 1, 0]]))[0]
            goal_dy = goal[1] - pos[t - 1, 1]
            if np.hypot(goal_dx, goal_dy) < 0.06 or t % 4 == 0:
                goal = pick_goal()
                goal_dx = circular_dx(np.array([goal[0]]), np.array([pos[t - 1, 0]]))[0]
                goal_dy = goal[1] - pos[t - 1, 1]

            goal_heading = np.arctan2(goal_dy, goal_dx)
            heading_2d = wrap_angle(0.05 * heading_2d + 0.95 * goal_heading + rng.normal(0.0, 0.12))
            target_speed = spatial_speed_field(
                np.array([pos[t - 1, 0]]),
                np.array([pos[t - 1, 1]]),
                speed_params,
            )[0]
            speed_state = 0.94 * speed_state + 0.06 * target_speed + rng.normal(0.0, 0.0025)
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
            ix = min(occupancy.shape[0] - 1, int(pos[t, 0] / arena_x_m * occupancy.shape[0]))
            iy = min(occupancy.shape[1] - 1, int(pos[t, 1] / arena_y_m * occupancy.shape[1]))
            occupancy[ix, iy] += 1.0

        # Convert unfolded wall position into a 3D cylinder and compute 3D velocity.
        pos3d = unfolded_to_cylinder_xyz(pos)
        vel3d = np.zeros_like(pos3d)
        vel3d[1:] = np.diff(pos3d, axis=0) / dt
        speed = np.linalg.norm(vel3d, axis=1)

        # Build 3D heading from the self animal's own motion, then intersect it with the wall.
        tangent, vertical, inward = cylinder_frame(pos)
        head_dir3d = np.zeros_like(vel3d)
        head_dir3d[0] = normalize_rows(0.85 * tangent[:1] + 0.55 * inward[:1])[0]
        for t in range(1, n):
            if speed[t] > speed_threshold_mps:
                move_dir = vel3d[t] / max(speed[t], 1e-12)
                candidate = move_dir + 0.22 * inward[t]
            else:
                base_dir = normalize_rows(0.85 * tangent[t : t + 1] + 0.55 * inward[t : t + 1])[0]
                candidate = 0.96 * head_dir3d[t - 1] + 0.04 * base_dir
            head_dir3d[t] = candidate / max(np.linalg.norm(candidate), 1e-12)

        heading = np.arctan2(
            np.sum(head_dir3d * vertical, axis=1),
            np.sum(head_dir3d * tangent, axis=1),
        )
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
                arena_x_m * hit_phi / (2.0 * np.pi),
                np.clip(pos3d[:, 2] + ray_t * head_dir3d[:, 2], 0.0, arena_y_m),
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
            "fast_mask": speed > speed_threshold_mps,
            "fast_fraction": float(np.mean(speed > speed_threshold_mps)),
            "coverage_fraction": float(np.mean(occupancy > 0)),
        }

    # Step 5: choose trajectories that satisfy the speed split and coverage targets.
    def generate_constrained_trajectory(base_seed, start, speed_params):
        best = None
        best_score = np.inf
        for attempt in range(24):
            traj = simulate_trajectory(base_seed + attempt, start, speed_params)
            score = 3.0 * abs(traj["fast_fraction"] - 0.80) + 2.0 * max(0.0, 0.95 - traj["coverage_fraction"])
            if score < best_score:
                best = traj
                best_score = score
            if 0.74 <= traj["fast_fraction"] <= 0.86 and traj["coverage_fraction"] >= 0.95:
                return traj
        return best

    # Step 6: simulate spikes with speed-gated self tuning, other tuning, theta tuning, and self-only spatial view.
    def simulate_spikes(rng, self_traj, other_traj):
        move_fast = self_traj["fast_mask"].astype(float)
        f_self = circular_gaussian(
            self_traj["pos"][:, 0],
            self_traj["pos"][:, 1],
            center=(0.55, 1.42),
            sd=(0.30, 0.26),
            amplitude=1.75,
        )
        f_other = circular_gaussian(
            other_traj["pos"][:, 0],
            other_traj["pos"][:, 1],
            center=(2.12, 0.55),
            sd=(0.34, 0.24),
            amplitude=1.15,
        )
        f_theta = 0.08 * np.cos(wrap_angle(self_traj["heading"] - 0.65))
        f_view = circular_gaussian(
            self_traj["view_pos"][:, 0],
            self_traj["view_pos"][:, 1],
            center=(2.35, 1.15),
            sd=(0.30, 0.22),
            amplitude=0.40,
        )
        log_rate_hz = np.log(1.2) + 0.20 * move_fast + move_fast * f_self + f_other + f_theta + f_view
        rate_hz = np.exp(log_rate_hz)
        y = rng.poisson(rate_hz * dt)
        return y, rate_hz

    # Step 7: build a stratified train/test split for held-out evaluation.
    def make_test_mask(rng, self_pos, other_pos, move_fast, test_frac=0.2):
        nbx, nby = 6, 4
        sx = np.clip((self_pos[:, 0] / arena_x_m * nbx).astype(int), 0, nbx - 1)
        sy = np.clip((self_pos[:, 1] / arena_y_m * nby).astype(int), 0, nby - 1)
        ox = np.clip((other_pos[:, 0] / arena_x_m * nbx).astype(int), 0, nbx - 1)
        oy = np.clip((other_pos[:, 1] / arena_y_m * nby).astype(int), 0, nby - 1)
        ids = sx + nbx * sy + nbx * nby * ox + nbx * nby * nbx * oy + (nbx * nby * nbx * nby) * move_fast.astype(int)
        test_mask = np.zeros(self_pos.shape[0], dtype=bool)
        for bin_id in np.unique(ids):
            idx = np.where(ids == bin_id)[0]
            if idx.size <= 1:
                continue
            n_test = min(max(1, int(round(test_frac * idx.size))), idx.size - 1)
            test_mask[rng.choice(idx, size=n_test, replace=False)] = True
        return test_mask

    # Step 8: fit the PGAM with the requested terms.
    def fit_model(y, self_traj, other_traj, train_mask):
        move_fast, self_x_fast, self_y_fast = masked_self_xy(self_traj)

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
        return gam.fit_full_and_reduced(
            handler.smooths_var,
            th_pval=0.01,
            max_iter=80,
            use_dgcv=True,
            filter_trials=train_mask,
            fit_initial_beta=True,
        )

    # Step 9: predict counts using exactly the same variables used during fitting.
    def predict_counts(model, self_traj, other_traj):
        move_fast, self_x_fast, self_y_fast = masked_self_xy(self_traj)
        x_lookup = {
            "move_fast": [move_fast],
            "self_xy_fast": [self_x_fast, self_y_fast],
            "theta": [self_traj["heading"]],
            "other_xy": [other_traj["pos"][:, 0], other_traj["pos"][:, 1]],
            "view_xy": [self_traj["view_pos"][:, 0], self_traj["view_pos"][:, 1]],
        }
        x_list = [x_lookup[var] for var in model.var_list]
        return np.clip(model.predict(x_list, var_list=model.var_list), 1e-12, None)

    # Step 10: generate self and other trajectories.
    self_speed_params = (0.25, 1.10, -0.55, 0.120, 0.024)
    other_speed_params = (0.90, -0.60, 0.20, 0.116, 0.025)
    self_traj = generate_constrained_trajectory(11, start=(0.3, 0.4), speed_params=self_speed_params)
    other_traj = generate_constrained_trajectory(31, start=(2.2, 1.5), speed_params=other_speed_params)

    # Step 11: simulate spikes and split train/test data.
    rng = np.random.default_rng(seed)
    y, rate_hz = simulate_spikes(rng, self_traj, other_traj)
    move_fast, _, _ = masked_self_xy(self_traj)
    test_mask = make_test_mask(
        np.random.default_rng(7),
        self_pos=self_traj["pos"],
        other_pos=other_traj["pos"],
        move_fast=move_fast,
        test_frac=0.2,
    )
    train_mask = ~test_mask

    # Step 12: fit the PGAM and predict the full session.
    model, reduced_model = fit_model(y, self_traj, other_traj, train_mask)
    mu_all = predict_counts(model, self_traj, other_traj)

    # Step 13: compute evaluation metrics and summary maps.
    y_test = y[test_mask]
    mu_test = mu_all[test_mask]
    mu_null = np.full(y_test.shape[0], max(y[~test_mask].mean(), 1e-12), dtype=float)
    heldout_pseudo_r2 = 1.0 - poisson_loglik(y_test, mu_test) / poisson_loglik(y_test, mu_null)
    cov_pvals = {row["covariate"]: row["p-val"] for row in model.covariate_significance}

    self_map_obs, self_x_edges, self_y_edges = smooth_map(
        self_traj["pos"][:, 0],
        self_traj["pos"][:, 1],
        y,
        mask=self_traj["fast_mask"],
    )
    self_map_pred, _, _ = smooth_map(
        self_traj["pos"][:, 0],
        self_traj["pos"][:, 1],
        mu_all,
        mask=self_traj["fast_mask"],
    )
    other_map_obs, other_x_edges, other_y_edges = smooth_map(other_traj["pos"][:, 0], other_traj["pos"][:, 1], y)
    other_map_pred, _, _ = smooth_map(other_traj["pos"][:, 0], other_traj["pos"][:, 1], mu_all)
    theta_centers, theta_obs = smooth_theta_curve(self_traj["heading"], y)
    _, theta_pred = smooth_theta_curve(self_traj["heading"], mu_all)
    view_map_obs, view_x_edges, view_y_edges = smooth_map(
        self_traj["view_pos"][:, 0],
        self_traj["view_pos"][:, 1],
        y,
    )
    view_map_pred, _, _ = smooth_map(
        self_traj["view_pos"][:, 0],
        self_traj["view_pos"][:, 1],
        mu_all,
    )
    t_axis_full = np.arange(n) * dt
    t_axis_rate, observed_rate_hz, predicted_rate_hz = binned_rate_traces(y, mu_all)
    rho_s, p_s = spearmanr(observed_rate_hz, predicted_rate_hz)
    rho_p, p_p = pearsonr(observed_rate_hz, predicted_rate_hz)

    # Step 14: make the summary figure with consistent observed/predicted comparisons.
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], height_ratios=[1.0, 1.0, 0.75])
    cmap = "jet"

    ax = fig.add_subplot(gs[0, 0])
    plot_wrapped_trajectory(ax, self_traj["pos"])
    ax.scatter(self_traj["pos"][y > 0, 0], self_traj["pos"][y > 0, 1], s=7, color="#d62728", alpha=0.72)
    ax.set_title("Self trajectory and spikes")
    ax.set_xlabel("Self x (m)")
    ax.set_ylabel("Self y (m)")
    ax.set_xlim(0.0, arena_x_m)
    ax.set_ylim(0.0, arena_y_m)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[0, 1])
    add_map_panel(ax, self_map_obs, self_x_edges, self_y_edges, "Observed self map | fast only", "Self x (m)", "Self y (m)", cmap)

    ax = fig.add_subplot(gs[0, 2])
    add_map_panel(
        ax,
        self_map_pred,
        self_x_edges,
        self_y_edges,
        f"Model self map | fast only\np={cov_pvals.get('self_xy_fast', np.nan):.1e}",
        "Self x (m)",
        "Self y (m)",
        cmap,
    )

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(t_axis_full, self_traj["speed"], color="#1f77b4", lw=0.8, label="Self speed")
    ax.plot(t_axis_full, other_traj["speed"], color="#2ca02c", lw=0.8, alpha=0.8, label="Other speed")
    ax.axhline(speed_threshold_mps, color="black", lw=1.0, ls="--", label="0.1 m/s")
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
    ax.set_xlim(0.0, arena_x_m)
    ax.set_ylim(0.0, arena_y_m)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[1, 1])
    add_map_panel(ax, other_map_obs, other_x_edges, other_y_edges, "Observed other map", "Other x (m)", "Other y (m)", cmap)

    ax = fig.add_subplot(gs[1, 2])
    add_map_panel(
        ax,
        other_map_pred,
        other_x_edges,
        other_y_edges,
        f"Model other map\np={cov_pvals.get('other_xy', np.nan):.1e}",
        "Other x (m)",
        "Other y (m)",
        cmap,
    )

    ax = fig.add_subplot(gs[1, 3], projection="polar")
    ax.plot(np.r_[theta_centers, theta_centers[0]], np.r_[theta_obs, theta_obs[0]], color="0.45", lw=1.4, label="Observed")
    ax.plot(np.r_[theta_centers, theta_centers[0]], np.r_[theta_pred, theta_pred[0]], color="#d62728", lw=1.4, label="Model")
    ax.set_title(f"Theta tuning\np={cov_pvals.get('theta', np.nan):.1e}", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15), fontsize=8, frameon=True)

    ax = fig.add_subplot(gs[2, 0])
    add_map_panel(
        ax,
        view_map_obs,
        view_x_edges,
        view_y_edges,
        f"Observed spatial view map\np={cov_pvals.get('view_xy', np.nan):.1e}",
        "View x (m)",
        "View y (m)",
        cmap,
    )

    ax = fig.add_subplot(gs[2, 1:4])
    ax.plot(t_axis_rate, observed_rate_hz, color="0.45", lw=1.0, label="Observed")
    ax.plot(t_axis_rate, predicted_rate_hz, color="#d62728", lw=1.0, alpha=0.85, label="PGAM")
    ax.text(
        0.01,
        0.98,
        "\n".join(
            [
                f"Spearman r={rho_s:.3f}, p={p_s:.2e}",
                f"Pearson r={rho_p:.3f}, p={p_p:.2e}",
                f"Held-out pseudo-R2={heldout_pseudo_r2:.3f}",
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

    fig.suptitle(
        "Social place cell simulation with speed-gated self tuning and PGAM recovery",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    maybe_show()

    # Step 15: print a concise summary for verification.
    print_summary(model, reduced_model, self_traj, other_traj, heldout_pseudo_r2, cov_pvals)
    print(f"Saved: {out_path}")

    return {
        "self_traj": self_traj,
        "other_traj": other_traj,
        "y": y,
        "rate_hz": rate_hz,
        "mu_all": mu_all,
        "model": model,
        "reduced_model": reduced_model,
        "heldout_pseudo_r2": heldout_pseudo_r2,
        "out_path": out_path,
    }


def main():
    run_social_place_cell_speed_gated_simulation()


if __name__ == "__main__":
    main()
