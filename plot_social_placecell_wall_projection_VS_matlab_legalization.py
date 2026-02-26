import argparse
import os
from pathlib import Path
import platform
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve, median_filter, label
from scipy.signal import convolve2d

system = platform.system()
if system == 'Linux':
    DEFAULT_DATA_DIR = Path("/mnt/g/MyGoogleDrive/lab/lib/data/ephys/socialPlaceCell")
elif system == 'Darwin':
    DEFAULT_DATA_DIR = Path("/Users/davidomer/My Drive/lab/lib/data/ephys/socialPlaceCell")

BOUNDS = {"x": (0.0, 0.4), "y": (0.0, 1.1), "z": (0.0, 2.1)}


def setup_matplotlib(force_agg: bool = False):
    if "MPLCONFIGDIR" not in os.environ:
        mplconfig = Path("/tmp/mpl")
        mplconfig.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mplconfig)

    import matplotlib

    if force_agg or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return matplotlib, plt


def maybe_show(plt, matplotlib, show: bool):
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show(block=False)
        plt.pause(0.001)


def finalize_figure(fig, output_dir: Path | None, filename: str):
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")


def resolve_data_dir(cli_data_dir: str) -> Path:
    if cli_data_dir:
        return Path(cli_data_dir).expanduser().resolve()

    env_data_dir = os.environ.get("SOCIAL_PLACECELL_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).expanduser().resolve()

    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / "socialPlaceCell",
        script_dir / "socialPlaceCell",
        Path.cwd() / "data" / "socialPlaceCell",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_DATA_DIR


def load_data(data_dir: Path):
    needed = ["self_xyz.mat", "other_xyz.mat", "spiketime.mat", "timeaxis.mat"]
    missing = [name for name in needed if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required .mat files in data directory: "
            f"{data_dir}. Missing: {', '.join(missing)}. "
            "Pass --data-dir or set SOCIAL_PLACECELL_DATA_DIR."
        )

    self_xyz = loadmat(data_dir / "self_xyz.mat")["self_xyz"]
    other_xyz = loadmat(data_dir / "other_xyz.mat")["other_xyz"]
    spiketimes = loadmat(data_dir / "spiketime.mat")["spiketime"].ravel()
    time_axis = loadmat(data_dir / "timeaxis.mat")["timeSynced"].ravel()
    return self_xyz, other_xyz, spiketimes, time_axis


def spikes_to_samples(spike_times, time_axis):
    dt = float(np.median(np.diff(time_axis))) if time_axis.size > 1 else 1.0
    edges = np.empty(time_axis.size + 1, dtype=float)
    edges[1:-1] = (time_axis[:-1] + time_axis[1:]) / 2.0
    edges[0] = time_axis[0] - dt / 2.0
    edges[-1] = time_axis[-1] + dt / 2.0
    return np.histogram(spike_times, bins=edges)[0].astype(int)


def project_to_6walls_keep_vertical(points_xyz, bounds, snap_bin_size=0.1):
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    valid_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    wall = np.full(points_xyz.shape[0], -1, dtype=int)
    u = np.full(points_xyz.shape[0], np.nan, dtype=float)
    v = np.full(points_xyz.shape[0], np.nan, dtype=float)

    if not np.any(valid_xyz):
        return np.column_stack([u, v]), wall

    dists = np.column_stack([x[valid_xyz] - x0, x1 - x[valid_xyz], y[valid_xyz] - y0, y1 - y[valid_xyz], z[valid_xyz] - z0, z1 - z[valid_xyz]])
    wall_valid = np.argmin(dists, axis=1)
    wall[valid_xyz] = wall_valid

    xw = x1 - x0
    yw = y1 - y0

    # North (y == y0)
    mask = valid_xyz & (wall == 2)
    u[mask] = x1 - x[mask]
    v[mask] = z[mask] - z0

    # West (x == x0)
    mask = valid_xyz & (wall == 0)
    u[mask] = xw + (y[mask] - y0)
    v[mask] = z[mask] - z0

    # South (y == y1)
    mask = valid_xyz & (wall == 3)
    u[mask] = xw + yw + (x[mask] - x0)
    v[mask] = z[mask] - z0

    # East (x == x1)
    mask = valid_xyz & (wall == 1)
    u[mask] = xw + yw + xw + (y1 - y[mask])
    v[mask] = z[mask] - z0

    snap = 0.5 * float(snap_bin_size)
    u_max = 2.0 * (xw + yw)
    v_max = z1 - z0

    # MATLAB-style wall-transition edge snapping:
    # for consecutive samples that switch walls, snap boundary points to the
    # center of edge bins in the unfolded map.
    def snap_vertical_edge(idx, w_from, w_to):
        if w_from == 2:  # North
            if w_to == 0:      # to West
                u[idx] = xw - snap
            elif w_to == 1:    # to East (wrap seam)
                u[idx] = snap
            elif w_to == 5:    # to Up
                v[idx] = v_max - snap
            elif w_to == 4:    # to Down
                v[idx] = snap
        elif w_from == 0:  # West
            if w_to == 2:      # to North
                u[idx] = xw + snap
            elif w_to == 3:    # to South
                u[idx] = xw + yw - snap
            elif w_to == 5:    # to Up
                v[idx] = v_max - snap
            elif w_to == 4:    # to Down
                v[idx] = snap
        elif w_from == 3:  # South
            if w_to == 0:      # to West
                u[idx] = xw + yw + snap
            elif w_to == 1:    # to East
                u[idx] = 2.0 * xw + yw - snap
            elif w_to == 5:    # to Up
                v[idx] = v_max - snap
            elif w_to == 4:    # to Down
                v[idx] = snap
        elif w_from == 1:  # East
            if w_to == 2:      # to North (wrap seam)
                u[idx] = u_max - snap
            elif w_to == 3:    # to South
                u[idx] = 2.0 * xw + yw + snap
            elif w_to == 5:    # to Up
                v[idx] = v_max - snap
            elif w_to == 4:    # to Down
                v[idx] = snap

    change_idx = np.flatnonzero(np.diff(wall) != 0)
    for k in change_idx:
        w0 = int(wall[k])
        w1 = int(wall[k + 1])
        if (w0 < 0) or (w1 < 0):
            continue
        snap_vertical_edge(k, w0, w1)
        snap_vertical_edge(k + 1, w1, w0)

    # Floor (z == z0) and ceiling (z == z1) are deliberately excluded from
    # the map by keeping NaN uv there.
    return np.column_stack([u, v]), wall


def draw_wall_contours(plt, bounds, order):
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    xw = x1 - x0
    yw = y1 - y0
    zw = z1 - z0
    lengths = {"N": xw, "S": xw, "E": yw, "W": yw}
    u0 = 0.0
    for wall in order:
        w = lengths[wall]
        plt.plot([u0, u0 + w, u0 + w, u0, u0], [0, 0, zw, zw, 0], color="black", lw=1)
        plt.text(
            u0 + w / 2,
            zw + 0.02 * zw,
            wall,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        u0 += w


def draw_wall_borders(ax, bounds, order, lw=0.5):
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    xw = x1 - x0
    yw = y1 - y0
    zw = z1 - z0
    lengths = {"N": xw, "S": xw, "E": yw, "W": yw}
    u0 = 0.0
    for wall in order:
        w = lengths[wall]
        ax.plot([u0, u0 + w, u0 + w, u0, u0], [0, 0, zw, zw, 0], color="black", lw=lw)
        u0 += w


def plot_thresholded_trajectory_points(plt, uv, mask, color="0.75", s=6, zorder=2):
    valid = mask & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    plt.scatter(uv[valid, 0], uv[valid, 1], s=s, color=color, zorder=zorder)


def sample_durations_seconds(time_axis):
    if time_axis.size <= 1:
        return np.ones(time_axis.size, dtype=float)
    dt = float(np.median(np.diff(time_axis)))
    edges = np.empty(time_axis.size + 1, dtype=float)
    edges[1:-1] = (time_axis[:-1] + time_axis[1:]) / 2.0
    edges[0] = time_axis[0] - dt / 2.0
    edges[-1] = time_axis[-1] + dt / 2.0
    dts = np.diff(edges)
    if dt > 0.1:
        dts = dts / 1000.0
    return dts


def make_bin_edges(start, stop, bin_size):
    edges = np.arange(start, stop + bin_size, bin_size, dtype=float)
    if edges[-1] < stop:
        edges = np.append(edges, stop)
    return edges


def matlab_fspecial_gaussian(sigma_px):
    if sigma_px <= 0:
        return np.array([[1.0]], dtype=float)
    hsize = int(np.ceil(5 * (sigma_px * 3)))
    if hsize % 2 == 0:
        hsize += 1
    radius = hsize // 2
    ax = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2)))
    s = np.sum(kernel)
    if s > 0:
        kernel /= s
    return kernel


def matlab_like_smooth_2d(map2d, kernel):
    # Match MATLAB-style topology: circular seam in unfolded-u (axis 0),
    # symmetric borders in v (axis 1), then 2D convolution.
    pad_u = kernel.shape[0] // 2
    pad_v = kernel.shape[1] // 2
    padded = np.pad(map2d, ((pad_u, pad_u), (0, 0)), mode="wrap")
    padded = np.pad(padded, ((0, 0), (pad_v, pad_v)), mode="symmetric")
    return convolve2d(padded, kernel, mode="valid")


def matlab_circular_neighbor_count(binary_map):
    """MATLAB-style 8-neighbor count with circular wrap along unfolded-u axis."""
    b = binary_map.astype(int)

    # Vertical shifts without wrapping (outside map is zero).
    up = np.zeros_like(b)
    up[:, 1:] = b[:, :-1]
    down = np.zeros_like(b)
    down[:, :-1] = b[:, 1:]

    # Horizontal shifts wrap circularly (unfolded wall seam).
    left = np.roll(b, 1, axis=0)
    right = np.roll(b, -1, axis=0)
    up_left = np.roll(up, 1, axis=0)
    up_right = np.roll(up, -1, axis=0)
    down_left = np.roll(down, 1, axis=0)
    down_right = np.roll(down, -1, axis=0)

    return left + right + up + down + up_left + up_right + down_left + down_right


def matlab_style_run_mask(xyz, time_axis, speed_th=0.1):
    # Mirrors trajectoryToRuns_displacement.m:
    # - median filter
    # - 1 s displacement thresholding
    # - merge short stationary gaps (<=500 time units)
    # - keep runs longer than 1000 time units
    if xyz.shape[0] < 3 or time_axis.size < 3:
        return np.zeros(time_axis.shape[0], dtype=bool)

    dt = float(np.mean(np.diff(time_axis)))
    if not np.isfinite(dt) or dt <= 0:
        return np.zeros(time_axis.shape[0], dtype=bool)

    dt1 = 1000.0
    min_movement = 1000.0
    min_stationary = 500.0

    samples = max(1, int(np.round(dt1 / dt)))
    if xyz.shape[0] <= samples:
        return np.zeros(time_axis.shape[0], dtype=bool)

    samples_to_med = max(1, int(np.round(100.0 / dt)))
    median_xyz = median_filter(xyz, size=(samples_to_med, 1), mode="nearest")

    displacement = np.linalg.norm(median_xyz[samples:, :] - median_xyz[:-samples, :], axis=1)
    time1 = time_axis[samples:]
    above_th = displacement > speed_th
    if not np.any(above_th):
        return np.zeros(time_axis.shape[0], dtype=bool)

    # 1D connected components over above-threshold samples.
    starts = []
    ends = []
    in_run = False
    start = 0
    for i, flag in enumerate(above_th):
        if flag and not in_run:
            in_run = True
            start = i
        elif (not flag) and in_run:
            in_run = False
            starts.append(start)
            ends.append(i - 1)
    if in_run:
        starts.append(start)
        ends.append(above_th.size - 1)

    starts = np.array(starts, dtype=int)
    ends = np.array(ends, dtype=int)
    if starts.size == 0:
        return np.zeros(time_axis.shape[0], dtype=bool)

    # Merge runs separated by short stationary periods.
    if starts.size > 1:
        breaks_lengths = time1[starts[1:]] - time1[ends[:-1]]
        combine_idx = np.where(breaks_lengths <= min_stationary)[0]
        if combine_idx.size > 0:
            starts = np.delete(starts, combine_idx + 1)
            ends = np.delete(ends, combine_idx)

    # Keep only long-enough runs.
    runs_lengths = time1[ends] - time1[starts]
    keep = runs_lengths > min_movement
    starts = starts[keep]
    ends = ends[keep]

    traj_mask = np.zeros(time_axis.shape[0], dtype=bool)
    for s, e in zip(starts, ends):
        orig_s = s + samples
        orig_e = e + samples
        traj_mask[orig_s:orig_e + 1] = True
    return traj_mask


def spikes_uv_from_runs(spike_times, time_axis, xyz, traj_mask, project_fn, bounds):
    # MATLAB-style spike assignment:
    # select spikes inside run windows and interpolate xyz at exact spike times.
    idx = np.flatnonzero(traj_mask)
    if idx.size == 0 or spike_times.size == 0:
        return np.empty((0, 2), dtype=float)

    split_points = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[split_points + 1], 0, idx[0])
    ends = np.append(idx[split_points], idx[-1])

    spikes_xyz = []
    for s, e in zip(starts, ends):
        t_run = time_axis[s:e + 1]
        xyz_run = xyz[s:e + 1, :]
        if t_run.size < 2:
            continue
        in_run = (spike_times >= t_run[0]) & (spike_times <= t_run[-1])
        ts = spike_times[in_run]
        if ts.size == 0:
            continue
        x_spk = np.interp(ts, t_run, xyz_run[:, 0])
        y_spk = np.interp(ts, t_run, xyz_run[:, 1])
        z_spk = np.interp(ts, t_run, xyz_run[:, 2])
        spikes_xyz.append(np.column_stack([x_spk, y_spk, z_spk]))

    if not spikes_xyz:
        return np.empty((0, 2), dtype=float)
    spikes_xyz = np.vstack(spikes_xyz)
    spikes_uv, _ = project_fn(spikes_xyz, bounds)
    valid = np.isfinite(spikes_uv[:, 0]) & np.isfinite(spikes_uv[:, 1])
    return spikes_uv[valid]


def compute_place_field_rate_map(
    uv,
    sample_dt_sec,
    occupancy_mask,
    spikes_uv=None,
    bin_size=0.1,
    min_occupancy_sec=0.1,
    smoothing_sigma_px=1.5,
):
    x0, x1 = BOUNDS["x"]
    y0, y1 = BOUNDS["y"]
    z0, z1 = BOUNDS["z"]
    u_max = 2.0 * ((x1 - x0) + (y1 - y0))
    v_max = z1 - z0
    u_edges = make_bin_edges(0.0, u_max, bin_size)
    v_edges = make_bin_edges(0.0, v_max, bin_size)

    valid_occ = occupancy_mask & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    uv_occ = uv[valid_occ]
    dt_occ = sample_dt_sec[valid_occ]

    occupancy_sec, _, _ = np.histogram2d(
        uv_occ[:, 0], uv_occ[:, 1], bins=[u_edges, v_edges], weights=dt_occ
    )
    if spikes_uv is None or spikes_uv.size == 0:
        spike_count_map = np.zeros_like(occupancy_sec, dtype=float)
    else:
        spike_count_map, _, _ = np.histogram2d(
            spikes_uv[:, 0], spikes_uv[:, 1], bins=[u_edges, v_edges]
        )

    rate_hz = np.full_like(occupancy_sec, np.nan, dtype=float)

    # MATLAB behavior: bins with <100 ms occupancy are treated as zero time spent.
    occupancy_for_mask = occupancy_sec.copy()
    occupancy_for_mask[occupancy_for_mask < min_occupancy_sec] = 0.0
    occupied = occupancy_for_mask > 0.0
    occupied_neighbor_count = matlab_circular_neighbor_count(occupied)

    # MATLAB-style legalization:
    # 1) occupied bins with >=1 occupied neighbors are legal
    # 2) zero-time bins with >=4 occupied neighbors are legal
    legal = ((~occupied) & (occupied_neighbor_count >= 4)) | (occupied & (occupied_neighbor_count >= 1))
    # MATLAB behavior: remove connected legal islands smaller than 4 pixels.
    structure = np.ones((3, 3), dtype=int)
    cc, n_cc = label(legal.astype(int), structure=structure)
    if n_cc > 0:
        for idx in range(1, n_cc + 1):
            comp = (cc == idx)
            if int(np.sum(comp)) < 4:
                legal[comp] = False

    # MATLAB-like order: smooth spike density and time-spent maps first, then divide.
    gaussian_kernel = matlab_fspecial_gaussian(smoothing_sigma_px)
    smoothed_spike_count_map = matlab_like_smooth_2d(spike_count_map, gaussian_kernel)
    smoothed_occupancy_sec = matlab_like_smooth_2d(occupancy_for_mask, gaussian_kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed_rate = smoothed_spike_count_map / smoothed_occupancy_sec
    valid = legal & np.isfinite(smoothed_rate)
    rate_hz[valid] = smoothed_rate[valid]
    return rate_hz, occupancy_sec, spike_count_map, u_edges, v_edges


def main():
    parser = argparse.ArgumentParser(description="Plot social placecell wall projections (VS-friendly)")
    parser.add_argument("--data-dir", default="", help="Path containing self_xyz.mat, other_xyz.mat, spiketime.mat, timeaxis.mat")
    parser.add_argument("--no-show", action="store_true", help="Do not open figure windows")
    parser.add_argument("--force-agg", action="store_true", help="Force matplotlib Agg backend")
    parser.add_argument("--save-dir", default="", help="Optional output directory for PNG files")
    args = parser.parse_args()

    matplotlib, plt = setup_matplotlib(force_agg=args.force_agg)
    show_plots = not args.no_show
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None

    data_dir = resolve_data_dir(args.data_dir)
    self_xyz, other_xyz, spiketimes, time_axis = load_data(data_dir)
    time_valid = np.isfinite(time_axis)
    self_xyz = self_xyz[time_valid]
    other_xyz = other_xyz[time_valid]
    time_axis = time_axis[time_valid]

    # Match MATLAB run extraction for the self map: do not discard samples just
    # because other_xyz is missing.
    self_valid = np.isfinite(self_xyz[:, 0]) & np.isfinite(self_xyz[:, 1]) & np.isfinite(self_xyz[:, 2])
    self_xyz = self_xyz[self_valid]
    other_xyz = other_xyz[self_valid]
    time_axis = time_axis[self_valid]

    speed_thresh = 0.1
    traj_mask = matlab_style_run_mask(self_xyz, time_axis, speed_th=speed_thresh)

    self_uv, _ = project_to_6walls_keep_vertical(self_xyz, BOUNDS)
    other_uv, _ = project_to_6walls_keep_vertical(other_xyz, BOUNDS)
    self_spike_uv = spikes_uv_from_runs(
        spiketimes, time_axis, self_xyz, traj_mask, project_to_6walls_keep_vertical, BOUNDS
    )
    other_spike_uv = spikes_uv_from_runs(
        spiketimes, time_axis, other_xyz, traj_mask, project_to_6walls_keep_vertical, BOUNDS
    )
    sample_dt_sec = sample_durations_seconds(time_axis)
    self_place_field_hz, self_occupancy_sec, self_spike_count_map, u_edges, v_edges = (
        compute_place_field_rate_map(
            self_uv,
            sample_dt_sec,
            occupancy_mask=traj_mask,
            spikes_uv=self_spike_uv,
            bin_size=0.1,
            smoothing_sigma_px=1.5,
        )
    )
    other_place_field_hz, other_occupancy_sec, other_spike_count_map, _, _ = compute_place_field_rate_map(
        other_uv,
        sample_dt_sec,
        occupancy_mask=traj_mask,
        spikes_uv=other_spike_uv,
        bin_size=0.1,
        smoothing_sigma_px=1.5,
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.9), sharex=True, sharey=True)
    total_spikes_used = int(self_spike_uv.shape[0])
    u_min, u_max = u_edges[0], u_edges[-1]
    v_min, v_max = v_edges[0], v_edges[-1]

    # Top row: trajectories and spikes.
    ax_t_self = axes[0, 0]
    plt.sca(ax_t_self)
    plot_thresholded_trajectory_points(plt, self_uv, traj_mask, color="0.75", s=6, zorder=2)
    if self_spike_uv.size > 0:
        ax_t_self.scatter(self_spike_uv[:, 0], self_spike_uv[:, 1], s=10, color="red", zorder=3)
    draw_wall_contours(plt, BOUNDS, order=["N", "E", "S", "W"])
    ax_t_self.set_title("Self trajectory (MATLAB-style runs) and spikes")
    ax_t_self.set_xlim(u_min, u_max)
    ax_t_self.set_ylim(v_min, v_max)
    ax_t_self.set_aspect("equal", "box")
    ax_t_self.set_ylabel("v (m)")

    ax_t_other = axes[0, 1]
    plt.sca(ax_t_other)
    plot_thresholded_trajectory_points(plt, other_uv, traj_mask, color="0.75", s=6, zorder=2)
    if other_spike_uv.size > 0:
        ax_t_other.scatter(other_spike_uv[:, 0], other_spike_uv[:, 1], s=10, color="red", zorder=3)
    draw_wall_contours(plt, BOUNDS, order=["N", "W", "S", "E"])
    ax_t_other.set_title("Other trajectory (MATLAB-style runs) and spikes")
    ax_t_other.set_xlim(u_min, u_max)
    ax_t_other.set_ylim(v_min, v_max)
    ax_t_other.set_aspect("equal", "box")

    # Bottom row: place firing maps.
    map_panels = [
        (axes[1, 0], "Self place field", self_place_field_hz, ["N", "E", "S", "W"]),
        (axes[1, 1], "Other place field", other_place_field_hz, ["N", "W", "S", "E"]),
    ]
    for ax, title, rate_map, order in map_panels:
        finite_vals = rate_map[np.isfinite(rate_map)]
        map_max = float(np.max(finite_vals)) if finite_vals.size else 0.0
        vmax = map_max if map_max > 0.0 else 1.0
        im = ax.imshow(
            rate_map.T,
            origin="lower",
            extent=[u_min, u_max, v_min, v_max],
            cmap="jet",
            vmin=0.0,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        draw_wall_borders(ax, BOUNDS, order=order, lw=0.5)
        ax.set_title(f"{title}\nmax = {map_max:.2f} Hz")
        ax.set_xlabel("u (m)")
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_min, v_max)
        ax.text(
            0.02,
            0.98,
            f"max: {map_max:.2f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Rate (Hz)")
    axes[1, 0].set_ylabel("v (m)")
    fig.text(
        0.5,
        0.985,
        f"Spikes used for place-field maps (MATLAB-style runs, th={speed_thresh:.1f}): {total_spikes_used}",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.95, wspace=0.08, hspace=0.12)
    finalize_figure(fig, save_dir, "behavior_and_place_fields_2x2.png")
    maybe_show(plt, matplotlib, show_plots)

    print(f"Loaded data from: {data_dir}")
    if save_dir is not None:
        print(f"Saved figures to: {save_dir}")

    if show_plots and matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
