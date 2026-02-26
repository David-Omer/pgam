#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.ndimage import median_filter
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parent
PGAM_SRC = REPO_ROOT / "PGAM" / "src" / "PGAM"
if str(PGAM_SRC) not in sys.path:
    sys.path.append(str(PGAM_SRC))

import GAM_library as gl
import gam_data_handlers as gdh

DEFAULT_DATA_DIR = Path("/Users/davidomer/My Drive/lab/lib/data/ephys/socialPlaceCell")
BOUNDS = {"x": (0.0, 0.4), "y": (0.0, 1.1), "z": (0.0, 2.1)}


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

    dists = np.column_stack(
        [
            x[valid_xyz] - x0,
            x1 - x[valid_xyz],
            y[valid_xyz] - y0,
            y1 - y[valid_xyz],
            z[valid_xyz] - z0,
            z1 - z[valid_xyz],
        ]
    )
    wall_valid = np.argmin(dists, axis=1)
    wall[valid_xyz] = wall_valid

    xw = x1 - x0
    yw = y1 - y0

    mask = valid_xyz & (wall == 2)  # North
    u[mask] = x1 - x[mask]
    v[mask] = z[mask] - z0

    mask = valid_xyz & (wall == 0)  # West
    u[mask] = xw + (y[mask] - y0)
    v[mask] = z[mask] - z0

    mask = valid_xyz & (wall == 3)  # South
    u[mask] = xw + yw + (x[mask] - x0)
    v[mask] = z[mask] - z0

    mask = valid_xyz & (wall == 1)  # East
    u[mask] = xw + yw + xw + (y1 - y[mask])
    v[mask] = z[mask] - z0

    snap = 0.5 * float(snap_bin_size)
    u_max = 2.0 * (xw + yw)
    v_max = z1 - z0

    def snap_vertical_edge(idx, w_from, w_to):
        if w_from == 2:
            if w_to == 0:
                u[idx] = xw - snap
            elif w_to == 1:
                u[idx] = snap
            elif w_to == 5:
                v[idx] = v_max - snap
            elif w_to == 4:
                v[idx] = snap
        elif w_from == 0:
            if w_to == 2:
                u[idx] = xw + snap
            elif w_to == 3:
                u[idx] = xw + yw - snap
            elif w_to == 5:
                v[idx] = v_max - snap
            elif w_to == 4:
                v[idx] = snap
        elif w_from == 3:
            if w_to == 0:
                u[idx] = xw + yw + snap
            elif w_to == 1:
                u[idx] = 2.0 * xw + yw - snap
            elif w_to == 5:
                v[idx] = v_max - snap
            elif w_to == 4:
                v[idx] = snap
        elif w_from == 1:
            if w_to == 2:
                u[idx] = u_max - snap
            elif w_to == 3:
                u[idx] = 2.0 * xw + yw + snap
            elif w_to == 5:
                v[idx] = v_max - snap
            elif w_to == 4:
                v[idx] = snap

    change_idx = np.flatnonzero(np.diff(wall) != 0)
    for k in change_idx:
        w0 = int(wall[k])
        w1 = int(wall[k + 1])
        if (w0 < 0) or (w1 < 0):
            continue
        snap_vertical_edge(k, w0, w1)
        snap_vertical_edge(k + 1, w1, w0)

    # Keep floor/ceiling as NaN uv exactly like the reference code.
    return np.column_stack([u, v]), wall


def matlab_style_run_mask(xyz, time_axis, speed_th=0.1):
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

    if starts.size > 1:
        breaks_lengths = time1[starts[1:]] - time1[ends[:-1]]
        combine_idx = np.where(breaks_lengths <= min_stationary)[0]
        if combine_idx.size > 0:
            starts = np.delete(starts, combine_idx + 1)
            ends = np.delete(ends, combine_idx)

    runs_lengths = time1[ends] - time1[starts]
    keep = runs_lengths > min_movement
    starts = starts[keep]
    ends = ends[keep]

    traj_mask = np.zeros(time_axis.shape[0], dtype=bool)
    for s, e in zip(starts, ends):
        orig_s = s + samples
        orig_e = e + samples
        traj_mask[orig_s : orig_e + 1] = True
    return traj_mask


def _load_spike_times(path: Path, unit_index: int) -> np.ndarray:
    spk = loadmat(path)["spiketime"]

    # Common case in your dataset: shape (1, N) float64.
    if np.issubdtype(spk.dtype, np.number):
        arr = np.asarray(spk).reshape(-1)
        return arr[np.isfinite(arr)]

    # Cell-array fallback (multiple units): pick one unit.
    flat = spk.ravel()
    if flat.size == 0:
        raise ValueError("spiketime.mat is empty")
    if unit_index < 0 or unit_index >= flat.size:
        raise IndexError(f"unit-index {unit_index} out of range [0, {flat.size - 1}]")
    arr = np.asarray(flat[unit_index]).reshape(-1)
    return arr[np.isfinite(arr)]


def _load_inputs(data_dir: Path, unit_index: int):
    needed = ["self_xyz.mat", "other_xyz.mat", "spiketime.mat", "timeaxis.mat"]
    missing = [f for f in needed if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {', '.join(missing)}")

    self_xyz = np.asarray(loadmat(data_dir / "self_xyz.mat")["self_xyz"], dtype=float)
    other_xyz = np.asarray(loadmat(data_dir / "other_xyz.mat")["other_xyz"], dtype=float)
    time_axis = np.asarray(loadmat(data_dir / "timeaxis.mat")["timeSynced"], dtype=float).reshape(-1)
    spike_times = _load_spike_times(data_dir / "spiketime.mat", unit_index=unit_index)

    n = min(self_xyz.shape[0], other_xyz.shape[0], time_axis.shape[0])
    self_xyz = self_xyz[:n, :]
    other_xyz = other_xyz[:n, :]
    time_axis = time_axis[:n]

    return self_xyz, other_xyz, spike_times, time_axis


def _fit_model(
    self_uv: np.ndarray,
    other_uv: np.ndarray,
    movement_state: np.ndarray,
    spike_counts: np.ndarray,
    knots_num: int,
    max_iter: int,
    pval_th: float,
):
    sm_handler = gdh.smooths_handler()

    # 1) Self position (u cyclic, v non-cyclic), then multiply by movement state.
    sm_handler.add_smooth(
        "self_pos_moving",
        [self_uv[:, 0], self_uv[:, 1]],
        ord=4,
        knots=None,
        knots_num=knots_num,
        perc_out_range=0.0,
        is_cyclic=[True, False],
        lam=None,
        penalty_type="der",
        der=2,
        knots_percentiles=(0, 100),
    )

    # Keep exact movement gating behavior but avoid exact-zero rank collapse in PGAM
    # internals by using a tiny epsilon for stationary samples.
    move_scale = np.where(np.asarray(movement_state) > 0, 1.0, 1e-6)
    Xself = sm_handler["self_pos_moving"].X
    if sp.issparse(Xself):
        sm_handler["self_pos_moving"].X = Xself.multiply(move_scale[:, None]).tocsr()
    else:
        sm_handler["self_pos_moving"].X = Xself * move_scale[:, None]

    # 2) Other position (u cyclic, v non-cyclic).
    sm_handler.add_smooth(
        "other_pos",
        [other_uv[:, 0], other_uv[:, 1]],
        ord=4,
        knots=None,
        knots_num=knots_num,
        perc_out_range=0.0,
        is_cyclic=[True, False],
        lam=None,
        penalty_type="der",
        der=2,
        knots_percentiles=(0, 100),
    )

    link = sm.genmod.families.links.Log()
    poiss = sm.genmod.families.family.Poisson(link=link)
    var_list = ["self_pos_moving", "other_pos"]

    gam = gl.general_additive_model(sm_handler, var_list, spike_counts, poiss, fisher_scoring=False)
    full, reduced = gam.fit_full_and_reduced(
        var_list,
        th_pval=pval_th,
        max_iter=max_iter,
        use_dgcv=True,
        compute_MI=False,
        filter_trials=np.ones(spike_counts.shape[0], dtype=bool),
    )

    return full, reduced


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load socialPlaceCell data, project self/other 3D positions to unfolded walls, "
            "compute MATLAB-style movement state (>0.1 m/s), and fit requested PGAM model."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path containing self_xyz.mat, other_xyz.mat, spiketime.mat, timeaxis.mat",
    )
    parser.add_argument("--unit-index", type=int, default=0, help="Unit index if spiketime.mat is a cell array")
    parser.add_argument("--speed-th", type=float, default=0.1, help="Movement threshold in m/s")
    parser.add_argument("--knots-num", type=int, default=15, help="Knots per dimension for both 2D smooths")
    parser.add_argument("--max-iter", type=int, default=100, help="Max optimization iterations")
    parser.add_argument("--pval-th", type=float, default=0.001, help="P-value threshold for reduced model")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on sample count after filtering (0 means use all)",
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=Path("social_placecell_pgam_fit.npz"),
        help="Output npz for processed inputs and lightweight fit summaries",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    self_xyz, other_xyz, spike_times, time_axis = _load_inputs(data_dir, unit_index=args.unit_index)

    # Same spike-time binning as existing code.
    spike_counts = spikes_to_samples(spike_times, time_axis).astype(float)

    # Same projection as previous script.
    self_uv, self_wall = project_to_6walls_keep_vertical(self_xyz, BOUNDS)
    other_uv, other_wall = project_to_6walls_keep_vertical(other_xyz, BOUNDS)

    # Same MATLAB-style velocity/run extraction logic as previous script.
    movement_state = matlab_style_run_mask(self_xyz, time_axis, speed_th=args.speed_th).astype(float)

    # Keep only rows usable by both 2D covariates and response.
    valid = (
        np.isfinite(time_axis)
        & np.isfinite(spike_counts)
        & np.isfinite(self_uv[:, 0])
        & np.isfinite(self_uv[:, 1])
        & np.isfinite(other_uv[:, 0])
        & np.isfinite(other_uv[:, 1])
    )

    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise RuntimeError("No valid samples after projection/finite filtering")

    if args.max_samples and idx.size > args.max_samples:
        keep = np.sort(rng.choice(idx, size=args.max_samples, replace=False))
    else:
        keep = idx

    y = spike_counts[keep]
    self_uv_fit = self_uv[keep]
    other_uv_fit = other_uv[keep]
    move_fit = movement_state[keep]

    print(f"Loaded: {data_dir}")
    print(f"Total samples: {time_axis.size:,}")
    print(f"Valid projected samples: {idx.size:,}")
    print(f"Samples used for fit: {keep.size:,}")
    print(f"Spike count sum in fit samples: {int(np.sum(y)):,}")
    print(f"Movement-state fraction: {float(np.mean(move_fit)):.3f}")

    full, reduced = _fit_model(
        self_uv=self_uv_fit,
        other_uv=other_uv_fit,
        movement_state=move_fit,
        spike_counts=y,
        knots_num=args.knots_num,
        max_iter=args.max_iter,
        pval_th=args.pval_th,
    )

    print("\nFull model covariate significance:")
    print(full.covariate_significance)
    if reduced is None:
        print("Reduced model: none (no covariates passed p-value threshold)")
        reduced_vars = np.array([], dtype=object)
    else:
        reduced_vars = np.array(reduced.var_list, dtype=object)
        print(f"Reduced model variables: {list(reduced.var_list)}")

    args.save_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.save_npz,
        fit_indices=keep,
        movement_state=move_fit,
        self_uv=self_uv_fit,
        other_uv=other_uv_fit,
        spike_counts=y,
        self_wall=self_wall[keep],
        other_wall=other_wall[keep],
        full_var_list=np.array(full.var_list, dtype=object),
        reduced_var_list=reduced_vars,
        covariate_significance=full.covariate_significance,
    )
    print(f"Saved: {args.save_npz.resolve()}")


if __name__ == "__main__":
    main()
