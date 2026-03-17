import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize as scipy_minimize

import PGAM.GAM_library as pgam_gam_library
import PGAM.der_wrt_smoothing as pgam_derivatives
from PGAM.GAM_library import general_additive_model
from PGAM.gam_data_handlers import smooths_handler


DATA_PATH = Path("simulated_neuorn.npz")
RESULTS_PATH = Path("simulated_neuorn_pgam_cv_results.json")
SUMMARY_PNG_PATH = Path("simulated_neuorn_pgam_cv_summary.png")
SPEED_THRESHOLD_MPS = 0.1
UNFOLDED_U_MAX_M = 6.0
UNFOLDED_V_MAX_M = 2.0


"""
This script fits the user-specified model:

    log lambda_t =
        beta_0
        + beta_m m_t
        + m_t * f_self(x_t^s, y_t^s)
        + f_other(x_t^o, y_t^o)
        + f_theta(theta_t)
        + f_view(x_t^view, y_t^view)
        + log(Delta)

Implementation notes:
- beta_m * m_t is implemented as a literal one-column parametric effect.
- m_t * f_self(...) is implemented by multiplying the self-position spline basis by m_t.
- log(Delta) is constant because spike counts are binned at a fixed behavior-bin width,
  so it is absorbed into the intercept in the Poisson count model.
"""


def quiet_minimize(*args, **kwargs):
    options = dict(kwargs.get("options") or {})
    options["disp"] = False
    kwargs["options"] = options
    return scipy_minimize(*args, **kwargs)


pgam_derivatives.minimize = quiet_minimize
pgam_gam_library.minimize = quiet_minimize


def poisson_loglik(y, mu):
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    y = np.asarray(y, dtype=float)
    return float(np.sum(y * np.log(mu) - mu))


def circular_shift(x, shift):
    return np.roll(x, int(shift))


def build_covariates(data):
    y = data["neuron_spike_counts"].reshape(-1, 24).sum(axis=1).astype(float)
    self_pos = data["self_pos_wall"].astype(float)
    other_pos = data["other_pos_wall"].astype(float)
    view_pos = data["self_spatial_view_wall"].astype(float)
    speed = data["self_speed_m_s"].astype(float)
    move_fast = (speed > SPEED_THRESHOLD_MPS).astype(float)
    theta = np.deg2rad(data["self_yaw_deg"].astype(float))

    return y, {
        "move_fast": [move_fast],
        "self_xy_fast": [self_pos[:, 0], self_pos[:, 1]],
        "theta": [theta],
        "other_xy": [other_pos[:, 0], other_pos[:, 1]],
        "view_xy": [view_pos[:, 0], view_pos[:, 1]],
    }


def subset_covariates(x_lookup, mask):
    return {name: [np.asarray(dim)[mask] for dim in dims] for name, dims in x_lookup.items()}


def build_handler(x_lookup):
    handler = smooths_handler()
    handler.add_smooth(
        "move_fast",
        x_lookup["move_fast"],
        knots=[np.array([0.0, 0.0, 1.0, 1.0], dtype=float)],
        ord=2,
        penalty_type="der",
        der=1,
        is_cyclic=[False],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "self_xy_fast",
        x_lookup["self_xy_fast"],
        knots_num=7,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True, False],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "theta",
        x_lookup["theta"],
        knots_num=9,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "other_xy",
        x_lookup["other_xy"],
        knots_num=7,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True, False],
        knots_percentiles=(0, 100),
    )
    handler.add_smooth(
        "view_xy",
        x_lookup["view_xy"],
        knots_num=7,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[True, False],
        knots_percentiles=(0, 100),
    )
    set_exact_linear_move_term(handler, np.asarray(x_lookup["move_fast"][0], dtype=float))
    apply_exact_self_interaction(handler, np.asarray(x_lookup["move_fast"][0], dtype=float))
    return handler


def set_exact_linear_move_term(handler, move_fast):
    smooth = handler["move_fast"]
    move_fast = np.asarray(move_fast, dtype=float).reshape(-1, 1)
    smooth.X = move_fast.copy()
    smooth.nan_filter = np.zeros(move_fast.shape[0], dtype=bool)
    smooth.colMean_X = np.zeros(0, dtype=float)
    smooth.B_list = [np.zeros((1, 1), dtype=float)]
    smooth.S_list = [np.zeros((1, 1), dtype=float)]
    smooth.lam = np.array([1.0], dtype=float)
    smooth.basis_dim = 1


def apply_exact_self_interaction(handler, move_fast):
    smooth = handler["self_xy_fast"]
    gate = np.asarray(move_fast, dtype=float).reshape(-1, 1)
    if sparse.issparse(smooth.X):
        smooth.X = sparse.diags(gate[:, 0]).dot(smooth.X)
    else:
        smooth.X = smooth.X * gate
    active = smooth.X[:, :-1]
    if sparse.issparse(active):
        active = active.toarray()
    smooth.colMean_X = np.asarray(active[~smooth.nan_filter, :], dtype=float).mean(axis=0)


def predict_model(model, x_lookup):
    n = x_lookup["move_fast"][0].shape[0]
    eta = np.full(n, model.beta[0], dtype=float)

    for var_name in model.var_list:
        if var_name == "move_fast":
            fX = np.asarray(x_lookup["move_fast"][0], dtype=float).reshape(-1, 1)
        else:
            X = x_lookup[var_name]
            nan_filter = np.array(np.sum(np.isnan(np.array(X)), axis=0), dtype=bool)
            fX = model.eval_basis(X, var_name, sparseX=False, domain_fun=model.domain_fun[var_name])
            if sparse.issparse(fX):
                fX = fX.toarray()
            if fX.shape[1] > 1:
                fX = fX[:, :-1] - model.smooth_info[var_name]["colMean_X"]
            if var_name == "self_xy_fast":
                gate = np.asarray(x_lookup["move_fast"][0], dtype=float).reshape(-1, 1)
                fX = fX * gate
            fX[nan_filter, :] = 0.0
        eta = eta + np.dot(fX, model.beta[model.index_dict[var_name]])

    return np.clip(model.family.link.inverse(eta), 1e-12, None)


def fit_fold_models(y_train, x_lookup_train, max_iter, pval_threshold):
    handler = build_handler(x_lookup_train)
    family = sm.genmod.families.family.Poisson(link=sm.genmod.families.links.log())
    gam = general_additive_model(handler, handler.smooths_var, y_train, family)
    full_model, reduced_model = gam.fit_full_and_reduced(
        handler.smooths_var,
        th_pval=pval_threshold,
        max_iter=max_iter,
        use_dgcv=True,
        fit_initial_beta=True,
        reducedAdaptive=False,
    )
    return full_model, reduced_model


def get_process_pool(max_workers):
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)


def get_parallel_executor(max_workers):
    try:
        return get_process_pool(max_workers)
    except (OSError, PermissionError):
        return ThreadPoolExecutor(max_workers=max_workers)


def resolve_job_count(requested_jobs, n_tasks):
    if requested_jobs is None or requested_jobs <= 0:
        requested_jobs = os.cpu_count() or 1
    return max(1, min(requested_jobs, n_tasks))


def evaluate_single_fold(fold_idx, test_idx, y, x_lookup, max_iter, pval_threshold):
    test_mask = np.zeros(y.shape[0], dtype=bool)
    test_mask[np.asarray(test_idx, dtype=int)] = True
    train_mask = ~test_mask

    y_train = y[train_mask]
    y_test = y[test_mask]
    x_train = subset_covariates(x_lookup, train_mask)
    x_test = subset_covariates(x_lookup, test_mask)

    start = perf_counter()
    full_model, reduced_model = fit_fold_models(y_train, x_train, max_iter, pval_threshold)
    fit_time_s = perf_counter() - start

    mu_null = np.full(y_test.shape[0], max(y_train.mean(), 1e-12), dtype=float)
    ll_null = poisson_loglik(y_test, mu_null)

    if reduced_model is None:
        mu_model = mu_null.copy()
        ll_model = ll_null
        kept_vars = []
    else:
        mu_model = predict_model(reduced_model, x_test)
        ll_model = poisson_loglik(y_test, mu_model)
        kept_vars = list(reduced_model.var_list)

    delta_ll = ll_model - ll_null
    bits_per_spike = delta_ll / max(np.sum(y_test), 1e-12) / np.log(2.0)
    return {
        "fold": fold_idx,
        "n_test": int(test_mask.sum()),
        "ll_model": ll_model,
        "ll_null": ll_null,
        "delta_ll": delta_ll,
        "bits_per_spike": bits_per_spike,
        "kept_vars": kept_vars,
        "full_vars": list(full_model.var_list),
        "fit_time_s": fit_time_s,
    }


def fit_full_model(y, x_lookup, max_iter, pval_threshold):
    fit_start = perf_counter()
    full_model, reduced_model = fit_fold_models(y, x_lookup, max_iter, pval_threshold)
    fit_elapsed_s = perf_counter() - fit_start
    mu_all = predict_model(full_model, x_lookup)
    cov_pvals = {row["covariate"]: row["p-val"] for row in full_model.covariate_significance}
    return {
        "full_model": full_model,
        "reduced_model": reduced_model,
        "mu_all": mu_all,
        "cov_pvals": cov_pvals,
        "fit_elapsed_s": fit_elapsed_s,
    }


def run_initial_model_fit(y, x_lookup, max_iter, pval_threshold):
    return fit_full_model(y, x_lookup, max_iter, pval_threshold)


def make_folds(n_samples, n_folds):
    fold_ids = np.array_split(np.arange(n_samples), n_folds)
    return [np.asarray(idx, dtype=int) for idx in fold_ids if len(idx) > 0]


def evaluate_reduced_vs_null(y, x_lookup, n_folds=10, max_iter=80, pval_threshold=0.01, n_jobs=1):
    folds = make_folds(y.shape[0], n_folds)
    fold_results = []
    fit_times = []
    worker_count = resolve_job_count(n_jobs, len(folds))

    if worker_count == 1:
        for fold_idx, test_idx in enumerate(folds):
            fold_result = evaluate_single_fold(fold_idx, test_idx, y, x_lookup, max_iter, pval_threshold)
            fold_results.append(fold_result)
            fit_times.append(fold_result["fit_time_s"])
    else:
        with get_parallel_executor(worker_count) as executor:
            futures = [
                executor.submit(evaluate_single_fold, fold_idx, test_idx, y, x_lookup, max_iter, pval_threshold)
                for fold_idx, test_idx in enumerate(folds)
            ]
            for future in as_completed(futures):
                fold_result = future.result()
                fold_results.append(fold_result)
                fit_times.append(fold_result["fit_time_s"])

    fold_results.sort(key=lambda row: row["fold"])

    return {
        "folds": fold_results,
        "mean_delta_ll": float(np.mean([row["delta_ll"] for row in fold_results])),
        "mean_bits_per_spike": float(np.mean([row["bits_per_spike"] for row in fold_results])),
        "mean_fit_time_s": float(np.mean(fit_times)),
    }


def run_test_1_against_constant_rate(y, x_lookup, n_folds, max_iter, pval_threshold, n_jobs):
    return evaluate_reduced_vs_null(
        y,
        x_lookup,
        n_folds=n_folds,
        max_iter=max_iter,
        pval_threshold=pval_threshold,
        n_jobs=n_jobs,
    )


def evaluate_single_shuffle(shuffle_idx, shift, y, x_lookup, n_folds, max_iter, pval_threshold):
    y_shift = circular_shift(y, shift)
    shuffled = evaluate_reduced_vs_null(
        y_shift,
        x_lookup,
        n_folds=n_folds,
        max_iter=max_iter,
        pval_threshold=pval_threshold,
        n_jobs=1,
    )
    return {
        "shuffle_idx": shuffle_idx,
        "shift": shift,
        "mean_delta_ll": shuffled["mean_delta_ll"],
        "mean_bits_per_spike": shuffled["mean_bits_per_spike"],
    }


def shuffle_test(y, x_lookup, n_folds, n_shuffles, max_iter, pval_threshold, rng, n_jobs=1):
    if n_shuffles <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    null_delta_ll = []
    null_bits = []
    min_shift = max(1, y.shape[0] // 20)
    shifts = [int(rng.integers(min_shift, y.shape[0] - min_shift)) for _ in range(n_shuffles)]
    worker_count = resolve_job_count(n_jobs, n_shuffles)
    shuffle_rows = []

    if worker_count == 1:
        for shuffle_idx, shift in enumerate(shifts):
            row = evaluate_single_shuffle(shuffle_idx, shift, y, x_lookup, n_folds, max_iter, pval_threshold)
            shuffle_rows.append(row)
    else:
        with get_parallel_executor(worker_count) as executor:
            futures = [
                executor.submit(
                    evaluate_single_shuffle,
                    shuffle_idx,
                    shift,
                    y,
                    x_lookup,
                    n_folds,
                    max_iter,
                    pval_threshold,
                )
                for shuffle_idx, shift in enumerate(shifts)
            ]
            for future in as_completed(futures):
                shuffle_rows.append(future.result())

    shuffle_rows.sort(key=lambda row: row["shuffle_idx"])
    for row in shuffle_rows:
        null_delta_ll.append(row["mean_delta_ll"])
        null_bits.append(row["mean_bits_per_spike"])
        print(
            f"shuffle {row['shuffle_idx'] + 1}/{n_shuffles}: shift={row['shift']}, "
            f"mean delta LL={row['mean_delta_ll']:.3f}, bits/spike={row['mean_bits_per_spike']:.6f}"
        )

    return np.asarray(null_delta_ll, dtype=float), np.asarray(null_bits, dtype=float)


def run_test_2_circular_shuffle(y, x_lookup, n_folds, n_shuffles, max_iter, pval_threshold, rng, n_jobs):
    shuffle_delta_ll, shuffle_bits = shuffle_test(
        y,
        x_lookup,
        n_folds=n_folds,
        n_shuffles=n_shuffles,
        max_iter=max_iter,
        pval_threshold=pval_threshold,
        rng=rng,
        n_jobs=n_jobs,
    )
    return {
        "delta_ll": shuffle_delta_ll,
        "bits_per_spike": shuffle_bits,
    }


def empirical_pvalue(null_values, observed_value):
    if null_values.size == 0:
        return None
    return float((1 + np.sum(null_values >= observed_value)) / (null_values.size + 1))


def save_results(results, out_path):
    out_path.write_text(json.dumps(results, indent=2))
    return out_path


def smooth_map(x, y, weights, dt, mask=None, x_bins=24, y_bins=12, sigma=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if mask is None:
        mask = np.ones(x.shape[0], dtype=bool)

    x_edges = np.linspace(0.0, UNFOLDED_U_MAX_M, x_bins + 1)
    y_edges = np.linspace(0.0, UNFOLDED_V_MAX_M, y_bins + 1)
    occ, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
    val, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges], weights=weights[mask])
    occ = gaussian_filter(occ.astype(float), sigma=sigma, mode=("wrap", "reflect"))
    val = gaussian_filter(val.astype(float), sigma=sigma, mode=("wrap", "reflect"))
    rate = val / np.maximum(occ * dt, 1e-12)
    rate[occ < 1.0] = np.nan
    return rate, x_edges, y_edges


def smooth_theta_curve(theta, weights, dt, n_az=24, n_pitch=18, sigma=1.0):
    theta = np.asarray(theta, dtype=float)
    weights = np.asarray(weights, dtype=float)
    az = theta
    pitch = np.zeros_like(theta)
    az_edges = np.linspace(-np.pi, np.pi, n_az + 1)
    pitch_edges = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_pitch + 1)
    occ, _, _ = np.histogram2d(az, pitch, bins=[az_edges, pitch_edges])
    val, _, _ = np.histogram2d(az, pitch, bins=[az_edges, pitch_edges], weights=weights)
    occ = gaussian_filter(occ.astype(float), sigma=sigma, mode=("wrap", "reflect"))
    val = gaussian_filter(val.astype(float), sigma=sigma, mode=("wrap", "reflect"))
    rate = val / np.maximum(occ * dt, 1e-12)
    rate[occ < 1.0] = np.nan
    return rate, az_edges, pitch_edges


def compute_plot_statistics(data, y, x_lookup, full_fit, observed_cv, shuffle_delta_ll, shuffle_bits):
    dt = 1.0 / 250.0
    move_fast = np.asarray(x_lookup["move_fast"][0], dtype=bool)
    mu_all = full_fit["mu_all"]

    self_obs, self_x_edges, self_y_edges = smooth_map(
        x_lookup["self_xy_fast"][0], x_lookup["self_xy_fast"][1], y, dt, mask=move_fast
    )
    self_pred, _, _ = smooth_map(
        x_lookup["self_xy_fast"][0], x_lookup["self_xy_fast"][1], mu_all, dt, mask=move_fast
    )
    other_obs, other_x_edges, other_y_edges = smooth_map(
        x_lookup["other_xy"][0], x_lookup["other_xy"][1], y, dt
    )
    other_pred, _, _ = smooth_map(
        x_lookup["other_xy"][0], x_lookup["other_xy"][1], mu_all, dt
    )
    view_obs, view_x_edges, view_y_edges = smooth_map(
        x_lookup["view_xy"][0], x_lookup["view_xy"][1], y, dt
    )
    view_pred, _, _ = smooth_map(
        x_lookup["view_xy"][0], x_lookup["view_xy"][1], mu_all, dt
    )
    theta_obs, theta_x_edges, theta_y_edges = smooth_theta_curve(x_lookup["theta"][0], y, dt)
    theta_pred, _, _ = smooth_theta_curve(x_lookup["theta"][0], mu_all, dt)

    fold_delta_ll = np.array([row["delta_ll"] for row in observed_cv["folds"]], dtype=float)
    fold_bits = np.array([row["bits_per_spike"] for row in observed_cv["folds"]], dtype=float)

    return {
        "cov_pvals": full_fit["cov_pvals"],
        "self_obs": self_obs,
        "self_pred": self_pred,
        "self_x_edges": self_x_edges,
        "self_y_edges": self_y_edges,
        "other_obs": other_obs,
        "other_pred": other_pred,
        "other_x_edges": other_x_edges,
        "other_y_edges": other_y_edges,
        "view_obs": view_obs,
        "view_pred": view_pred,
        "view_x_edges": view_x_edges,
        "view_y_edges": view_y_edges,
        "theta_obs": theta_obs,
        "theta_pred": theta_pred,
        "theta_x_edges": theta_x_edges,
        "theta_y_edges": theta_y_edges,
        "fold_delta_ll": fold_delta_ll,
        "fold_bits": fold_bits,
        "shuffle_delta_ll": np.asarray(shuffle_delta_ll, dtype=float),
        "shuffle_bits": np.asarray(shuffle_bits, dtype=float),
        "observed_mean_delta_ll": observed_cv["mean_delta_ll"],
        "observed_mean_bits": observed_cv["mean_bits_per_spike"],
    }


def add_map_panel(ax, rate_map, x_edges, y_edges, title, xlabel, ylabel, cmap="jet"):
    vmax = float(np.nanmax(rate_map)) if np.any(np.isfinite(rate_map)) else 1.0
    vmax = max(vmax, 1e-6)
    ax.imshow(
        rate_map.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap=cmap,
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


def add_hist_panel(ax, values, observed_value, title, xlabel, zero_label=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        ax.text(0.5, 0.5, "No shuffle samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return
    ax.hist(values, bins=min(20, max(5, values.size // 2)), color="0.75", edgecolor="0.35")
    ax.axvline(observed_value, color="#d62728", lw=2.0, label="Observed mean")
    if zero_label is not None:
        ax.axvline(0.0, color="black", lw=1.0, ls="--", label=zero_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_summary_figure(plot_stats, out_path):
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.9])
    pvals = plot_stats["cov_pvals"]

    ax = fig.add_subplot(gs[0, 0])
    add_map_panel(ax, plot_stats["self_obs"], plot_stats["self_x_edges"], plot_stats["self_y_edges"], "Observed self map | moving", "u (m)", "v (m)")
    ax = fig.add_subplot(gs[0, 1])
    add_map_panel(
        ax,
        plot_stats["self_pred"],
        plot_stats["self_x_edges"],
        plot_stats["self_y_edges"],
        f"Model self map | moving\np={pvals.get('self_xy_fast', np.nan):.1e}",
        "u (m)",
        "v (m)",
    )
    ax = fig.add_subplot(gs[0, 2])
    add_map_panel(ax, plot_stats["other_obs"], plot_stats["other_x_edges"], plot_stats["other_y_edges"], "Observed other map", "u (m)", "v (m)")
    ax = fig.add_subplot(gs[0, 3])
    add_map_panel(
        ax,
        plot_stats["other_pred"],
        plot_stats["other_x_edges"],
        plot_stats["other_y_edges"],
        f"Model other map\np={pvals.get('other_xy', np.nan):.1e}",
        "u (m)",
        "v (m)",
    )

    ax = fig.add_subplot(gs[1, 0])
    add_map_panel(ax, plot_stats["view_obs"], plot_stats["view_x_edges"], plot_stats["view_y_edges"], "Observed spatial view", "u (m)", "v (m)")
    ax = fig.add_subplot(gs[1, 1])
    add_map_panel(
        ax,
        plot_stats["view_pred"],
        plot_stats["view_x_edges"],
        plot_stats["view_y_edges"],
        f"Model spatial view\np={pvals.get('view_xy', np.nan):.1e}",
        "u (m)",
        "v (m)",
    )
    ax = fig.add_subplot(gs[1, 2])
    add_map_panel(ax, plot_stats["theta_obs"], plot_stats["theta_x_edges"], plot_stats["theta_y_edges"], "Observed head direction", "Azimuth (rad)", "Pitch (rad)")
    ax = fig.add_subplot(gs[1, 3])
    add_map_panel(
        ax,
        plot_stats["theta_pred"],
        plot_stats["theta_x_edges"],
        plot_stats["theta_y_edges"],
        f"Model head direction\np={pvals.get('theta', np.nan):.1e}",
        "Azimuth (rad)",
        "Pitch (rad)",
    )

    ax = fig.add_subplot(gs[2, 0])
    add_hist_panel(
        ax,
        plot_stats["fold_delta_ll"],
        plot_stats["observed_mean_delta_ll"],
        "CV delta log likelihood vs constant",
        "Delta log likelihood",
        zero_label="Constant-rate null",
    )
    ax = fig.add_subplot(gs[2, 1])
    add_hist_panel(
        ax,
        plot_stats["fold_bits"],
        plot_stats["observed_mean_bits"],
        "CV bits/spike vs constant",
        "Bits/spike",
        zero_label="Constant-rate null",
    )
    ax = fig.add_subplot(gs[2, 2])
    add_hist_panel(
        ax,
        plot_stats["shuffle_delta_ll"],
        plot_stats["observed_mean_delta_ll"],
        "Circular shuffle null: delta log likelihood",
        "Mean delta log likelihood",
    )
    ax = fig.add_subplot(gs[2, 3])
    add_hist_panel(
        ax,
        plot_stats["shuffle_bits"],
        plot_stats["observed_mean_bits"],
        "Circular shuffle null: bits/spike",
        "Mean bits/spike",
    )

    fig.suptitle("PGAM fit: observed maps, model smooths, and likelihood controls", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Cross-validated PGAM analysis for simulated social-wall data.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--pval-threshold", type=float, default=0.01)
    parser.add_argument("--fold-jobs", type=int, default=1)
    parser.add_argument("--shuffle-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--png", type=Path, default=SUMMARY_PNG_PATH)
    args = parser.parse_args()

    data = np.load(args.data)
    y, x_lookup = build_covariates(data)

    full_fit = run_initial_model_fit(
        y,
        x_lookup,
        max_iter=args.max_iter,
        pval_threshold=args.pval_threshold,
    )
    print(
        f"full-data fit: {full_fit['fit_elapsed_s']:.2f} s, "
        f"kept vars={list(full_fit['reduced_model'].var_list) if full_fit['reduced_model'] is not None else []}"
    )

    observed = run_test_1_against_constant_rate(
        y,
        x_lookup,
        n_folds=args.folds,
        max_iter=args.max_iter,
        pval_threshold=args.pval_threshold,
        n_jobs=args.fold_jobs,
    )
    print(
        f"test 1: mean delta LL={observed['mean_delta_ll']:.3f}, "
        f"mean bits/spike={observed['mean_bits_per_spike']:.6f}, "
        f"mean fit time/fold={observed['mean_fit_time_s']:.2f} s"
    )

    rng = np.random.default_rng(args.seed)
    shuffle_results = run_test_2_circular_shuffle(
        y,
        x_lookup,
        n_folds=args.folds,
        n_shuffles=args.shuffles,
        max_iter=args.max_iter,
        pval_threshold=args.pval_threshold,
        rng=rng,
        n_jobs=args.shuffle_jobs,
    )
    shuffle_delta_ll = shuffle_results["delta_ll"]
    shuffle_bits = shuffle_results["bits_per_spike"]

    p_delta_ll = empirical_pvalue(shuffle_delta_ll, observed["mean_delta_ll"])
    p_bits = empirical_pvalue(shuffle_bits, observed["mean_bits_per_spike"])
    plot_stats = compute_plot_statistics(data, y, x_lookup, full_fit, observed, shuffle_delta_ll, shuffle_bits)
    png_path = plot_summary_figure(plot_stats, args.png)

    results = {
        "data_path": str(args.data),
        "model_formula": (
            "log lambda_t = beta_0 + beta_m m_t + m_t*f_self(x_t^s,y_t^s) + "
            "f_other(x_t^o,y_t^o) + f_theta(theta_t) + f_view(x_t^view,y_t^view) + log(Delta)"
        ),
        "offset_note": (
            "Delta is constant after binning spikes to behavior bins, so log(Delta) is "
            "absorbed into the intercept when fitting Poisson counts per bin."
        ),
        "folds": args.folds,
        "shuffles": args.shuffles,
        "max_iter": args.max_iter,
        "pval_threshold": args.pval_threshold,
        "fold_jobs": args.fold_jobs,
        "shuffle_jobs": args.shuffle_jobs,
        "summary_png": str(png_path),
        "full_model_fit_time_s": full_fit["fit_elapsed_s"],
        "full_model_p_values": full_fit["cov_pvals"],
        "observed": observed,
        "shuffle": {
            "delta_ll": shuffle_delta_ll.tolist(),
            "bits_per_spike": shuffle_bits.tolist(),
            "mean_delta_ll": None if shuffle_delta_ll.size == 0 else float(np.mean(shuffle_delta_ll)),
            "mean_bits_per_spike": None if shuffle_bits.size == 0 else float(np.mean(shuffle_bits)),
            "p_delta_ll": p_delta_ll,
            "p_bits_per_spike": p_bits,
        },
    }

    out_path = save_results(results, args.out)
    print(f"saved results: {out_path}")
    print(f"saved figure: {png_path}")
    if p_delta_ll is not None:
        print(f"p(delta LL): {p_delta_ll:.6f}")
        print(f"p(bits/spike): {p_bits:.6f}")


if __name__ == "__main__":
    main()
