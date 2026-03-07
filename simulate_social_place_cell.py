import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import spearmanr

from PGAM.GAM_library import general_additive_model
from PGAM.gam_data_handlers import smooths_handler


def maybe_show() -> None:
    if matplotlib.get_backend().lower() != "agg":
        plt.show(block=False)
        plt.pause(0.001)


def place_rate(x: np.ndarray, y: np.ndarray, center, sd, peak: float) -> np.ndarray:
    return peak * np.exp(-0.5 * ((x - center[0]) / sd[0]) ** 2 - 0.5 * ((y - center[1]) / sd[1]) ** 2)


def simulate_trajectories(rng: np.random.Generator, n: int, vel_corr: float = 0.7):
    pos1 = np.zeros((n, 2))
    pos2 = np.zeros((n, 2))
    pos1[0] = np.array([0.22, 0.25])
    pos2[0] = np.array([0.78, 0.72])
    v1 = rng.normal(0, 0.015, size=(n, 2)) * 3
    noise2 = rng.normal(0, 0.015, size=(n, 2)) * 3
    mix = np.sqrt(max(0.0, 1.0 - vel_corr**2))
    v2 = vel_corr * v1 + mix * noise2
    desired_offset = pos2[0] - pos1[0]
    offset_strength = 0.008
    center = np.array([0.5, 0.5])
    center_pull = 0.0

    for t in range(1, n):
        # Keep headings aligned, but softly preserve the initial spatial offset.
        offset_error = (pos1[t - 1] + desired_offset) - pos2[t - 1]
        v2[t] += offset_strength * offset_error
        v1[t] += center_pull * (center - pos1[t - 1])
        v2[t] += center_pull * (center - pos2[t - 1])
        pos1[t] = pos1[t - 1] + v1[t]
        pos2[t] = pos2[t - 1] + v2[t]
        for d, (lo, hi) in enumerate([(0, 1), (0, 1)]):
            if pos1[t, d] < lo:
                pos1[t, d] = 2 * lo - pos1[t, d]
                v1[t, d] *= -1
            if pos1[t, d] > hi:
                pos1[t, d] = 2 * hi - pos1[t, d]
                v1[t, d] *= -1
            if pos2[t, d] < lo:
                pos2[t, d] = 2 * lo - pos2[t, d]
                v2[t, d] *= -1
            if pos2[t, d] > hi:
                pos2[t, d] = 2 * hi - pos2[t, d]
                v2[t, d] *= -1

    return pos1, pos2, v1, v2


def compute_firing_map(pos: np.ndarray, y: np.ndarray, dt: float, nbins: int = 24):
    edges = np.linspace(0, 1, nbins + 1)
    occ, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges, edges])
    spikes, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges, edges], weights=y)

    occ = gaussian_filter(np.nan_to_num(occ), sigma=1.3, mode="reflect", cval=0)
    spikes = gaussian_filter(np.nan_to_num(spikes), sigma=1.3, mode="reflect", cval=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = spikes / np.maximum(occ * dt, 1e-12)
    rate[occ < 1.0] = np.nan
    return rate, edges


def make_stratified_test_mask(
    rng: np.random.Generator,
    pos1: np.ndarray,
    pos2: np.ndarray,
    test_frac: float = 0.2,
    nbins: int = 6,
) -> np.ndarray:
    edges = np.linspace(0, 1, nbins + 1)
    bins = np.clip(
        np.column_stack(
            [
                np.digitize(pos1[:, 0], edges) - 1,
                np.digitize(pos1[:, 1], edges) - 1,
                np.digitize(pos2[:, 0], edges) - 1,
                np.digitize(pos2[:, 1], edges) - 1,
            ]
        ),
        0,
        nbins - 1,
    )
    ids = bins[:, 0] + nbins * bins[:, 1] + (nbins**2) * bins[:, 2] + (nbins**3) * bins[:, 3]

    test_mask = np.zeros(pos1.shape[0], dtype=bool)
    for bid in np.unique(ids):
        idx = np.where(ids == bid)[0]
        if idx.size <= 1:
            continue
        n_test = max(1, int(round(test_frac * idx.size)))
        n_test = min(n_test, idx.size - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        test_mask[test_idx] = True
    return test_mask


def fit_pgam_full_model(y: np.ndarray, pos1: np.ndarray, pos2: np.ndarray, train_mask: np.ndarray, knots_num: int = 8):
    sm_handler = smooths_handler()
    sm_handler.add_smooth(
        "self_xy",
        [pos1[:, 0], pos1[:, 1]],
        knots_num=knots_num,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[False, False],
        knots_percentiles=(0, 100),
    )
    sm_handler.add_smooth(
        "other_xy",
        [pos2[:, 0], pos2[:, 1]],
        knots_num=knots_num,
        ord=4,
        penalty_type="der",
        der=2,
        is_cyclic=[False, False],
        knots_percentiles=(0, 100),
    )

    link = sm.genmod.families.links.log()
    poiss_family = sm.genmod.families.family.Poisson(link=link)
    pgam = general_additive_model(sm_handler, sm_handler.smooths_var, y, poiss_family)

    full, reduced = pgam.fit_full_and_reduced(
        sm_handler.smooths_var,
        th_pval=0.001,
        max_iter=10**2,
        use_dgcv=True,
        filter_trials=train_mask,
        fit_initial_beta=True,
    )
    return full, reduced


def predict_counts(model, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    x_list = []
    if "self_xy" in model.var_list:
        x_list.append([pos1[:, 0], pos1[:, 1]])
    if "other_xy" in model.var_list:
        x_list.append([pos2[:, 0], pos2[:, 1]])
    mu = model.predict(x_list, var_list=model.var_list)
    return np.clip(mu, 1e-12, None)


def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sum(y * np.log(mu) - mu))


def recovered_map(
    model,
    which: str,
    pos1: np.ndarray,
    pos2: np.ndarray,
    res: int = 40,
    marginal_samples: int = 400,
):
    xs = np.linspace(0, 1, res)
    ys = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n = pos1.shape[0]
    if n > marginal_samples:
        sample_idx = np.linspace(0, n - 1, marginal_samples, dtype=int)
    else:
        sample_idx = np.arange(n)

    if which == "self":
        pred = np.zeros(x_flat.shape[0], dtype=float)
        for idx in sample_idx:
            pred += model.predict(
                [[x_flat, y_flat], [np.full_like(x_flat, pos2[idx, 0]), np.full_like(y_flat, pos2[idx, 1])]],
                var_list=["self_xy", "other_xy"],
            )
        pred /= sample_idx.size
    elif which == "other":
        pred = np.zeros(x_flat.shape[0], dtype=float)
        for idx in sample_idx:
            pred += model.predict(
                [[np.full_like(x_flat, pos1[idx, 0]), np.full_like(y_flat, pos1[idx, 1])], [x_flat, y_flat]],
                var_list=["self_xy", "other_xy"],
            )
        pred /= sample_idx.size
    else:
        raise ValueError("which must be 'self' or 'other'")

    return xs, ys, pred.reshape(res, res)


def analyze_and_plot_cell(
    label: str,
    y: np.ndarray,
    pos1: np.ndarray,
    pos2: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    pf_center_self,
    pf_center_other,
    out_path: str,
) -> None:
    n = y.shape[0]
    test_mask = make_stratified_test_mask(rng, pos1, pos2, test_frac=0.2, nbins=6)
    train_mask = ~test_mask

    model_full, model_reduced = fit_pgam_full_model(y, pos1, pos2, train_mask=train_mask)
    mu_full_test = predict_counts(model_full, pos1[test_mask], pos2[test_mask])

    y_test = y[test_mask]
    null_rate = y[train_mask].mean()
    mu_null = np.full_like(y_test, fill_value=max(null_rate, 1e-12), dtype=float)
    ll_null = poisson_loglik(y_test, mu_null)
    ll_full = poisson_loglik(y_test, mu_full_test)
    r2_full = 1 - ll_full / ll_null

    cov_pvals = {row["covariate"]: row["p-val"] for row in model_full.covariate_significance}
    p_self = float(cov_pvals.get("self_xy", np.nan))
    p_other = float(cov_pvals.get("other_xy", np.nan))

    print(f"\n=== {label} ===")
    if model_reduced is not None:
        print("Reduced model kept:", list(model_reduced.var_list))
    print(f"Train pseudo-R2(full): {model_full.pseudo_r2:.3f}")
    print(f"Held-out pseudo-R2(full): {r2_full:.3f}")
    print(f"covariate p-values: self_xy={p_self:.2e}, other_xy={p_other:.2e}")

    mu_full_all = predict_counts(model_full, pos1, pos2)
    t_axis = np.arange(n) * dt
    observed_rate_hz = gaussian_filter1d(y / dt, sigma=2)
    predicted_rate_hz = gaussian_filter1d(mu_full_all / dt, sigma=2)
    rho_s, p_s = spearmanr(observed_rate_hz, predicted_rate_hz)

    firing_self, edges_self = compute_firing_map(pos1, y, dt)
    firing_other, edges_other = compute_firing_map(pos2, y, dt)
    model_map_self, _ = compute_firing_map(pos1, mu_full_all, dt)
    model_map_other, _ = compute_firing_map(pos2, mu_full_all, dt)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], height_ratios=[1.0, 1.0, 0.58])
    cmap = "viridis"

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(pos1[:, 0] * 100, pos1[:, 1] * 100, color="0.82", lw=0.6, zorder=1)
    ax.scatter(pos1[y > 0, 0] * 100, pos1[y > 0, 1] * 100, s=6, color="#d62728", alpha=0.75, zorder=2)
    if pf_center_self is not None:
        ax.scatter(pf_center_self[0] * 100, pf_center_self[1] * 100, marker="*", s=90, color="black", zorder=3)
    ax.set_title("Animal 1 - Trajectory & Spikes")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(
        firing_self.T,
        origin="lower",
        extent=[edges_self[0] * 100, edges_self[-1] * 100, edges_self[0] * 100, edges_self[-1] * 100],
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title("Animal 1 - Firing map")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[0, 2])
    im_rec_1 = ax.imshow(
        model_map_self.T,
        origin="lower",
        extent=[edges_self[0] * 100, edges_self[-1] * 100, edges_self[0] * 100, edges_self[-1] * 100],
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title(f"Animal 1 - Model firing map\n(p={p_self:.1e})")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im_rec_1, cax=cax, label="Rate (Hz)")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(pos2[:, 0] * 100, pos2[:, 1] * 100, color="0.82", lw=0.6, zorder=1)
    ax.scatter(pos2[y > 0, 0] * 100, pos2[y > 0, 1] * 100, s=6, color="#d62728", alpha=0.75, zorder=2)
    if pf_center_other is not None:
        ax.scatter(pf_center_other[0] * 100, pf_center_other[1] * 100, marker="*", s=90, color="black", zorder=3)
    ax.set_title("Animal 2 - Trajectory & Spikes")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(
        firing_other.T,
        origin="lower",
        extent=[edges_other[0] * 100, edges_other[-1] * 100, edges_other[0] * 100, edges_other[-1] * 100],
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title("Animal 2 - Firing map")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")

    ax = fig.add_subplot(gs[1, 2])
    im_rec_2 = ax.imshow(
        model_map_other.T,
        origin="lower",
        extent=[edges_other[0] * 100, edges_other[-1] * 100, edges_other[0] * 100, edges_other[-1] * 100],
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title(f"Animal 2 - Model firing map\n(p={p_other:.1e})")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")
    cax = fig.add_subplot(gs[1, 3])
    fig.colorbar(im_rec_2, cax=cax, label="Rate (Hz)")

    ax = fig.add_subplot(gs[2, :3])
    ax.plot(t_axis, observed_rate_hz, color="0.45", lw=0.9, label="Observed (smoothed)")
    ax.plot(t_axis, predicted_rate_hz, color="#d62728", lw=0.9, alpha=0.7, label="PGAM predicted")
    ax.set_title("Observed vs predicted rate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.text(
        0.01,
        0.98,
        f"Spearman r={rho_s:.3f}, p={p_s:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"Two-Animal Random Walk - PGAM Neural Tuning Recovery ({label})", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    maybe_show()
    print(f"Spearman corr (observed vs predicted): r={rho_s:.3f}, p={p_s:.2e}")
    print(f"Saved: {out_path}")


def main() -> None:
    rng = np.random.default_rng(0)
    t_sec = 200.0
    fs = 50
    dt = 1.0 / fs
    n = int(t_sec * fs)

    # Correlated movement by design.
    pos1, pos2, v1, v2 = simulate_trajectories(rng, n, vel_corr=0.72)
    self_speed = np.linalg.norm(v1, axis=1)
    speed_threshold = np.quantile(self_speed, 0.45)
    fast_self = (self_speed >= speed_threshold).astype(float)

    # Cell 1: tuned to self and other positions.
    center1 = [0.2, 0.2]
    center2 = [0.8, 0.8]
    sd1 = [0.15, 0.15]
    sd2 = [0.15, 0.15]
    peak1 = 12.0
    peak2 = 12.0
    lamb1 = place_rate(pos1[:, 0], pos1[:, 1], center1, sd1, peak1)
    lamb2 = place_rate(pos2[:, 0], pos2[:, 1], center2, sd2, peak2)
    # Self-position tuning is active only above the self-speed threshold.
    # Other-position tuning is always active.
    y_cell1 = rng.poisson((lamb1 * fast_self + lamb2) * dt)

    # Cell 2: self-position tuning + direction-alignment modulation.
    # No direct tuning to other position.
    center1_b = [0.68, 0.32]
    sd1_b = [0.13, 0.13]
    peak1_b = 13.0
    self_drive_b = place_rate(pos1[:, 0], pos1[:, 1], center1_b, sd1_b, peak1_b)

    speed1 = np.linalg.norm(v1, axis=1)
    speed2 = np.linalg.norm(v2, axis=1)
    valid = (speed1 > 1e-6) & (speed2 > 1e-6)
    align = np.zeros(n, dtype=float)
    align[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (speed1[valid] * speed2[valid])
    align = np.clip(align, -1.0, 1.0)
    align = gaussian_filter1d(align, sigma=2)
    align_pos = np.clip(align, 0.0, 1.0)
    move_gain = 1.0 + 0.55 * align_pos

    # Cell 2 self-position tuning is also active only above the self-speed threshold.
    lamb_cell2 = 0.5 + self_drive_b * move_gain * fast_self
    y_cell2 = rng.poisson(lamb_cell2 * dt)

    print(f"Self-speed threshold: {speed_threshold:.3f}")
    print(f"Fraction above threshold: {fast_self.mean():.3f}")

    analyze_and_plot_cell(
        label="Cell 1: self+other position",
        y=y_cell1,
        pos1=pos1,
        pos2=pos2,
        dt=dt,
        rng=rng,
        pf_center_self=center1,
        pf_center_other=center2,
        out_path="simulate_social_place_cell_pgam_summary.png",
    )

    analyze_and_plot_cell(
        label="Cell 2: self position + direction alignment",
        y=y_cell2,
        pos1=pos1,
        pos2=pos2,
        dt=dt,
        rng=rng,
        pf_center_self=center1_b,
        pf_center_other=None,
        out_path="simulate_social_place_cell_pgam_summary_cell2.png",
    )


if __name__ == "__main__":
    main()
