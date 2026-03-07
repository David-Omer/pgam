

import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import matplotlib
from pygam import PoissonGAM, s, te
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def maybe_show():
    if matplotlib.get_backend().lower() != "agg":
        plt.show(block=False)
        plt.pause(0.001)

rng = np.random.default_rng(0)
T   = 200.0                        # seconds
fs  = 50                          # Hz (binning at 20 ms)
dt  = 1.0 / fs
N   = int(T * fs)


pos1 = np.zeros((N, 2))
pos2 = np.zeros((N, 2))
v1   = rng.normal(0, 0.015, size=(N, 2))*3
#noise   = rng.normal(0, 0.015, size=(N, 2))
#v2  = v1 + noise
v2   = rng.normal(0, 0.015, size=(N, 2))*3

for t in range(1, N):
    pos1[t] = pos1[t-1] + v1[t]
    for d, (lo, hi) in enumerate([(0,1), (0,1)]):
        if pos1[t, d] < lo:
            pos1[t, d] = 2*lo - pos1[t, d]; v1[t, d] *= -1
        if pos1[t, d] > hi:
            pos1[t, d] = 2*hi - pos1[t, d]; v1[t, d] *= -1
    pos2[t] = pos2[t-1] + v2[t]
    for d, (lo, hi) in enumerate([(0, 1), (0, 1)]):
        if pos2[t, d] < lo:
            pos2[t, d] = 2 * lo - pos2[t, d]; v2[t, d] *= -1
        if pos2[t, d] > hi:
            pos2[t, d] = 2 * hi - pos2[t, d]; v2[t, d] *= -1

def place_rate(x, y,center, sd, peak):  return peak * np.exp(-0.5*((x-center[0])/sd[0])**2 - 0.5*((y-center[1])/sd[1])**2)

lamb1 = place_rate(pos1[:,0], pos1[:,1],[0.2, 0.2],[0.1, 0.15], 12.0)
lamb2 = place_rate(pos2[:,0], pos2[:,1],[0.8, 0.8],[0.15, 0.1],12.0)
y = rng.poisson((lamb1 + lamb2) * dt)
y1 = rng.poisson(lamb1 * dt)
y2 = rng.poisson(lamb2 * dt)

def compute_firing_map(pos, y, nbins=20):
    xedges = np.linspace(0,1,nbins+1)
    yedges = np.linspace(0,1,nbins+1)
    occ, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[xedges, yedges])
    spikes, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[xedges, yedges], weights=y)

    occ = np.nan_to_num(occ)
    spikes = np.nan_to_num(spikes)
    spikes = gaussian_filter(spikes, sigma=2, mode='reflect', cval=0)
    occ = gaussian_filter(occ, sigma=2, mode='reflect', cval=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        rate = spikes / (occ * dt)
    rate[occ < 1] = np.nan  # mask out bins with poor/no occupancy
    return rate, occ, xedges, yedges

def compute_binned_design_2d(pos, y, nbins, dt):
    edges = np.linspace(0, 1, nbins + 1)
    occ, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges, edges])
    spikes, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges, edges], weights=y)
    centers = (edges[:-1] + edges[1:]) / 2.0
    XX, YY = np.meshgrid(centers, centers, indexing="ij")
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    exposure = (occ * dt).ravel()
    y_bins = spikes.ravel()
    valid = exposure > 0
    return grid[valid], y_bins[valid], exposure[valid], edges

def compute_binned_design_4d(pos1, pos2, y, nbins, dt):
    edges = np.linspace(0, 1, nbins + 1)
    samples = np.column_stack([pos1[:, 0], pos1[:, 1], pos2[:, 0], pos2[:, 1]])
    occ = np.histogramdd(samples, bins=[edges, edges, edges, edges])[0]
    spikes = np.histogramdd(samples, bins=[edges, edges, edges, edges], weights=y)[0]
    centers = (edges[:-1] + edges[1:]) / 2.0
    grid = np.meshgrid(centers, centers, centers, centers, indexing="ij")
    grid = np.column_stack([g.ravel() for g in grid])
    exposure = (occ * dt).ravel()
    y_bins = spikes.ravel()
    valid = exposure > 0
    return grid[valid], y_bins[valid], exposure[valid], edges

def compute_binned_design_4d_with_edges(pos1, pos2, y, edges, dt):
    samples = np.column_stack([pos1[:, 0], pos1[:, 1], pos2[:, 0], pos2[:, 1]])
    occ = np.histogramdd(samples, bins=[edges, edges, edges, edges])[0]
    spikes = np.histogramdd(samples, bins=[edges, edges, edges, edges], weights=y)[0]
    centers = (edges[:-1] + edges[1:]) / 2.0
    grid = np.meshgrid(centers, centers, centers, centers, indexing="ij")
    grid = np.column_stack([g.ravel() for g in grid])
    exposure = (occ * dt).ravel()
    y_bins = spikes.ravel()
    valid = exposure > 0
    return grid[valid], y_bins[valid], exposure[valid]

rate1, occ1, xe1, ye1 = compute_firing_map(pos1, y, nbins=20)
rate2, occ2, xe2, ye2 = compute_firing_map(pos2, y, nbins=20)


fig1, ax = plt.subplots(2, 2, figsize=(6,6))
ax[0,0].plot(pos1[:,0], pos1[:,1], color='0.8', lw=1, zorder=1)
ax[0,0].scatter(pos1[y1>0,0], pos1[y1>0,1], s=8, color='red', label='spikes self',zorder=2)
ax[0,0].scatter(pos2[y2>0,0], pos2[y2>0,1], s=8, color='blue', label='spikes other',zorder=2)
ax[0,0].set_xlim(0,1);
ax[0,0].set_ylim(0,1)
ax[0,0].set_xlabel('x'); plt.ylabel('y')
ax[0,0].set_title('self trajectory and spikes')
ax[0,0].set_aspect('equal', 'box')

im1 = ax[0,1].imshow(rate1.T, origin='lower', extent=[xe1[0], xe1[-1], ye1[0], ye1[-1]], aspect='equal')
fig1.colorbar(im1, ax=ax[0,1], label='Firing rate (Hz)')
ax[0,1].set_title('self firing map')
ax[0,1].set_xlabel('x'); plt.ylabel('y')

im2= ax[1,0].imshow(occ1.T * dt, origin='lower', extent=[xe1[0], xe1[-1], ye1[0], ye1[-1]], aspect='equal')
fig1.colorbar(im2, ax=ax[1,0], label='Occupancy (s)')
ax[1,0].set_title('self Occupancy map')
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')
ax[1,1].axis('off')
fig1.tight_layout()
maybe_show()



fig2, ax = plt.subplots(2, 2, figsize=(6,6))
ax[0,0].plot(pos2[:,0], pos2[:,1], color='0.8', lw=1, zorder=1)
ax[0,0].scatter(pos1[y1>0,0], pos1[y1>0,1], s=8, color='red', label='spikes self',zorder=2)
ax[0,0].scatter(pos2[y2>0,0], pos2[y2>0,1], s=8, color='blue', label='spikes other',zorder=2)
ax[0,0].set_xlim(0,1)
ax[0,0].set_ylim(0,1)
ax[0,0].set_xlabel('x'); plt.ylabel('y')
ax[0,0].set_title('other trajectory and spikes')
ax[0,0].set_aspect('equal', 'box')

im1 = ax[0,1].imshow(rate2.T, origin='lower', extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]], aspect='equal')
fig2.colorbar(im1, ax=ax[0,1], label='Firing rate (Hz)')
ax[0,1].set_title('other firing map')
ax[0,1].set_xlabel('x'); plt.ylabel('y')

im2= ax[1,0].imshow(occ2.T * dt, origin='lower', extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]], aspect='equal')
fig2.colorbar(im2, ax=ax[1,0], label='Occupancy (s)')
ax[1,0].set_title('self Occupancy map')
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')
ax[1,1].axis('off')
fig2.tight_layout()
maybe_show()


nbins_gam = 6
n_splines_2d = 6
X_social_place, y_social_bins, exposure_social, edges_social = compute_binned_design_4d(
    pos1, pos2, y, nbins=nbins_gam, dt=dt
)
X_self_place = X_social_place
X_other_place = X_social_place
y_self_bins = y_social_bins
y_other_bins = y_social_bins
exposure_self = exposure_social
exposure_other = exposure_social

lam_default = 0.6

# (A) social place model: log λ = te(x_self, y_self) + te(x_other, y_other)
def make_social_gam(lam):
    return PoissonGAM(
        te(0, 1, n_splines=n_splines_2d) + te(2, 3, n_splines=n_splines_2d),
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        lam=lam,
    )

gam_social_place = make_social_gam(lam_default).fit(
    X_social_place, y_social_bins, exposure=exposure_social
)

# (B) self place model: log λ = te(x_self, y_self)
def make_self_gam(lam):
    return PoissonGAM(
        te(0, 1, n_splines=n_splines_2d),
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        lam=lam,
    )

gam_self_place = make_self_gam(lam_default).fit(
    X_self_place, y_self_bins, exposure=exposure_self
)

# (C) other place model: log λ = te(x_other, y_other)
def make_other_gam(lam):
    return PoissonGAM(
        te(2, 3, n_splines=n_splines_2d),
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        lam=lam,
    )

gam_other_place = make_other_gam(lam_default).fit(
    X_other_place, y_other_bins, exposure=exposure_other
)

def get_explained_deviance(gam):
    stats = getattr(gam, "statistics_", None)
    if not stats:
        return np.nan
    return stats["pseudo_r2"]["explained_deviance"]

print("social place model explained deviance:", round(get_explained_deviance(gam_social_place), 3))
print("self place model explained deviance:",   round(get_explained_deviance(gam_self_place), 3))
print("other place model explained deviance:",  round(get_explained_deviance(gam_other_place), 3))
def _lam_str(gam):
    lam = getattr(gam, "lam", None)
    if lam is None:
        return "unknown"
    if np.isscalar(lam):
        return f"{lam:.3g}"
    return "[" + ", ".join(f"{v:.3g}" for v in np.ravel(lam)) + "]"

print(f"n_splines_2d={n_splines_2d}")
print(f"lam social={_lam_str(gam_social_place)}")
print(f"lam self={_lam_str(gam_self_place)}")
print(f"lam other={_lam_str(gam_other_place)}")

from scipy.stats import chi2

X_both   = X_social_place
X_self   = X_self_place
X_other  = X_other_place

def loglik_poisson(gam, X, y, exposure):
    mu = gam.predict_mu(X) * exposure  # expected spikes/bin
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum(y * np.log(mu) - mu))

test_frac = 0.2
bins = np.clip(
    np.column_stack([
        np.digitize(pos1[:, 0], edges_social) - 1,
        np.digitize(pos1[:, 1], edges_social) - 1,
        np.digitize(pos2[:, 0], edges_social) - 1,
        np.digitize(pos2[:, 1], edges_social) - 1,
    ]),
    0,
    nbins_gam - 1,
)
bin_ids = (
    bins[:, 0]
    + nbins_gam * bins[:, 1]
    + (nbins_gam ** 2) * bins[:, 2]
    + (nbins_gam ** 3) * bins[:, 3]
)
test_mask = np.zeros(N, dtype=bool)
for bid in np.unique(bin_ids):
    idx_bin = np.where(bin_ids == bid)[0]
    if idx_bin.size <= 1:
        continue
    n_test = max(1, int(round(test_frac * idx_bin.size)))
    n_test = min(n_test, idx_bin.size - 1)
    test_idx = rng.choice(idx_bin, size=n_test, replace=False)
    test_mask[test_idx] = True
test_time = np.where(test_mask)[0]
train_time = np.where(~test_mask)[0]

X_both_train, y_both_train, exposure_both_train = compute_binned_design_4d_with_edges(
    pos1[train_time], pos2[train_time], y[train_time], edges_social, dt
)
X_both_test, y_both_test, exposure_both_test = compute_binned_design_4d_with_edges(
    pos1[test_time], pos2[test_time], y[test_time], edges_social, dt
)

# Fit on train only (refit gridsearch on train)
gam_both_train = make_social_gam(lam_default).fit(
    X_both_train, y_both_train, exposure=exposure_both_train
)
gam_self_train = make_self_gam(lam_default).fit(
    X_both_train, y_both_train, exposure=exposure_both_train
)
gam_other_train = make_other_gam(lam_default).fit(
    X_both_train, y_both_train, exposure=exposure_both_train
)

mu_null = np.full(
    y_both_test.size, y_both_train.sum() / exposure_both_train.sum(), dtype=float
)
mu_null = mu_null * exposure_both_test
ll_null_test = float(np.sum(y_both_test * np.log(mu_null) - mu_null))

#
ll_self_test = loglik_poisson(gam_self_train, X_both_test, y_both_test, exposure_both_test)
ll_other_test = loglik_poisson(gam_other_train, X_both_test, y_both_test, exposure_both_test)
ll_both_test = loglik_poisson(gam_both_train, X_both_test, y_both_test, exposure_both_test)

# McFadden pseudo-R^2 on test
r2_self  = 1 - ll_self_test  / ll_null_test
r2_other = 1 - ll_other_test / ll_null_test
r2_both  = 1 - ll_both_test  / ll_null_test

# Likelihood-ratio tests on test set (approximate, for illustration)
# Compare BOTH vs SELF-only and BOTH vs OTHER-only
# df approximated by effective dof difference; use param counts as proxy here
k_self  = gam_self_train.statistics_['edof']
k_other = gam_other_train.statistics_['edof']
k_both  = gam_both_train.statistics_['edof']

lrt_both_vs_self  = 2*(ll_both_test - ll_self_test)
lrt_both_vs_other = 2*(ll_both_test - ll_other_test)

p_both_vs_self  = chi2.sf(lrt_both_vs_self,  df=max(1, int(round(k_both - k_self))))
p_both_vs_other = chi2.sf(lrt_both_vs_other, df=max(1, int(round(k_both - k_other))))

print("\n=== Held-out comparisons ===")
print(f"Test log-lik: SELF={ll_self_test:.1f}, OTHER={ll_other_test:.1f}, BOTH={ll_both_test:.1f}, NULL={ll_null_test:.1f}")
print(f"Pseudo-R^2 (test): SELF={r2_self:.3f}, OTHER={r2_other:.3f}, BOTH={r2_both:.3f}")
print(f"LRT BOTH vs SELF:  stat={lrt_both_vs_self:.1f}, df≈{max(1, int(round(k_both - k_self)))}, p={p_both_vs_self:.2e}")
print(f"LRT BOTH vs OTHER: stat={lrt_both_vs_other:.1f}, df≈{max(1, int(round(k_both - k_other)))}, p={p_both_vs_other:.2e}")

# Bar plot of pseudo-R^2 on test
labels = ["Self-only", "Other-only", "Both"]
vals   = [r2_self, r2_other, r2_both]
plt.figure(figsize=(5.2,3.6))
plt.bar(labels, vals)
plt.ylabel('Pseudo-$R^2$ (test)')
plt.ylim(0, 1)
plt.title('Model comparison on held-out data')
plt.tight_layout()
maybe_show()

# -------------------------
# 3) Visualize the recovered place field (partial effect)
# -------------------------
def plot_tensor_map(gam, which, title):
    """Visualize the partial rate map for one tensor while holding the other fixed."""
    res = 40
    xs  = np.linspace(0,1,res)
    ys  = np.linspace(0,1,res)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    grid2d = np.column_stack([XX.ravel(), YY.ravel()])

    x_self_med, y_self_med = np.median(pos1[:,0]), np.median(pos1[:,1])
    x_oth_med,  y_oth_med  = np.median(pos2[:,0]), np.median(pos2[:,1])

    if which == 'self':
        Xgrid = np.column_stack([
            grid2d[:,0], grid2d[:,1],  # x_self, y_self vary
            np.full(grid2d.shape[0], x_oth_med),
            np.full(grid2d.shape[0], y_oth_med)
        ])
    elif which == 'other':
        Xgrid = np.column_stack([
            np.full(grid2d.shape[0], x_self_med),
            np.full(grid2d.shape[0], y_self_med),
            grid2d[:,0], grid2d[:,1]   # x_other, y_other vary
        ])
    else:
        raise ValueError("which must be 'self' or 'other'")

    lam = gam.predict_mu(Xgrid)  # Hz (exposure=1)
    plt.figure(figsize=(5.3, 4.6))
    plt.imshow(lam.reshape(res, res).T, origin='lower',
               extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='equal')
    plt.colorbar(label='Firing rate (Hz)')
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    maybe_show()

plot_tensor_map(gam_social_place, 'self',  "Recovered self map (holding other fixed)")
plot_tensor_map(gam_social_place, 'other', "Recovered conspecific map (holding self fixed)")

