from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


matplotlib.use("Agg")


# Enclosure dimensions follow the four-wall layout described by Piza et al. 2024.
# The unfolded wall coordinate uses (u, v) with wall order: north, east, south, west.
# North/south walls are 1 m wide and 2 m tall, east/west walls are 2 m wide and 2 m tall.
WIDTH_X_M = 1.0
DEPTH_Y_M = 2.0
HEIGHT_Z_M = 2.0
WALL_PERIMETER_M = 2.0 * (WIDTH_X_M + DEPTH_Y_M)

BEHAVIOR_FS = 250
SPIKE_FS = 6000
DURATION_S = 900.0

DATA_OUT = Path("simulated_neuorn.npz")
FIG_OUT = Path("simulated_neuorn_summary.png")
FIELD_FIG_OUT = Path("simulated_neuorn_fields.png")


def wrap_u(u):
    return np.mod(u, WALL_PERIMETER_M)


def circular_du(u1, u2):
    du = u1 - u2
    return (du + WALL_PERIMETER_M / 2.0) % WALL_PERIMETER_M - WALL_PERIMETER_M / 2.0


def reflect_z(z):
    bounces = 0
    while z < 0.0 or z > HEIGHT_Z_M:
        if z < 0.0:
            z = -z
            bounces += 1
        elif z > HEIGHT_Z_M:
            z = 2.0 * HEIGHT_Z_M - z
            bounces += 1
    return z, bounces


def normalize_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def normalize_vec(x):
    return x / max(np.linalg.norm(x), 1e-12)


def angle_between(v1, v2):
    dot = np.clip(np.dot(normalize_vec(v1), normalize_vec(v2)), -1.0, 1.0)
    return np.arccos(dot)


def slerp(v0, v1, frac):
    v0 = normalize_vec(v0)
    v1 = normalize_vec(v1)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    if dot > 0.9995:
        return normalize_vec((1.0 - frac) * v0 + frac * v1)
    omega = np.arccos(dot)
    so = np.sin(omega)
    return np.sin((1.0 - frac) * omega) / so * v0 + np.sin(frac * omega) / so * v1


def unfolded_to_xyz(u, z):
    u = wrap_u(np.asarray(u))
    z = np.asarray(z)
    xyz = np.zeros((u.shape[0], 3), dtype=float)

    north = (u >= 0.0) & (u < WIDTH_X_M)
    east = (u >= WIDTH_X_M) & (u < WIDTH_X_M + DEPTH_Y_M)
    south = (u >= WIDTH_X_M + DEPTH_Y_M) & (u < 2.0 * WIDTH_X_M + DEPTH_Y_M)
    west = ~(north | east | south)

    xyz[north, 0] = WIDTH_X_M - u[north]
    xyz[north, 1] = DEPTH_Y_M
    xyz[north, 2] = z[north]

    xyz[east, 0] = WIDTH_X_M
    xyz[east, 1] = u[east] - WIDTH_X_M
    xyz[east, 2] = z[east]

    xyz[south, 0] = u[south] - (WIDTH_X_M + DEPTH_Y_M)
    xyz[south, 1] = 0.0
    xyz[south, 2] = z[south]

    xyz[west, 0] = 0.0
    xyz[west, 1] = DEPTH_Y_M - (u[west] - (2.0 * WIDTH_X_M + DEPTH_Y_M))
    xyz[west, 2] = z[west]
    return xyz


def wall_frame(u):
    u = wrap_u(np.asarray(u))
    tangent = np.zeros((u.shape[0], 3), dtype=float)
    inward = np.zeros((u.shape[0], 3), dtype=float)
    vertical = np.tile(np.array([[0.0, 0.0, 1.0]]), (u.shape[0], 1))

    north = (u >= 0.0) & (u < WIDTH_X_M)
    east = (u >= WIDTH_X_M) & (u < WIDTH_X_M + DEPTH_Y_M)
    south = (u >= WIDTH_X_M + DEPTH_Y_M) & (u < 2.0 * WIDTH_X_M + DEPTH_Y_M)
    west = ~(north | east | south)

    tangent[north] = np.array([-1.0, 0.0, 0.0])
    inward[north] = np.array([0.0, -1.0, 0.0])

    tangent[east] = np.array([0.0, 1.0, 0.0])
    inward[east] = np.array([-1.0, 0.0, 0.0])

    tangent[south] = np.array([1.0, 0.0, 0.0])
    inward[south] = np.array([0.0, 1.0, 0.0])

    tangent[west] = np.array([0.0, -1.0, 0.0])
    inward[west] = np.array([1.0, 0.0, 0.0])
    return tangent, vertical, inward


def xyz_to_unfolded(xyz):
    x, y, z = xyz
    eps = 1e-6
    if np.isclose(y, DEPTH_Y_M, atol=eps):
        u = np.clip(WIDTH_X_M - x, 0.0, WIDTH_X_M)
    elif np.isclose(x, WIDTH_X_M, atol=eps):
        u = WIDTH_X_M + np.clip(y, 0.0, DEPTH_Y_M)
    elif np.isclose(y, 0.0, atol=eps):
        u = WIDTH_X_M + DEPTH_Y_M + np.clip(x, 0.0, WIDTH_X_M)
    else:
        u = 2.0 * WIDTH_X_M + DEPTH_Y_M + np.clip(DEPTH_Y_M - y, 0.0, DEPTH_Y_M)
    return np.array([wrap_u(u), np.clip(z, 0.0, HEIGHT_Z_M)], dtype=float)


def view_intersection(pos_xyz, head_dir):
    px, py, pz = pos_xyz
    direction = normalize_vec(head_dir)
    dx, dy, dz = direction
    candidates = []

    if dx > 1e-8:
        tau = (WIDTH_X_M - px) / dx
        hit = pos_xyz + tau * direction
        if tau > 1e-6 and 0.0 <= hit[1] <= DEPTH_Y_M and 0.0 <= hit[2] <= HEIGHT_Z_M:
            candidates.append(hit)
    if dx < -1e-8:
        tau = -px / dx
        hit = pos_xyz + tau * direction
        if tau > 1e-6 and 0.0 <= hit[1] <= DEPTH_Y_M and 0.0 <= hit[2] <= HEIGHT_Z_M:
            candidates.append(hit)
    if dy > 1e-8:
        tau = (DEPTH_Y_M - py) / dy
        hit = pos_xyz + tau * direction
        if tau > 1e-6 and 0.0 <= hit[0] <= WIDTH_X_M and 0.0 <= hit[2] <= HEIGHT_Z_M:
            candidates.append(hit)
    if dy < -1e-8:
        tau = -py / dy
        hit = pos_xyz + tau * direction
        if tau > 1e-6 and 0.0 <= hit[0] <= WIDTH_X_M and 0.0 <= hit[2] <= HEIGHT_Z_M:
            candidates.append(hit)

    if not candidates:
        x, y, z = pos_xyz
        eps = 1e-6
        if np.isclose(y, DEPTH_Y_M, atol=eps):
            return np.array([np.clip(WIDTH_X_M - x, 0.0, WIDTH_X_M), np.clip(z, 0.0, HEIGHT_Z_M)])
        if np.isclose(x, WIDTH_X_M, atol=eps):
            return np.array([2.0 * WIDTH_X_M + DEPTH_Y_M + (DEPTH_Y_M - np.clip(y, 0.0, DEPTH_Y_M)), np.clip(z, 0.0, HEIGHT_Z_M)])
        if np.isclose(y, 0.0, atol=eps):
            return np.array([WIDTH_X_M + DEPTH_Y_M + np.clip(x, 0.0, WIDTH_X_M), np.clip(z, 0.0, HEIGHT_Z_M)])
        return np.array([WIDTH_X_M + np.clip(y, 0.0, DEPTH_Y_M), np.clip(z, 0.0, HEIGHT_Z_M)])

    distances = [np.linalg.norm(hit - pos_xyz) for hit in candidates]
    hit = candidates[int(np.argmin(distances))]
    return xyz_to_unfolded(hit)


def circular_gaussian(u, z, center_u, center_z, sigma_u, sigma_z, amplitude):
    du = circular_du(u, center_u)
    dz = z - center_z
    return amplitude * np.exp(-0.5 * (du / sigma_u) ** 2 - 0.5 * (dz / sigma_z) ** 2)


def angular_gaussian_deg(az_deg, pitch_deg, center_az_deg, center_pitch_deg, sigma_az_deg, sigma_pitch_deg, amplitude):
    daz = (az_deg - center_az_deg + 180.0) % 360.0 - 180.0
    dpitch = pitch_deg - center_pitch_deg
    return amplitude * np.exp(-0.5 * (daz / sigma_az_deg) ** 2 - 0.5 * (dpitch / sigma_pitch_deg) ** 2)


def choose_target_on_wall(rng):
    wall_choice = rng.integers(0, 4)
    if wall_choice == 0:
        return np.array([rng.uniform(0.0, WIDTH_X_M), 0.0, rng.uniform(0.05, HEIGHT_Z_M - 0.05)])
    if wall_choice == 1:
        return np.array([WIDTH_X_M, rng.uniform(0.0, DEPTH_Y_M), rng.uniform(0.05, HEIGHT_Z_M - 0.05)])
    if wall_choice == 2:
        return np.array([rng.uniform(0.0, WIDTH_X_M), DEPTH_Y_M, rng.uniform(0.05, HEIGHT_Z_M - 0.05)])
    return np.array([0.0, rng.uniform(0.0, DEPTH_Y_M), rng.uniform(0.05, HEIGHT_Z_M - 0.05)])


def simulate_animal(rng, duration_s=DURATION_S, fs=BEHAVIOR_FS):
    dt = 1.0 / fs
    n = int(duration_s * fs)
    vertical_steer_gain = 2.2
    vertical_goal_prob = 0.45

    pos = np.zeros((n, 2), dtype=float)
    vel_uv = np.zeros((n, 2), dtype=float)
    speed = np.zeros(n, dtype=float)
    state = np.zeros(n, dtype=int)
    head_dir = np.zeros((n, 3), dtype=float)
    view = np.zeros((n, 2), dtype=float)
    occupancy = np.zeros((24, 16), dtype=float)

    pos[0] = np.array([rng.uniform(0.0, WALL_PERIMETER_M), rng.uniform(0.2, HEIGHT_Z_M - 0.2)], dtype=float)
    tangent0, vertical0, inward0 = wall_frame(pos[:1, 0])
    head_dir[0] = normalize_vec(0.85 * inward0[0] + 0.15 * tangent0[0] + 0.10 * vertical0[0])

    epoch_mode = "move"
    epoch_steps_left = 0
    move_goal = pos[0].copy()
    speed_state = 0.18
    saccade = None
    fixation_steps_left = 0

    def occupancy_goal(current_pos):
        flat = occupancy.ravel()
        candidates = np.where(flat <= flat.min() + 0.5)[0]
        choice = candidates[rng.integers(0, candidates.size)]
        iu = choice // occupancy.shape[1]
        iz = choice % occupancy.shape[1]
        goal = np.array(
            [
                (iu + rng.uniform(0.05, 0.95)) * WALL_PERIMETER_M / occupancy.shape[0],
                (iz + rng.uniform(0.05, 0.95)) * HEIGHT_Z_M / occupancy.shape[1],
            ],
            dtype=float,
        )
        if rng.random() < 0.35:
            goal[0] = wrap_u(current_pos[0] + rng.normal(0.0, 0.35))
        return goal

    def vertical_goal(current_pos):
        target_z = rng.uniform(1.72, 1.95) if current_pos[1] < HEIGHT_Z_M / 2.0 else rng.uniform(0.05, 0.28)
        target_u = wrap_u(current_pos[0] + rng.normal(0.0, 0.18))
        return np.array([target_u, target_z], dtype=float)

    for t in range(1, n):
        iu = min(occupancy.shape[0] - 1, int(pos[t - 1, 0] / WALL_PERIMETER_M * occupancy.shape[0]))
        iz = min(occupancy.shape[1] - 1, int(pos[t - 1, 1] / HEIGHT_Z_M * occupancy.shape[1]))
        occupancy[iu, iz] += 1.0

        if epoch_steps_left <= 0:
            epoch_mode = "stop" if epoch_mode == "move" else "move"
            if epoch_mode == "move":
                epoch_steps_left = int(rng.uniform(1.5, 6.0) * fs)
                if rng.random() < vertical_goal_prob:
                    move_goal = vertical_goal(pos[t - 1])
                else:
                    move_goal = occupancy_goal(pos[t - 1])
                saccade = None
                fixation_steps_left = 0
            else:
                epoch_steps_left = int(rng.uniform(0.35, 2.0) * fs)
                fixation_steps_left = int(rng.uniform(0.06, 0.25) * fs)
                saccade = None

        if epoch_mode == "move":
            state[t] = 1
            goal_du = circular_du(move_goal[0], pos[t - 1, 0])
            goal_dz = move_goal[1] - pos[t - 1, 1]
            if np.hypot(goal_du, goal_dz) < 0.08:
                if rng.random() < vertical_goal_prob:
                    move_goal = vertical_goal(pos[t - 1])
                else:
                    move_goal = occupancy_goal(pos[t - 1])
                goal_du = circular_du(move_goal[0], pos[t - 1, 0])
                goal_dz = move_goal[1] - pos[t - 1, 1]

            goal_heading = np.arctan2(vertical_steer_gain * goal_dz, goal_du)
            heading_noise = rng.normal(0.0, np.deg2rad(5.0))
            speed_target = rng.uniform(0.1, 0.5)
            speed_state = 0.92 * speed_state + 0.08 * speed_target
            speed_state = float(np.clip(speed_state, 0.1, 0.5))

            du = speed_state * np.cos(goal_heading + heading_noise) * dt
            dz = speed_state * np.sin(goal_heading + heading_noise) * dt
            new_u = wrap_u(pos[t - 1, 0] + du)
            new_z, bounces = reflect_z(pos[t - 1, 1] + dz)
            if bounces % 2 == 1:
                dz = -dz

            pos[t] = np.array([new_u, new_z], dtype=float)
            vel_uv[t] = np.array([circular_du(pos[t, 0], pos[t - 1, 0]) / dt, (pos[t, 1] - pos[t - 1, 1]) / dt], dtype=float)
            speed[t] = np.linalg.norm(vel_uv[t])

            tangent, vertical, inward = wall_frame(pos[t : t + 1, 0])
            move_dir = normalize_vec(vel_uv[t, 0] * tangent[0] + vel_uv[t, 1] * vertical[0])
            head_dir[t] = normalize_vec(0.90 * move_dir + 0.40 * inward[0])
        else:
            state[t] = 0
            pos[t] = pos[t - 1]
            vel_uv[t] = 0.0
            speed[t] = 0.0
            pos_xyz = unfolded_to_xyz(pos[t : t + 1, 0], pos[t : t + 1, 1])[0]

            if saccade is not None:
                saccade["step"] += 1
                frac = saccade["step"] / saccade["duration"]
                frac = np.clip(frac, 0.0, 1.0)
                smooth_frac = 0.5 - 0.5 * np.cos(np.pi * frac)
                head_dir[t] = slerp(saccade["start"], saccade["target"], smooth_frac)
                if saccade["step"] >= saccade["duration"]:
                    saccade = None
                    fixation_steps_left = int(rng.uniform(0.06, 0.28) * fs)
            else:
                head_dir[t] = head_dir[t - 1]
                fixation_steps_left -= 1
                if fixation_steps_left <= 0:
                    accepted = False
                    for _ in range(80):
                        target_xyz = choose_target_on_wall(rng)
                        target_dir = normalize_vec(target_xyz - pos_xyz)
                        amp_deg = np.rad2deg(angle_between(head_dir[t - 1], target_dir))
                        if 10.0 <= amp_deg <= 250.0:
                            peak_speed_deg = rng.uniform(10.0, 600.0)
                            duration_steps = max(2, int(np.ceil(2.0 * amp_deg / peak_speed_deg / dt)))
                            saccade = {
                                "start": head_dir[t - 1].copy(),
                                "target": target_dir,
                                "duration": duration_steps,
                                "step": 0,
                            }
                            head_dir[t] = slerp(saccade["start"], saccade["target"], 0.0)
                            accepted = True
                            break
                    if not accepted:
                        tangent, vertical, inward = wall_frame(pos[t : t + 1, 0])
                        target_dir = normalize_vec(0.85 * inward[0] + 0.20 * tangent[0] + 0.10 * vertical[0])
                        head_dir[t] = target_dir
                        fixation_steps_left = int(rng.uniform(0.1, 0.3) * fs)

        view[t] = view_intersection(unfolded_to_xyz(pos[t : t + 1, 0], pos[t : t + 1, 1])[0], head_dir[t])
        epoch_steps_left -= 1

    view[0] = view_intersection(unfolded_to_xyz(pos[:1, 0], pos[:1, 1])[0], head_dir[0])
    pos_xyz = unfolded_to_xyz(pos[:, 0], pos[:, 1])

    head_dir = normalize_rows(head_dir)
    angular_speed_deg_s = np.zeros(n, dtype=float)
    angular_speed_deg_s[1:] = np.rad2deg(
        np.arccos(np.clip(np.sum(head_dir[1:] * head_dir[:-1], axis=1), -1.0, 1.0))
    ) / dt
    yaw_deg = np.rad2deg(np.arctan2(head_dir[:, 1], head_dir[:, 0]))
    pitch_deg = np.rad2deg(np.arctan2(head_dir[:, 2], np.hypot(head_dir[:, 0], head_dir[:, 1])))

    return {
        "time_s": np.arange(n, dtype=float) / fs,
        "pos_wall": pos,
        "pos_xyz": pos_xyz,
        "speed_m_s": speed,
        "move_mask": state.astype(bool),
        "head_dir": head_dir,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "angular_speed_deg_s": angular_speed_deg_s,
        "spatial_view_wall": view,
    }


def simulate_spiking(self_animal, other_animal, rng):
    u_self = self_animal["pos_wall"][:, 0]
    z_self = self_animal["pos_wall"][:, 1]
    u_other = other_animal["pos_wall"][:, 0]
    z_other = other_animal["pos_wall"][:, 1]
    u_view = self_animal["spatial_view_wall"][:, 0]
    z_view = self_animal["spatial_view_wall"][:, 1]

    f_self = circular_gaussian(
        u_self, z_self, center_u=1.05, center_z=1.25, sigma_u=0.38, sigma_z=0.30, amplitude=0.85
    )
    f_other = circular_gaussian(
        u_other, z_other, center_u=4.30, center_z=1.45, sigma_u=0.45, sigma_z=0.28, amplitude=0.70
    )
    f_view = circular_gaussian(
        u_view, z_view, center_u=2.60, center_z=1.55, sigma_u=0.45, sigma_z=0.25, amplitude=1.00
    )
    f_head = angular_gaussian_deg(
        self_animal["yaw_deg"],
        self_animal["pitch_deg"],
        center_az_deg=55.0,
        center_pitch_deg=18.0,
        sigma_az_deg=38.0,
        sigma_pitch_deg=16.0,
        amplitude=0.45,
    )
    move_bonus = 0.15 * self_animal["move_mask"].astype(float)

    log_rate_behavior = np.log(3.0) + f_self + f_other + f_view + f_head + move_bonus
    rate_behavior_hz = np.exp(log_rate_behavior)

    upsample = SPIKE_FS // BEHAVIOR_FS
    rate_spike_hz = np.repeat(rate_behavior_hz, upsample)
    spike_counts = rng.poisson(rate_spike_hz / SPIKE_FS)
    spike_time_s = np.arange(rate_spike_hz.shape[0], dtype=float) / SPIKE_FS

    return {
        "behavior_rate_hz": rate_behavior_hz,
        "spike_rate_hz": rate_spike_hz,
        "spike_counts": spike_counts,
        "spike_time_s": spike_time_s,
        "f_self": f_self,
        "f_other": f_other,
        "f_view": f_view,
        "f_head": f_head,
    }


def save_dataset(self_animal, other_animal, spikes, out_path=DATA_OUT):
    np.savez(
        out_path,
        behavior_time_s=self_animal["time_s"],
        spike_time_s=spikes["spike_time_s"],
        self_pos_wall=self_animal["pos_wall"],
        self_pos_xyz=self_animal["pos_xyz"],
        self_speed_m_s=self_animal["speed_m_s"],
        self_move_mask=self_animal["move_mask"],
        self_head_dir=self_animal["head_dir"],
        self_yaw_deg=self_animal["yaw_deg"],
        self_pitch_deg=self_animal["pitch_deg"],
        self_angular_speed_deg_s=self_animal["angular_speed_deg_s"],
        self_spatial_view_wall=self_animal["spatial_view_wall"],
        other_pos_wall=other_animal["pos_wall"],
        other_pos_xyz=other_animal["pos_xyz"],
        other_speed_m_s=other_animal["speed_m_s"],
        other_move_mask=other_animal["move_mask"],
        other_head_dir=other_animal["head_dir"],
        other_yaw_deg=other_animal["yaw_deg"],
        other_pitch_deg=other_animal["pitch_deg"],
        other_angular_speed_deg_s=other_animal["angular_speed_deg_s"],
        other_spatial_view_wall=other_animal["spatial_view_wall"],
        neuron_rate_behavior_hz=spikes["behavior_rate_hz"],
        neuron_rate_spike_hz=spikes["spike_rate_hz"],
        neuron_spike_counts=spikes["spike_counts"],
        neuron_f_self=spikes["f_self"],
        neuron_f_other=spikes["f_other"],
        neuron_f_view=spikes["f_view"],
        neuron_f_head=spikes["f_head"],
    )
    return out_path


def plot_summary(self_animal, other_animal, spikes, out_path=FIG_OUT, window_s=20.0):
    behavior_time = self_animal["time_s"]
    spike_time = spikes["spike_time_s"]
    beh_keep = behavior_time <= window_s
    spike_keep = spike_time <= window_s

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1])
    axes = [fig.add_subplot(gs[i, 0], sharex=None if i == 0 else None) for i in range(0, 6)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    axes[0].plot(behavior_time[beh_keep], self_animal["speed_m_s"][beh_keep], label="self speed", lw=1.2)
    axes[0].plot(behavior_time[beh_keep], other_animal["speed_m_s"][beh_keep], label="other speed", lw=1.0, alpha=0.85)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Simulated behavior and Poisson spike rate")

    axes[1].step(behavior_time[beh_keep], self_animal["move_mask"][beh_keep].astype(float), where="post", label="self moving", lw=1.2)
    axes[1].step(behavior_time[beh_keep], other_animal["move_mask"][beh_keep].astype(float) + 0.05, where="post", label="other moving", lw=1.0, alpha=0.85)
    axes[1].set_ylabel("Move state")
    axes[1].legend(loc="upper right")

    axes[2].plot(behavior_time[beh_keep], self_animal["yaw_deg"][beh_keep], label="yaw", lw=1.1)
    axes[2].plot(behavior_time[beh_keep], self_animal["pitch_deg"][beh_keep], label="pitch", lw=1.1)
    axes[2].set_ylabel("Head angle (deg)")
    axes[2].legend(loc="upper right")

    axes[3].plot(behavior_time[beh_keep], self_animal["angular_speed_deg_s"][beh_keep], label="self AHV", lw=1.2)
    axes[3].plot(behavior_time[beh_keep], other_animal["angular_speed_deg_s"][beh_keep], label="other AHV", lw=1.0, alpha=0.85)
    axes[3].set_ylabel("AHV (deg/s)")
    axes[3].legend(loc="upper right")

    axes[4].plot(behavior_time[beh_keep], self_animal["spatial_view_wall"][beh_keep, 0], label="view u", lw=1.1)
    axes[4].plot(behavior_time[beh_keep], self_animal["spatial_view_wall"][beh_keep, 1], label="view v", lw=1.1)
    axes[4].set_ylabel("Spatial view")
    axes[4].legend(loc="upper right")

    axes[5].plot(spike_time[spike_keep], spikes["spike_rate_hz"][spike_keep], color="k", lw=1.0, label="inst. rate")
    spike_events = spike_time[spike_keep & (spikes["spike_counts"] > 0)]
    if spike_events.size:
        axes[5].vlines(spike_events, 0.0, np.percentile(spikes["spike_rate_hz"][spike_keep], 95), color="#d62728", alpha=0.35, lw=0.5)
    axes[5].set_ylabel("Rate (Hz)")
    axes[5].set_xlabel("Time (s)")
    axes[5].legend(loc="upper right")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_tuning_fields(self_animal, other_animal, spikes, out_path=FIELD_FIG_OUT):
    fig, axes = plt.subplots(5, 2, figsize=(13, 20), constrained_layout=True)

    beh_spike_counts = spikes["spike_counts"].reshape(-1, SPIKE_FS // BEHAVIOR_FS).sum(axis=1)
    dt = 1.0 / BEHAVIOR_FS

    def observed_rate_map(x, y, x_edges, y_edges, wrap_x_axis=False):
        occ, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        spk, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=beh_spike_counts)
        occ_smooth = gaussian_filter(
            occ.astype(float),
            sigma=1.5,
            mode=("wrap", "reflect") if wrap_x_axis else ("nearest", "reflect"),
        )
        spk_smooth = gaussian_filter(
            spk.astype(float),
            sigma=1.5,
            mode=("wrap", "reflect") if wrap_x_axis else ("nearest", "reflect"),
        )
        rate = spk_smooth / np.maximum(occ_smooth * dt, 1e-12)
        rate[occ < 1.0] = np.nan
        return rate.T

    def raw_ground_truth_uv_map(kind):
        u_centers = 0.5 * (uv_edges[0][:-1] + uv_edges[0][1:])
        v_centers = 0.5 * (uv_edges[1][:-1] + uv_edges[1][1:])
        uu, vv = np.meshgrid(u_centers, v_centers, indexing="xy")
        if kind == "self":
            field = np.exp(np.log(3.0) + circular_gaussian(uu, vv, 1.05, 1.25, 0.38, 0.30, 0.85))
        elif kind == "other":
            field = np.exp(np.log(3.0) + circular_gaussian(uu, vv, 4.30, 1.45, 0.45, 0.28, 0.70))
        else:
            field = np.exp(np.log(3.0) + circular_gaussian(uu, vv, 2.60, 1.55, 0.45, 0.25, 1.00))
        return field

    def raw_ground_truth_head_map():
        az_centers = 0.5 * (az_edges[:-1] + az_edges[1:])
        pitch_centers = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])
        aa, pp = np.meshgrid(az_centers, pitch_centers, indexing="xy")
        return np.exp(np.log(3.0) + angular_gaussian_deg(aa, pp, 55.0, 18.0, 38.0, 16.0, 0.45))

    uv_edges = [np.linspace(0.0, WALL_PERIMETER_M, 33), np.linspace(0.0, HEIGHT_Z_M, 17)]
    az_edges = np.linspace(-180.0, 180.0, 37)
    pitch_edges = np.linspace(-90.0, 90.0, 19)

    self_empirical = raw_ground_truth_uv_map("self")
    self_field = observed_rate_map(self_animal["pos_wall"][:, 0], self_animal["pos_wall"][:, 1], uv_edges[0], uv_edges[1], wrap_x_axis=True)
    other_empirical = raw_ground_truth_uv_map("other")
    other_field = observed_rate_map(other_animal["pos_wall"][:, 0], other_animal["pos_wall"][:, 1], uv_edges[0], uv_edges[1], wrap_x_axis=True)
    view_empirical = raw_ground_truth_uv_map("view")
    view_field = observed_rate_map(
        self_animal["spatial_view_wall"][:, 0],
        self_animal["spatial_view_wall"][:, 1],
        uv_edges[0],
        uv_edges[1],
        wrap_x_axis=True,
    )
    head_empirical = raw_ground_truth_head_map()
    head_field = observed_rate_map(
        self_animal["yaw_deg"],
        self_animal["pitch_deg"],
        az_edges,
        pitch_edges,
        wrap_x_axis=True,
    )

    def plot_wrapped_uv(ax, uv, color, lw):
        jumps = np.abs(np.diff(uv[:, 0])) > (WALL_PERIMETER_M / 2.0)
        start = 0
        for idx in np.where(jumps)[0]:
            stop = idx + 1
            if stop - start >= 2:
                ax.plot(uv[start:stop, 0], uv[start:stop, 1], color=color, lw=lw, alpha=0.95)
            start = stop
        if uv.shape[0] - start >= 2:
            ax.plot(uv[start:, 0], uv[start:, 1], color=color, lw=lw, alpha=0.95)

    spike_bins = np.flatnonzero(spikes["spike_counts"].reshape(-1, SPIKE_FS // BEHAVIOR_FS).sum(axis=1) > 0)

    traj_panels = [
        (axes[0, 0], self_animal["pos_wall"], "Self Trajectory + Spikes"),
        (axes[0, 1], other_animal["pos_wall"], "Other Trajectory + Spikes"),
    ]
    for ax, uv, title in traj_panels:
        plot_wrapped_uv(ax, uv, "k", 0.8)
        ax.scatter(uv[spike_bins, 0], uv[spike_bins, 1], s=9, color="#d62728", alpha=0.7)
        for boundary in [WIDTH_X_M, WIDTH_X_M + DEPTH_Y_M, 2.0 * WIDTH_X_M + DEPTH_Y_M]:
            ax.axvline(boundary, color="0.75", ls="--", lw=0.8)
        ax.set_xlim(0.0, WALL_PERIMETER_M)
        ax.set_ylim(0.0, HEIGHT_Z_M)
        ax.set_xlabel("u (m)")
        ax.set_ylabel("v (m)")
        ax.set_title(title)

    panels = [
        (axes[1, 0], self_empirical, "Self Place Empirical", [0.0, WALL_PERIMETER_M, 0.0, HEIGHT_Z_M], "u (m)", "v (m)"),
        (axes[1, 1], self_field, "Self Place Observed", [0.0, WALL_PERIMETER_M, 0.0, HEIGHT_Z_M], "u (m)", "v (m)"),
        (axes[2, 0], other_empirical, "Other Place Empirical", [0.0, WALL_PERIMETER_M, 0.0, HEIGHT_Z_M], "u (m)", "v (m)"),
        (axes[2, 1], other_field, "Other Place Observed", [0.0, WALL_PERIMETER_M, 0.0, HEIGHT_Z_M], "u (m)", "v (m)"),
        (axes[3, 0], view_empirical, "Spatial View Empirical", [0.0, WALL_PERIMETER_M, 0.0, HEIGHT_Z_M], "u (m)", "v (m)"),
        (axes[3, 1], view_field, "Spatial View Observed", [0.0, WALL_PERIMETER_M, 0.0, HEIGHT_Z_M], "u (m)", "v (m)"),
        (axes[4, 0], head_empirical, "Head Direction Empirical", [-180.0, 180.0, -90.0, 90.0], "Azimuth (deg)", "Pitch (deg)"),
        (axes[4, 1], head_field, "Head Direction Observed", [-180.0, 180.0, -90.0, 90.0], "Azimuth (deg)", "Pitch (deg)"),
    ]

    for ax, field, title, extent, xlabel, ylabel in panels:
        vmax = float(np.nanmax(field)) if np.any(np.isfinite(field)) else 1.0
        vmax = max(vmax, 1e-12)
        im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap="jet", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, shrink=0.85, label="Rate (Hz)")
        if extent[1] == WALL_PERIMETER_M:
            for boundary in [WIDTH_X_M, WIDTH_X_M + DEPTH_Y_M, 2.0 * WIDTH_X_M + DEPTH_Y_M]:
                ax.axvline(boundary, color="w", ls="--", lw=0.7, alpha=0.7)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    rng = np.random.default_rng(7)
    self_animal = simulate_animal(rng)
    other_animal = simulate_animal(np.random.default_rng(17))
    spikes = simulate_spiking(self_animal, other_animal, rng)

    data_path = save_dataset(self_animal, other_animal, spikes)
    fig_path = plot_summary(self_animal, other_animal, spikes)
    field_fig_path = plot_tuning_fields(self_animal, other_animal, spikes)

    print(f"Saved data: {data_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved field figure: {field_fig_path}")
    print(f"Behavior samples: {self_animal['time_s'].shape[0]}")
    print(f"Spike-rate samples: {spikes['spike_rate_hz'].shape[0]}")
    print(f"Mean rate: {spikes['behavior_rate_hz'].mean():.2f} Hz")
    print(f"Total spikes: {spikes['spike_counts'].sum()}")


if __name__ == "__main__":
    main()
