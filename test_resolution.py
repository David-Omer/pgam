import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Simulation parameters
duration = 600 # seconds
speed = 0.15 # m/s
arena_size = 1.0 # meters
place_field_center = (0.5, 0.5)
place_field_sigma = 0.15
peak_rate = 20.0 # Hz

# Function to generate data at a given sampling rate
def generate_data(fs):
    dt = 1.0 / fs
    n = int(duration * fs)
    
    # 1. Simple smooth random walk (Lissajous curve for simplicity of full coverage)
    t = np.arange(n) * dt
    x = 0.5 + 0.45 * np.sin(2 * np.pi * t * 0.031)
    y = 0.5 + 0.45 * np.sin(2 * np.pi * t * 0.043)
    
    # 2. True underlying rate based on position
    dx = x - place_field_center[0]
    dy = y - place_field_center[1]
    rate_hz = peak_rate * np.exp(-0.5 * (dx**2 + dy**2) / place_field_sigma**2)
    
    # 3. Generate spikes (Poisson)
    spikes = np.random.poisson(rate_hz * dt)
    return x, y, spikes, dt

# Function to calculate rate map
def calculate_map(x, y, spikes, dt, bins=20):
    edges = np.linspace(0, arena_size, bins + 1)
    
    # Occupancy (time spent in each bin)
    occ, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    occ_time = occ * dt
    
    # Spike counts
    spike_counts, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=spikes)
    
    # Smooth both (standard practice)
    occ_time_sm = gaussian_filter(occ_time, sigma=1.0)
    spike_counts_sm = gaussian_filter(spike_counts, sigma=1.0)
    
    # Rate
    rate_map = spike_counts_sm / np.maximum(occ_time_sm, 1e-12)
    rate_map[occ_time_sm < 0.01] = np.nan
    return rate_map

# Compare 10 Hz and 240 Hz
np.random.seed(42)
x_10, y_10, spikes_10, dt_10 = generate_data(10)
map_10 = calculate_map(x_10, y_10, spikes_10, dt_10)

x_240, y_240, spikes_240, dt_240 = generate_data(240)
map_240 = calculate_map(x_240, y_240, spikes_240, dt_240)

print(f"Max rate 10Hz map: {np.nanmax(map_10):.2f} Hz")
print(f"Max rate 240Hz map: {np.nanmax(map_240):.2f} Hz")
