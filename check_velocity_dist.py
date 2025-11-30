import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Define data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'src', 'time_trials', 'src', 'LineFollowing', 'data')

print(f"Searching for data in: {DATA_DIR}")

all_data = []
for root, dirs, files in os.walk(DATA_DIR):
    if 'log.csv' in files:
        csv_path = os.path.join(root, 'log.csv')
        try:
            df = pd.read_csv(csv_path)
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {csv_path}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

if not all_data:
    print("No data found!")
    exit()

df = pd.concat(all_data, ignore_index=True)

# Filter like training
initial_len = len(df)
# Filter stopped
df = df[(df.iloc[:, 1] != 0) | (df.iloc[:, 2] != 0)]
# Filter negative v
df = df[df.iloc[:, 1] >= 0]

# Risky maneuver filtering - remove high speed + sharp turns
initial_len = len(df)
df = df[~((df.iloc[:, 1] > 1.5) & (np.abs(df.iloc[:, 2]) > 2.0))]
print(f"Removed {initial_len - len(df)} risky samples (v > 1.5 and |w| > 2.0).")

print(f"Total samples after filtering: {len(df)}")

v = df.iloc[:, 1]
w = df.iloc[:, 2]

print("\nVelocity Statistics:")
print(f"Mean: {v.mean():.4f}")
print(f"Std:  {v.std():.4f}")
print(f"Min:  {v.min():.4f}")
print(f"Max:  {v.max():.4f}")

# Check unique values
unique_v = v.unique()
print(f"\nUnique velocity values: {len(unique_v)}")
if len(unique_v) < 20:
    print(f"Values: {np.sort(unique_v)}")

# Check if mostly constant
mode_v = v.mode()[0]
count_mode = (v == mode_v).sum()
print(f"\nMost common velocity: {mode_v} ({count_mode}/{len(df)} samples, {count_mode/len(df)*100:.1f}%)")

print("\nAngular Velocity Statistics:")
print(f"Mean: {w.mean():.4f}")
print(f"Std:  {w.std():.4f}")
print(f"Min:  {w.min():.4f}")
print(f"Max:  {w.max():.4f}")

# Check distribution of w
print("\nAngular Velocity Bins (approx) with Mean Velocity:")
bins = np.linspace(-3, 3, 11)
# Use digitize to get bin indices
indices = np.digitize(w, bins)
for i in range(1, len(bins)):
    mask = indices == i
    count = mask.sum()
    if count > 0:
        mean_v = v[mask].mean()
        print(f"[{bins[i-1]:.1f}, {bins[i]:.1f}]: {count} samples, Mean V: {mean_v:.3f}")
    else:
        print(f"[{bins[i-1]:.1f}, {bins[i]:.1f}]: 0 samples")
