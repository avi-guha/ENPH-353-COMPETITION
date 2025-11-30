import pandas as pd
import os
import numpy as np
import ast
import cv2
from collections import defaultdict

# Get data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'src', 'time_trials', 'src', 'LineFollowing', 'data')

print(f"Analyzing data in: {DATA_DIR}\n")

all_data = []
for root, dirs, files in os.walk(DATA_DIR):
    if 'log.csv' in files:
        csv_path = os.path.join(root, 'log.csv')
        try:
            df = pd.read_csv(csv_path)
            df['base_path'] = root
            df['run_name'] = os.path.basename(root)
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {os.path.basename(root)}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

if not all_data:
    print("No data found!")
    exit()

# Concatenate but keep run info
full_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal samples: {len(full_df)}")

# Apply same filters as training
initial = len(full_df)
full_df = full_df[(full_df.iloc[:, 1] != 0) | (full_df.iloc[:, 2] != 0)]
full_df = full_df[full_df.iloc[:, 1] >= 0]
full_df = full_df[~((full_df.iloc[:, 1] > 1.5) & (np.abs(full_df.iloc[:, 2]) > 2.0))]
print(f"After filtering: {len(full_df)} samples ({initial - len(full_df)} removed)\n")

v = full_df.iloc[:, 1].values
w = full_df.iloc[:, 2].values

print("="*60)
print("1. TEMPORAL CONSISTENCY ANALYSIS")
print("="*60)

# Analyze per-run temporal jumps
run_stats = []
for run_name in full_df['run_name'].unique():
    run_df = full_df[full_df['run_name'] == run_name].copy()
    run_df = run_df.sort_index()  # Ensure chronological order
    
    if len(run_df) < 2:
        continue
    
    # Calculate consecutive differences
    w_vals = run_df.iloc[:, 2].values
    w_diffs = np.abs(np.diff(w_vals))
    
    # Find large jumps (> 1.5 rad/s change between frames)
    large_jumps = w_diffs > 1.5
    num_jumps = large_jumps.sum()
    
    run_stats.append({
        'run': run_name,
        'samples': len(run_df),
        'jumps': num_jumps,
        'jump_rate': num_jumps / len(run_df) * 100,
        'max_jump': w_diffs.max(),
        'mean_diff': w_diffs.mean()
    })

# Sort by jump rate
run_stats = sorted(run_stats, key=lambda x: x['jump_rate'], reverse=True)

print("\nRuns with most temporal inconsistencies (sudden steering changes):")
print(f"{'Run':<15} {'Samples':>8} {'Jumps':>6} {'Rate':>7} {'Max Jump':>10} {'Avg Diff':>10}")
print("-"*75)
for stat in run_stats[:10]:  # Show worst 10
    print(f"{stat['run']:<15} {stat['samples']:>8} {stat['jumps']:>6} {stat['jump_rate']:>6.1f}% {stat['max_jump']:>9.2f} {stat['mean_diff']:>9.3f}")

print("\n" + "="*60)
print("2. LABEL VARIANCE ANALYSIS")
print("="*60)

# Bin w values and check variance within bins
bins = np.linspace(-3, 3, 21)  # 20 bins
bin_indices = np.digitize(w, bins)

print("\nSteering variance by bin (similar situations should have similar steering):")
print(f"{'W Range':<20} {'Count':>8} {'Std Dev':>10}")
print("-"*40)
for i in range(1, len(bins)):
    mask = bin_indices == i
    if mask.sum() > 10:  # Only analyze bins with enough samples
        w_in_bin = w[mask]
        v_in_bin = v[mask]
        std_w = w_in_bin.std()
        std_v = v_in_bin.std()
        print(f"[{bins[i-1]:>5.1f}, {bins[i]:>5.1f}] {mask.sum():>8} {std_w:>9.3f}")

print("\n" + "="*60)
print("3. VELOCITY-STEERING CORRELATION")
print("="*60)

# Check if velocity is consistent for given steering angles
print("\nAverage velocity for different steering magnitudes:")
print(f"{'|w| Range':<20} {'Count':>8} {'Mean V':>10} {'Std V':>10}")
print("-"*50)

abs_w = np.abs(w)
w_bins = [(0, 0.5, "Straight"), (0.5, 1.5, "Moderate"), (1.5, 2.5, "Sharp"), (2.5, 3.0, "Extreme")]
for low, high, label in w_bins:
    mask = (abs_w >= low) & (abs_w < high)
    if mask.sum() > 0:
        mean_v = v[mask].mean()
        std_v = v[mask].std()
        print(f"{label:<20} {mask.sum():>8} {mean_v:>9.3f} {std_v:>9.3f}")

print("\n" + "="*60)
print("4. DATA QUALITY SCORE")
print("="*60)

# Calculate overall consistency score
avg_jump_rate = np.mean([s['jump_rate'] for s in run_stats])
overall_w_std = w.std()

print(f"\nAverage temporal jump rate: {avg_jump_rate:.1f}%")
print(f"Overall w standard deviation: {overall_w_std:.3f}")

if avg_jump_rate < 5:
    print("✓ Temporal consistency: GOOD")
elif avg_jump_rate < 15:
    print("⚠ Temporal consistency: MODERATE (some jerky driving)")
else:
    print("✗ Temporal consistency: POOR (very inconsistent steering)")

print("\n" + "="*60)
print("5. RECOMMENDATIONS")
print("="*60)

if avg_jump_rate > 10:
    print("\n⚠  HIGH INCONSISTENCY DETECTED")
    print("   Consider:")
    print("   - Removing runs with jump_rate > 15%")
    print("   - Collecting more data with smoother driving")
    print(f"   - Worst offenders: {', '.join([s['run'] for s in run_stats[:3]])}")
else:
    print("\n✓ Data quality looks reasonable")
    print("   The model plateau is likely due to:")
    print("   - Inherent task difficulty (line following is hard)")
    print("   - Camera/track lighting variations")
    print("   - Model capacity limitations")
