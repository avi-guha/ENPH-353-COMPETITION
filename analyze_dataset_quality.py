import pandas as pd
import numpy as np
import os
import cv2
import ast

def analyze_quality():
    script_dir = r'src/time_trials/src/LineFollowing'
    data_dir = os.path.join(script_dir, 'data')
    
    print(f"Scanning data in {data_dir}...")
    
    all_data = []
    for root, dirs, files in os.walk(data_dir):
        if 'log.csv' in files:
            csv_path = os.path.join(root, 'log.csv')
            try:
                df = pd.read_csv(csv_path)
                df['base_path'] = root
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
    
    if not all_data:
        print("No data found.")
        return

    df = pd.concat(all_data, ignore_index=True)
    print(f"Total raw samples: {len(df)}")
    
    # 1. Check for NaNs/Infs
    if df.isnull().values.any():
        print("WARNING: Dataset contains NaN values!")
        print(df.isnull().sum())
    
    # 2. Check Value Ranges
    v = df.iloc[:, 1]
    w = df.iloc[:, 2]
    
    print("\n--- Value Ranges ---")
    print(f"v range: [{v.min():.2f}, {v.max():.2f}]")
    print(f"w range: [{w.min():.2f}, {w.max():.2f}]")
    
    outliers_w = df[(w > 3.0) | (w < -3.0)]
    if len(outliers_w) > 0:
        print(f"WARNING: Found {len(outliers_w)} samples with |w| > 3.0")
        
    # 3. Check for "Impossible" Driving (High Speed + Sharp Turn)
    # Heuristic: If v > 1.5 and |w| > 2.0, it might be sliding/unstable
    risky_maneuvers = df[(v > 1.5) & (np.abs(w) > 2.0)]
    if len(risky_maneuvers) > 0:
        print(f"WARNING: Found {len(risky_maneuvers)} samples with High Speed (v>1.5) AND Sharp Turn (|w|>2.0)")

    # 4. Check Image Quality (Sampled)
    print("\n--- Checking Image Quality (Sampling 500 images) ---")
    sample_indices = np.random.choice(len(df), min(500, len(df)), replace=False)
    
    black_images = 0
    white_images = 0
    corrupt_images = 0
    
    for idx in sample_indices:
        base_path = df.iloc[idx]['base_path']
        rel_path = df.iloc[idx]['image_path']
        img_path = os.path.join(base_path, rel_path)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                corrupt_images += 1
                continue
                
            mean_val = np.mean(img)
            if mean_val < 5: # Almost pitch black
                black_images += 1
            elif mean_val > 250: # Almost pure white
                white_images += 1
                
        except Exception as e:
            corrupt_images += 1
            
    if corrupt_images > 0:
        print(f"CRITICAL: Found {corrupt_images} corrupt/unreadable images in sample!")
    if black_images > 0:
        print(f"WARNING: Found {black_images} pitch black images (camera covered?)")
    if white_images > 0:
        print(f"WARNING: Found {white_images} pure white images (glare?)")
        
    if corrupt_images == 0 and black_images == 0 and white_images == 0:
        print("Image quality check passed (no obvious corruptions in sample).")

    # 5. Check Stopped Frames
    stopped = df[(v == 0) & (w == 0)]
    print(f"\nStopped frames (v=0, w=0): {len(stopped)} ({len(stopped)/len(df)*100:.1f}%)")
    print("Note: Training script filters these out, but high % implies inefficient collection.")

if __name__ == "__main__":
    analyze_quality()
