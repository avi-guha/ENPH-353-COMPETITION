#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
import ast
from model_angular import PilotNet_AngularOnly

class DrivingDataset_AngularOnly(Dataset):
    def __init__(self, dataframe, transform=None, augment=False):
        self.annotations = dataframe
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Image path is relative to the log file's directory (base_path)
        base_path = self.annotations.iloc[index]['base_path']
        rel_path = self.annotations.iloc[index]['image_path']
        img_path = os.path.join(base_path, rel_path)
        image = cv2.imread(img_path)
        
        # Get angular velocity only
        w = self.annotations.iloc[index, 2]
        
        # Get LIDAR
        scan_str = self.annotations.iloc[index, 3]
        scan = ast.literal_eval(scan_str)
        scan = np.array(scan, dtype=np.float32)

        # Augmentation
        if self.augment:
            # 1. Horizontal Flip (50%)
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                w = -w
                scan = scan[::-1].copy()
            
            # 2. Brightness/Contrast (30%)
            if np.random.rand() < 0.3:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                beta = np.random.uniform(-20, 20)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                
            # 3. Gaussian Noise (20%)
            if np.random.rand() < 0.2:
                noise = np.random.normal(0, 8, image.shape)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # 4. Color jitter (20%)
            if np.random.rand() < 0.2:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Convert BGR to YUV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (120, 120))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        # Normalize w to [-1, 1]
        label = torch.tensor(w / 3.0, dtype=torch.float32)  # Single value, not a list
        
        scan = torch.tensor(scan, dtype=torch.float32)
        scan = scan / 30.0

        return image, scan, label

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    
    # Normalization constant
    W_SCALE = 3.0  # w range: [-3, 3]
    
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_angular.pth')
    
    # Collect all log files
    all_data = []
    
    # Exclude runs with high temporal inconsistency (>15% jump rate)
    BAD_RUNS = ['run_7', 'run_29', 'run_26', 'run_27']
    
    for root, dirs, files in os.walk(DATA_DIR):
        if 'log.csv' in files:
            run_name = os.path.basename(root)
            
            # Skip bad runs
            if run_name in BAD_RUNS:
                print(f"SKIPPING {run_name} (high temporal inconsistency)")
                continue
            
            csv_path = os.path.join(root, 'log.csv')
            try:
                df = pd.read_csv(csv_path)
                df['base_path'] = root
                all_data.append(df)
                print(f"Loaded {len(df)} samples from {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

    if not all_data:
        print(f"Error: No log.csv files found in {DATA_DIR}")
        return

    full_dataframe = pd.concat(all_data, ignore_index=True)
    print(f"Total samples loaded: {len(full_dataframe)}")
    
    # Filter out samples with empty scans
    def has_valid_scan(scan_str):
        try:
            scan = ast.literal_eval(scan_str)
            return len(scan) > 0
        except:
            return False
    
    full_dataframe = full_dataframe[full_dataframe.iloc[:, 3].apply(has_valid_scan)]
    
    # Filter out stopped data (v=0 AND w=0)
    initial_len = len(full_dataframe)
    full_dataframe = full_dataframe[(full_dataframe.iloc[:, 1] != 0) | (full_dataframe.iloc[:, 2] != 0)]
    print(f"Removed {initial_len - len(full_dataframe)} stopped samples.")
    
    # Filter out negative linear velocity
    initial_len = len(full_dataframe)
    full_dataframe = full_dataframe[full_dataframe.iloc[:, 1] >= 0]
    print(f"Removed {initial_len - len(full_dataframe)} negative velocity samples.")

    # Risky maneuver filtering
    initial_len = len(full_dataframe)
    full_dataframe = full_dataframe[~((full_dataframe.iloc[:, 1] > 1.5) & (np.abs(full_dataframe.iloc[:, 2]) > 2.0))]
    print(f"Removed {initial_len - len(full_dataframe)} risky samples.")
    
    print(f"Valid samples after filtering: {len(full_dataframe)}")
    
    if len(full_dataframe) == 0:
        print("Error: No valid samples found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split into train/val
    train_size = int(0.8 * len(full_dataframe))
    
    indices = np.arange(len(full_dataframe))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_df = full_dataframe.iloc[train_indices]
    val_df = full_dataframe.iloc[val_indices]
    
    train_dataset = DrivingDataset_AngularOnly(dataframe=train_df, augment=True)
    val_dataset = DrivingDataset_AngularOnly(dataframe=val_df, augment=False)

    # NO WeightedRandomSampler - let model learn natural distribution
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = PilotNet_AngularOnly().to(device)
    
    # Load existing model if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except RuntimeError as e:
            print(f"Model architecture changed, starting fresh: {e}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, scans, labels in train_loader:
            images = images.to(device)
            scans = scans.to(device)
            labels = labels.to(device).unsqueeze(1)  # Shape: [batch, 1]

            optimizer.zero_grad()
            outputs = model(images, scans)  # Shape: [batch, 1]
            
            # Simple MSE loss on angular velocity only
            loss = nn.functional.mse_loss(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, scans, labels in val_loader:
                    images = images.to(device)
                    scans = scans.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    
                    outputs = model(images, scans)
                    loss = nn.functional.mse_loss(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Denormalize for interpretability
            real_w_error = np.sqrt(avg_val_loss) * W_SCALE
            
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"  -> Real angular error: w={real_w_error:.3f} rad/s ({real_w_error*57.3:.1f}°/s)")
            
            # Step scheduler
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"✓ Saved best model to {MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    print("Training complete.")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == '__main__':
    train()
