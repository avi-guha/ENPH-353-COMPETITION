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
from model import PilotNet

class DrivingDataset(Dataset):
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
        
        # Get labels
        v = self.annotations.iloc[index, 1]
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
                scan = scan[::-1].copy() # Reverse LIDAR scan
            
            # 2. Brightness/Contrast (30%)
            if np.random.rand() < 0.3:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2) # Contrast
                beta = np.random.uniform(-20, 20) # Brightness
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                
            # 3. Gaussian Noise (20%) - Fixed to avoid overflow
            if np.random.rand() < 0.2:
                noise = np.random.normal(0, 8, image.shape)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # 4. Color jitter (20%) - Slight hue/saturation shifts
            if np.random.rand() < 0.2:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Convert BGR to YUV (NVIDIA paper uses YUV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Resize to 120x120 (Square aspect ratio)
        image = cv2.resize(image, (120, 120))
        # Normalize
        image = image / 255.0
        # Transpose to Channel First (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        label = torch.tensor([v, w], dtype=torch.float32)
        
        # Normalize labels to [-1, 1] for training
        # v: [-2, 2] -> [-1, 1] (divide by 2)
        # w: [-3, 3] -> [-1, 1] (divide by 3)
        label[0] = label[0] / 2.0  # v
        label[1] = label[1] / 3.0  # w
        
        scan = torch.tensor(scan, dtype=torch.float32)
        # Normalize LIDAR (0-30 range -> 0-1)
        scan = scan / 30.0

        return image, scan, label

def train():
    # Hyperparameters
    BATCH_SIZE = 64  # Larger batch for more stable gradients
    LEARNING_RATE = 3e-4  # Slightly higher LR with cosine annealing
    EPOCHS = 150  # More epochs with early stopping
    WEIGHT_DECAY = 1e-4  # L2 regularization
    
    # Normalization constants for outputs
    V_SCALE = 2.0  # v range: [-2, 2]
    W_SCALE = 3.0  # w range: [-3, 3]
    
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_model.pth')
    
    # Collect all log files
    all_data = []
    for root, dirs, files in os.walk(DATA_DIR):
        if 'log.csv' in files:
            csv_path = os.path.join(root, 'log.csv')
            try:
                df = pd.read_csv(csv_path)
                # Store the directory of this log file to resolve relative image paths later
                df['base_path'] = root
                all_data.append(df)
                print(f"Loaded {len(df)} samples from {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

    if not all_data:
        print(f"Error: No log.csv files found in {DATA_DIR} or its subdirectories.")
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
    # We want to keep frames where robot is turning even if v=0 (though usually v>0 when turning)
    # User requested: "Get rid of any frames in which velocity and angular velocity is 0"
    initial_len = len(full_dataframe)
    full_dataframe = full_dataframe[(full_dataframe.iloc[:, 1] != 0) | (full_dataframe.iloc[:, 2] != 0)]
    print(f"Removed {initial_len - len(full_dataframe)} stopped samples (v=0 and w=0).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split into train/val
    # We create two datasets: one for training (with augmentation) and one for validation (without)
    train_size = int(0.8 * len(full_dataframe))
    val_size = len(full_dataframe) - train_size
    
    # Shuffle indices
    indices = np.arange(len(full_dataframe))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_df = full_dataframe.iloc[train_indices]
    val_df = full_dataframe.iloc[val_indices]
    
    train_dataset = DrivingDataset(dataframe=train_df, augment=True)
    val_dataset = DrivingDataset(dataframe=val_df, augment=False)

    # Calculate weights for WeightedRandomSampler
    # We want to balance the distribution of w
    y_train = train_df.iloc[:, 2].values # w values
    
    # Bin the continuous w values
    bins = np.linspace(-3, 3, 21) # 20 bins
    binned_w = np.digitize(y_train, bins)
    
    # Count samples per bin
    bin_counts = np.bincount(binned_w)
    
    # Calculate weight per bin (inverse frequency)
    # Avoid division by zero for empty bins
    bin_weights = np.zeros_like(bin_counts, dtype=np.float32)
    bin_weights[bin_counts > 0] = 1.0 / bin_counts[bin_counts > 0]
    
    # Assign weight to each sample
    sample_weights = bin_weights[binned_w]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Use sampler for training (shuffle must be False when using sampler)
    # num_workers=8 allows loading data in parallel processes
    # pin_memory=True speeds up transfer to CUDA
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = PilotNet().to(device)
    
    # Load existing model if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except RuntimeError as e:
            print(f"Model architecture changed, starting fresh: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing with Warm Restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20  # Stop if no improvement for 20 epochs

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, scans, labels in train_loader:
            images = images.to(device)
            scans = scans.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, scans)
            
            # Weighted Loss - angular velocity is more important for line following
            # outputs: [batch, 2], labels: [batch, 2]
            # 0: v, 1: w
            loss_v = nn.functional.mse_loss(outputs[:, 0], labels[:, 0])
            loss_w = nn.functional.mse_loss(outputs[:, 1], labels[:, 1])
            
            # Weight w loss higher - steering is critical for line following
            loss = loss_v + 2.0 * loss_w
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
        
        # Step scheduler
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_loss_v = 0.0
            val_loss_w = 0.0
            
            with torch.no_grad():
                for images, scans, labels in val_loader:
                    images = images.to(device)
                    scans = scans.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images, scans)
                    
                    loss_v = nn.functional.mse_loss(outputs[:, 0], labels[:, 0])
                    loss_w = nn.functional.mse_loss(outputs[:, 1], labels[:, 1])
                    loss = loss_v + loss_w
                    
                    val_loss += loss.item()
                    val_loss_v += loss_v.item()
                    val_loss_w += loss_w.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_v = val_loss_v / len(val_loader)
            avg_val_loss_w = val_loss_w / len(val_loader)
            
            # Denormalize for interpretability (approximate real-world error)
            # MSE is computed on normalized values [-1, 1]
            # To get real error: sqrt(mse) * scale  
            real_v_error = np.sqrt(avg_val_loss_v) * V_SCALE
            real_w_error = np.sqrt(avg_val_loss_w) * W_SCALE
            
            print(f"Val Loss: {avg_val_loss:.4f} (v: {avg_val_loss_v:.4f}, w: {avg_val_loss_w:.4f})")
            print(f"  -> Real errors: v={real_v_error:.3f} m/s, w={real_w_error:.3f} rad/s ({real_w_error*57.3:.1f}°/s)")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"✓ Saved best model to {MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                    break
        else:
            # Save last model if no validation
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}")

    print("Training complete.")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == '__main__':
    train()
