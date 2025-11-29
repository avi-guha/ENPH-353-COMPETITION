#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
import ast
from model_xLidar import PilotNet

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
        
        # Augmentation
        if self.augment:
            # 1. Horizontal Flip (50%)
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                w = -w
            
            # 2. Brightness/Contrast (30%)
            if np.random.rand() < 0.3:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2) # Contrast
                beta = np.random.uniform(-20, 20) # Brightness
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                
            # 3. Gaussian Noise (30%)
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
                image = cv2.add(image, noise)

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
        
        return image, label

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5 # Lower learning rate for stability
    EPOCHS = 50
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_xLidar.pth')
    
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
    
    # Filter out stopped data (v=0 AND w=0)
    initial_len = len(full_dataframe)
    full_dataframe = full_dataframe[(full_dataframe.iloc[:, 1] != 0) | (full_dataframe.iloc[:, 2] != 0)]
    print(f"Removed {initial_len - len(full_dataframe)} stopped samples (v=0 and w=0).")
    
    # Filter out negative linear velocity (v < 0)
    initial_len = len(full_dataframe)
    full_dataframe = full_dataframe[full_dataframe.iloc[:, 1] >= 0]
    print(f"Removed {initial_len - len(full_dataframe)} samples with negative velocity (v < 0).")

    # Filter out risky maneuvers (v > 1.5 AND |w| > 2.0)
    initial_len = len(full_dataframe)
    # Keep samples where NOT (v > 1.5 AND |w| > 2.0)
    full_dataframe = full_dataframe[~((full_dataframe.iloc[:, 1] > 1.5) & (np.abs(full_dataframe.iloc[:, 2]) > 2.0))]
    print(f"Removed {initial_len - len(full_dataframe)} risky samples (v > 1.5 and |w| > 2.0).")
    
    print(f"Valid samples after filtering: {len(full_dataframe)}")
    
    if len(full_dataframe) == 0:
        print("Error: No valid samples found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split into train/val
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
    y_train = train_df.iloc[:, 2].values # w values
    
    # Bin the continuous w values
    bins = np.linspace(-3, 3, 21) # 20 bins
    binned_w = np.digitize(y_train, bins)
    
    # Count samples per bin
    bin_counts = np.bincount(binned_w)
    
    # Calculate weight per bin (inverse frequency)
    bin_weights = np.zeros_like(bin_counts, dtype=np.float32)
    bin_weights[bin_counts > 0] = 1.0 / bin_counts[bin_counts > 0]
    
    # Assign weight to each sample
    sample_weights = bin_weights[binned_w]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Use sampler for training
    # num_workers=4 allows loading data in parallel processes
    # pin_memory=True speeds up transfer to CUDA
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = PilotNet().to(device)
    
    # Load existing model if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("No existing model found, starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Weighted Loss
            loss_v = nn.functional.mse_loss(outputs[:, 0], labels[:, 0])
            loss_w = nn.functional.mse_loss(outputs[:, 1], labels[:, 1])
            
            # Weight w loss equal to v loss
            loss = loss_v + loss_w
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_loss_v = 0.0
            val_loss_w = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    
                    loss_v = nn.functional.mse_loss(outputs[:, 0], labels[:, 0])
                    loss_w = nn.functional.mse_loss(outputs[:, 1], labels[:, 1])
                    loss = loss_v + loss_w
                    
                    val_loss += loss.item()
                    val_loss_v += loss_v.item()
                    val_loss_w += loss_w.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_v = val_loss_v / len(val_loader)
            avg_val_loss_w = val_loss_w / len(val_loader)
            
            print(f"Val Loss: {avg_val_loss:.4f} (v: {avg_val_loss_v:.4f}, w: {avg_val_loss_w:.4f})")
            
            # Step the scheduler
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Saved best model to {MODEL_PATH}")
        else:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}")

    print("Training complete.")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == '__main__':
    train()
