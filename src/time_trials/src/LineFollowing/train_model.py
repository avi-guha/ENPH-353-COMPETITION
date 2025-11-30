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
    def __init__(self, dataframe, augment=False):
        self.annotations = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_path = os.path.join(row['base_path'], row['image_path'])
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        v = row['v']
        w = row['w']
        
        scan = np.array(ast.literal_eval(row['scan']), dtype=np.float32)

        # Simple augmentation - only horizontal flip
        if self.augment and np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            w = -w
            scan = scan[::-1].copy()

        # Preprocess image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (120, 120))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW format
        
        # Normalize labels to [-1, 1]
        v_norm = v / 2.0  # v range: [0, 2] -> [0, 1]
        w_norm = w / 3.0  # w range: [-3, 3] -> [-1, 1]
        
        # Normalize LIDAR
        scan = np.clip(scan, 0, 30) / 30.0

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(scan, dtype=torch.float32),
            torch.tensor([v_norm, w_norm], dtype=torch.float32)
        )


def train():
    # Simple hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    EPOCHS = 50
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_model.pth')
    
    # Load all data
    all_data = []
    for root, dirs, files in os.walk(DATA_DIR):
        if 'log.csv' in files:
            csv_path = os.path.join(root, 'log.csv')
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 1:  # Skip empty files
                    df['base_path'] = root
                    all_data.append(df)
                    print(f"Loaded {len(df)} samples from {os.path.basename(root)}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

    if not all_data:
        print("No data found!")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples: {len(full_df)}")
    
    # Minimal filtering - just remove stopped frames
    full_df = full_df[(full_df['v'] != 0) | (full_df['w'] != 0)]
    print(f"After removing stopped frames: {len(full_df)}")

    # Train/val split (80/20)
    indices = np.random.permutation(len(full_df))
    split = int(0.8 * len(full_df))
    train_df = full_df.iloc[indices[:split]]
    val_df = full_df.iloc[indices[split:]]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = DrivingDataset(train_df, augment=True)
    val_dataset = DrivingDataset(val_df, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PilotNet().to(device)
    
    # Load existing model if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("No existing model found, starting fresh")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 0
    max_patience = 10

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_loss_v = 0.0
        train_loss_w = 0.0
        
        for images, scans, labels in train_loader:
            images = images.to(device)
            scans = scans.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, scans)
            
            # Weighted Loss: 5x importance on steering (w)
            # Use MSE for velocity (smooth regression)
            loss_v = nn.functional.mse_loss(outputs[:, 0], labels[:, 0])
            # Use Huber Loss (SmoothL1) for steering to be robust to outliers
            loss_w = nn.functional.smooth_l1_loss(outputs[:, 1], labels[:, 1])
            
            loss = loss_v + 5.0 * loss_w
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_v += loss_v.item()
            train_loss_w += loss_w.item()

        avg_train = train_loss / len(train_loader)
        avg_train_v = train_loss_v / len(train_loader)
        avg_train_w = train_loss_w / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_v = 0.0
        val_loss_w = 0.0
        val_mse_w = 0.0  # Track MSE separately for accurate RMSE reporting
        
        with torch.no_grad():
            for images, scans, labels in val_loader:
                images = images.to(device)
                scans = scans.to(device)
                labels = labels.to(device)
                outputs = model(images, scans)
                
                loss_v = nn.functional.mse_loss(outputs[:, 0], labels[:, 0])
                loss_w = nn.functional.smooth_l1_loss(outputs[:, 1], labels[:, 1])
                mse_w = nn.functional.mse_loss(outputs[:, 1], labels[:, 1]) # For reporting
                
                loss = loss_v + 5.0 * loss_w
                
                val_loss += loss.item()
                val_loss_v += loss_v.item()
                val_loss_w += loss_w.item()
                val_mse_w += mse_w.item()
        
        avg_val = val_loss / len(val_loader)
        avg_val_v = val_loss_v / len(val_loader)
        avg_val_w = val_loss_w / len(val_loader)
        avg_val_mse_w = val_mse_w / len(val_loader)
        
        scheduler.step(avg_val)
        
        # Real-world error estimate (approximate)
        # MSE is on normalized values. 
        # v_norm = v/2, w_norm = w/3
        # Real RMSE = sqrt(MSE) * scale
        v_rmse = np.sqrt(avg_val_v) * 2.0
        w_rmse = np.sqrt(avg_val_mse_w) * 3.0
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train:.4f} (v: {avg_train_v:.4f}, w: {avg_train_w:.4f})")
        print(f"  Val Loss:   {avg_val:.4f} (v: {avg_val_v:.4f}, w: {avg_val_w:.4f})")
        print(f"  -> Real RMSE: v={v_rmse:.2f} m/s, w={w_rmse:.2f} rad/s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Saved best model")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == '__main__':
    train()
