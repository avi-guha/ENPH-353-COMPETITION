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
from model_vision import PilotNetVision

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
            # Fallback or error
            # For robustness, let's just return a zero tensor if image fails (shouldn't happen)
            image = np.zeros((120, 120, 3), dtype=np.uint8)
        
        v = row['v']
        w = row['w']
        
        # Augmentation
        if self.augment and np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            w = -w
            
        # Preprocess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (120, 120))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # Normalize labels
        # w: [-3, 3] -> [-1, 1]
        w_norm = w / 3.0
        
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor([w_norm], dtype=torch.float32)
        )

def train():
    # STANDARD HYPERPARAMETERS (No fancy stuff)
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_model_vision_angular.pth')
    
    # Load Data
    all_data = []
    for root, dirs, files in os.walk(DATA_DIR):
        if 'log.csv' in files:
            try:
                df = pd.read_csv(os.path.join(root, 'log.csv'))
                if len(df) > 10:
                    df['base_path'] = root
                    all_data.append(df)
                    print(f"Loaded {len(df)} samples from {os.path.basename(root)}")
            except:
                pass
                
    if not all_data:
        print("No data found!")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Basic Filtering
    full_df = full_df[(full_df['v'] != 0) | (full_df['w'] != 0)]
    print(f"Total training samples: {len(full_df)}")
    
    # Split
    indices = np.random.permutation(len(full_df))
    split = int(0.8 * len(full_df))
    train_df = full_df.iloc[indices[:split]]
    val_df = full_df.iloc[indices[split:]]
    
    train_loader = DataLoader(DrivingDataset(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(DrivingDataset(val_df, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = PilotNetVision().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(avg_val)
        
        # Real RMSE
        w_rmse = np.sqrt(avg_val) * 3.0
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}, w_RMSE={w_rmse:.3f} rad/s")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> Saved Best Model")

if __name__ == '__main__':
    train()
