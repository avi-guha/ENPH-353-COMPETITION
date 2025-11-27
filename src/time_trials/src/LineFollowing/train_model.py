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
from model import PilotNet

class DrivingDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.annotations = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Image path is relative to the log file's directory (base_path)
        base_path = self.annotations.iloc[index]['base_path']
        rel_path = self.annotations.iloc[index]['image_path']
        img_path = os.path.join(base_path, rel_path)
        image = cv2.imread(img_path)
        # Convert BGR to YUV (NVIDIA paper uses YUV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Resize to 120x120 (Square aspect ratio)
        image = cv2.resize(image, (120, 120))
        # Normalize
        image = image / 255.0
        # Transpose to Channel First (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        v = self.annotations.iloc[index, 1]
        w = self.annotations.iloc[index, 2]
        label = torch.tensor([v, w], dtype=torch.float32)
        
        # LIDAR
        scan_str = self.annotations.iloc[index, 3]
        scan = ast.literal_eval(scan_str)
        scan = torch.tensor(scan, dtype=torch.float32)
        # Normalize LIDAR (0-30 range -> 0-1)
        scan = scan / 30.0

        return image, scan, label

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5 # Lower learning rate for stability
    EPOCHS = 50
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
    
    print(f"Valid samples after filtering: {len(full_dataframe)}")
    
    if len(full_dataframe) == 0:
        print("Error: No valid samples found. Please recollect data with scan data enabled.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = DrivingDataset(dataframe=full_dataframe)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if val_size == 0:
        print("Warning: Not enough data for validation. Training on all data.")
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = None
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PilotNet().to(device)
    
    # Load existing model if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("No existing model found, starting from scratch.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, scans, labels in train_loader:
            images = images.to(device)
            scans = scans.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, scans)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, scans, labels in val_loader:
                    images = images.to(device)
                    scans = scans.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images, scans)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Val Loss: {avg_val_loss:.4f}")
            
            # Step the scheduler
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Saved best model to {MODEL_PATH}")
        else:
            # Save last model if no validation
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}")

    print("Training complete.")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == '__main__':
    train()
