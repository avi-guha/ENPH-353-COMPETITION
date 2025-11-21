import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from model import PilotNet

class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        # Convert BGR to YUV (NVIDIA paper uses YUV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Resize to 200x66
        image = cv2.resize(image, (200, 66))
        # Normalize
        image = image / 255.0
        # Transpose to Channel First (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        v = self.annotations.iloc[index, 1]
        w = self.annotations.iloc[index, 2]
        label = torch.tensor([v, w], dtype=torch.float32)

        return image, label

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4 # Lower learning rate for stability
    EPOCHS = 50
    DATA_DIR = 'data' # Assumes data is in a 'data' folder relative to this script
    CSV_FILE = os.path.join(DATA_DIR, 'log.csv')

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run data_collector.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = DrivingDataset(csv_file=CSV_FILE, root_dir=DATA_DIR)
    
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
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
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print("Saved best model")
        else:
            # Save last model if no validation
             torch.save(model.state_dict(), 'model.pth')

    print("Training complete.")

if __name__ == '__main__':
    train()
