import torch
import argparse
import numpy as np
from model import MultiModalPolicyNet
from dataset import get_dataloader

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    _, val_loader = get_dataloader(args.data_dir, batch_size=1)
    
    # Load Model
    model = MultiModalPolicyNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Metrics
    mae_v = 0.0
    mae_w = 0.0
    count = 0
    
    MAX_V = 2.5
    MAX_W = 3.5
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for images, lidars, targets in val_loader:
            images = images.to(device)
            lidars = lidars.to(device)
            targets = targets.to(device)
            
            # Predict
            outputs = model(images, lidars)
            
            # Denormalize predictions
            v_pred = outputs[:, 0] * MAX_V
            w_pred = outputs[:, 1] * MAX_W
            
            # Targets are already raw in dataset, so use them directly
            v_true = targets[:, 0]
            w_true = targets[:, 1]
            
            # Accumulate Error
            mae_v += torch.abs(v_pred - v_true).item()
            mae_w += torch.abs(w_pred - w_true).item()
            count += 1
            
    print(f"Evaluated on {count} samples.")
    print(f"MAE Linear Velocity: {mae_v/count:.4f} m/s")
    print(f"MAE Angular Velocity: {mae_w/count:.4f} rad/s")

if __name__ == "__main__":
    # Resolve data directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '../data')
    default_model_path = os.path.join(script_dir, '../models/best_model.pth')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Path to data directory')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to trained model')
    args = parser.parse_args()
    
    evaluate(args)
