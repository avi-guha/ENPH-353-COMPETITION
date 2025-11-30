import torch
import argparse
import os
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
    mae_w = 0.0
    mse_w = 0.0
    count = 0
    
    MAX_W = 3.5
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for images, lidars, targets in val_loader:
            images = images.to(device)
            lidars = lidars.to(device)
            targets = targets.to(device)
            
            # Predict (single omega value)
            outputs = model(images)
            
            # Denormalize prediction
            w_pred = outputs[:, 0] * MAX_W
            
            # Target angular velocity
            w_true = targets[:, 1]
            
            # Accumulate Error
            mae_w += torch.abs(w_pred - w_true).item()
            mse_w += ((w_pred - w_true) ** 2).item()
            count += 1
            
    print(f"\n{'='*50}")
    print(f"Evaluated on {count} samples")
    print(f"{'='*50}")
    print(f"MAE Angular Velocity: {mae_w/count:.4f} rad/s ({(mae_w/count)*57.3:.1f}°/s)")
    print(f"RMSE Angular Velocity: {np.sqrt(mse_w/count):.4f} rad/s ({np.sqrt(mse_w/count)*57.3:.1f}°/s)")
    print(f"MSE Loss (normalized): {(mse_w/count)/(MAX_W**2):.6f}")
    print(f"{'='*50}\n")

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
