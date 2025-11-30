import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from model import MultiModalPolicyNet
from dataset import get_dataloader

# Hyperparameters
BATCH_SIZE = 128 # Increased for RTX 4080 Super
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
LAMBDA_W = 5.0 # Weight for angular velocity loss
LAMBDA_W = 5.0 # Weight for angular velocity loss
DATA_DIR = '../data' # Relative to this script
MODELS_DIR = '../models' # Relative to this script

class DrivingLoss(nn.Module):
    def __init__(self, lambda_w=5.0):
        super().__init__()
        self.lambda_w = lambda_w
        self.criterion = nn.SmoothL1Loss()

    def forward(self, pred, target):
        # pred: (B, 2) -> [v_pred, w_pred]
        # target: (B, 2) -> [v_true, w_true]
        
        # Max values from dataset.py
        MAX_V = 2.5
        MAX_W = 3.5
        
        v_target_norm = target[:, 0] / MAX_V
        w_target_norm = target[:, 1] / MAX_W
        
        # Model outputs
        v_pred_norm = pred[:, 0]
        w_pred_norm = pred[:, 1]
        
        v_loss = self.criterion(v_pred_norm, v_target_norm)
        w_loss = self.criterion(w_pred_norm, w_target_norm)
        
        total_loss = v_loss + (self.lambda_w * w_loss)
        return total_loss, v_loss, w_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable cudnn benchmark for optimized performance on fixed input sizes
    torch.backends.cudnn.benchmark = True

    # Data
    # Increased num_workers for faster data loading
    train_loader, val_loader = get_dataloader(args.data_dir, batch_size=BATCH_SIZE)
    
    # Model
    model = MultiModalPolicyNet().to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Loss
    criterion = DrivingLoss(lambda_w=LAMBDA_W)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        running_v_loss = 0.0
        running_w_loss = 0.0
        
        for batch_idx, (images, lidars, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            lidars = lidars.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                outputs = model(images, lidars)
                loss, v_loss, w_loss = criterion(outputs, targets)
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            
            # Unscale for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Scaler Step
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            running_v_loss += v_loss.item()
            running_w_loss += w_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for images, lidars, targets in val_loader:
                images = images.to(device, non_blocking=True)
                lidars = lidars.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # No need for autocast in validation usually, but consistent is fine
                outputs = model(images, lidars)
                loss, _, _ = criterion(outputs, targets)
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        
        # Step Scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Checkpoint (for resuming)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }
        
        # Ensure models dir exists
        os.makedirs(args.models_dir, exist_ok=True)
        
        torch.save(checkpoint, os.path.join(args.models_dir, "checkpoint.pth"))
        
        # Save Best Model (weights only)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.models_dir, "best_model.pth"))
            print(f">>> Saved Best Model to {args.models_dir}")

if __name__ == "__main__":
    # Resolve data directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '../data')
    default_models_dir = os.path.join(script_dir, '../models')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default=default_models_dir, help='Path to save models')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(args)
