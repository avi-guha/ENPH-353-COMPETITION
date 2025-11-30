import torch
from torch.utils.data import Dataset
import os
import csv
import cv2
import numpy as np
import ast
from torchvision import transforms

# Constants
IMG_H, IMG_W = 120, 120
MAX_LIDAR_DIST = 30.0
MAX_V = 2.5 # Observed max is 2.0
MAX_W = 3.5  # Observed max is 3.0

class DrivingDataset(Dataset):
    def __init__(self, root_dir, transform=None, filter_data=False):
        """
        Args:
            root_dir (string): Directory with all the run folders (e.g. 'data/')
            transform (callable, optional): Optional transform to be applied on a sample.
            filter_data (bool): If True, filter out stationary and backwards driving.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Recursively find all log.csv files
        self._load_data()
        
        original_count = len(self.samples)
        print(f"Loaded {original_count} samples from {root_dir}")
        
        if filter_data:
            self._filter_hard_data()
            print(f"After filtering: {len(self.samples)} samples remaining ({original_count - len(self.samples)} removed)")

    def _filter_hard_data(self):
        initial_len = len(self.samples)
        filtered_samples = []
        
        # Import random for straight-line filtering
        import random
        random.seed(42)
        
        for s in self.samples:
            v = s['v']
            w = s['w']
            
            # Filter 1: Backwards driving
            if v < 0:
                continue
                
            # Filter 2: Stationary (or near stationary)
            # Allow if turning significantly (e.g. recovering)
            if v < 0.1 and abs(w) < 0.5:
                continue
                
            # Filter 3: Extreme turns at high speed (unsafe/unrealistic)
            # Relaxed: Only filter if REALLY fast and REALLY sharp
            if v > 2.0 and abs(w) > 3.0:
                continue
                
            # Filter 4: Downsample straight driving (50% removal)
            # Define "straight" as |w| < 0.3 rad/s (~17 deg/s)
            if abs(w) < 0.1:
                # Keep only 50% of straight samples
                if random.random() < 0.5:
                    continue
                
            filtered_samples.append(s)
            
        self.samples = filtered_samples

    def _load_data(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file == 'log.csv':
                    csv_path = os.path.join(root, file)
                    self._parse_csv(csv_path, root)

    def _parse_csv(self, csv_path, run_dir):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header if present (check first char of first line)
            try:
                header = next(reader)
                if header[0] != 'image_path':
                    # If no header, reset pointer (though data_collector writes header)
                    f.seek(0)
                else:
                    pass # Header skipped
            except StopIteration:
                return # Empty file

            for row in reader:
                if len(row) < 4:
                    continue
                
                # row: [image_path, v, w, scan_str]
                rel_img_path = row[0]
                v = float(row[1])
                w = float(row[2])
                scan_str = row[3]
                
                abs_img_path = os.path.join(run_dir, rel_img_path)
                
                # Check if image exists
                if not os.path.exists(abs_img_path):
                    continue
                    
                self.samples.append({
                    'image_path': abs_img_path,
                    'v': v,
                    'w': w,
                    'scan_str': scan_str
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Image
        image = cv2.imread(sample['image_path'])
        if image is None:
            # Handle bad image by returning next one or error
            # For simplicity, just error out or return zeros (but better to fix data)
            raise FileNotFoundError(f"Failed to load image: {sample['image_path']}")
            
        # Resize to target resolution
        image = cv2.resize(image, (IMG_W, IMG_H))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To Tensor and Normalize
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform with Augmentation
            # Only apply augmentation if we are in training mode (which we don't strictly know here, 
            # but usually we want robust features). 
            # Ideally, we pass a different transform for Train vs Val in get_dataloader.
            # But for now, let's add mild augmentation to the default.
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(), # [0, 1]
                # Add noise?
            ])
            image = transform(image)
        
        # 2. Parse LiDAR
        # scan_str is "[1.0, 2.0, ...]"
        try:
            scan = ast.literal_eval(sample['scan_str'])
            scan = np.array(scan, dtype=np.float32)
        except:
            # Fallback for empty or malformed scan
            scan = np.full((720,), MAX_LIDAR_DIST, dtype=np.float32)
            
        # Handle length (ensure 720)
        if len(scan) != 720:
            # Resize or pad?
            # For now, just resize/resample if different, or pad
            # But since we verified it's 720, we just assert/truncate
            if len(scan) > 720:
                scan = scan[:720]
            else:
                scan = np.pad(scan, (0, 720 - len(scan)), 'constant', constant_values=MAX_LIDAR_DIST)
        
        # Normalize LiDAR
        scan = np.clip(scan, 0, MAX_LIDAR_DIST)
        scan = scan / MAX_LIDAR_DIST
        scan_tensor = torch.from_numpy(scan).float()
        
        # 3. State and Targets
        v = sample['v']
        w = sample['w']
        
        target_tensor = torch.tensor([v, w], dtype=torch.float32)
        
        return image, scan_tensor, target_tensor

def get_dataloader(root_dir, batch_size=32, split=0.8, filter_data=False):
    # Define Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # We need to split first, then apply transforms. 
    # But Dataset takes transform in __init__.
    # So we create two datasets with same root but different transforms?
    # No, that would load data twice.
    # Better: Load data once, then split, then wrap in a helper class or just override transform attribute?
    # Overriding attribute is risky with random_split.
    
    # Simple approach: Create two instances. It's fast since we just parse CSVs.
    # Apply filtering ONLY to training data usually? Or both?
    # Usually we want validation to represent reality, but if reality has garbage, we filter both.
    # Let's filter both for now to ensure clean metrics.
    train_dataset_full = DrivingDataset(root_dir, transform=train_transform, filter_data=filter_data)
    val_dataset_full = DrivingDataset(root_dir, transform=val_transform, filter_data=filter_data)
    
    # Ensure consistent split
    # We use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    # Get indices
    total_size = len(train_dataset_full)
    train_size = int(split * total_size)
    val_size = total_size - train_size
    
    # We can't easily split indices and map to different datasets with random_split.
    # Instead, let's use Subset.
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader
