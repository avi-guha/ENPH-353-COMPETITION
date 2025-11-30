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
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the run folders (e.g. 'data/')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Recursively find all log.csv files
        self._load_data()
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")

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
            # Default transform
            image = transforms.ToTensor()(image) # [0, 1]
        
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

def get_dataloader(root_dir, batch_size=32, split=0.8):
    full_dataset = DrivingDataset(root_dir)
    
    train_size = int(split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
