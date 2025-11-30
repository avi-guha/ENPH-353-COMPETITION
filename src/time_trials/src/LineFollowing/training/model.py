import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """
    Encodes RGB images (B, 3, 120, 120) into a feature vector.
    Adapted from NVIDIA PilotNet for 120x120 square input.
    Includes Batch Normalization and Dropout.
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        # Input: (B, 3, 120, 120)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(24)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(36)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(48)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Flatten size calculation: 64 * 8 * 8 = 4096
        self.flat_size = 4096
        
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.dropout1 = nn.Dropout(0.3) # Increased dropout for regularization
        self.fc2 = nn.Linear(512, feature_dim)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 3, 120, 120)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return x

class LidarEncoder(nn.Module):
    """
    Encodes 1D LiDAR scans (B, 1, 720) into a feature vector using 1D Convolutions.
    """
    def __init__(self, input_channels=1, input_length=720, feature_dim=64):
        super().__init__()
        
        # 1D CNN to extract geometric features from range data
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Adaptive pooling allows flexibility in input length if needed, 
        # but mainly reduces dimensionality efficiently.
        self.global_pool = nn.AdaptiveAvgPool1d(4) # Reduces to 4 spatial features per channel
        
        flat_size = 64 * 4
        self.fc = nn.Linear(flat_size, feature_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (B, 1, 720) or (B, 720) -> unsqueeze if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x

class MultiModalPolicyNet(nn.Module):
    """
    Fusion network combining Image and LiDAR encoders.
    """
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.image_encoder = ImageEncoder(feature_dim=256)
        self.lidar_encoder = LidarEncoder(feature_dim=64)
        
        # Fusion Head
        # Input dim = sum of encoder output dims
        fusion_input_dim = 256 + 64 
        
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.head = nn.Linear(64, 2) # Output: [v_cmd, w_cmd]

    def forward(self, img, lidar):
        """
        img: (B, 3, 120, 120)
        lidar: (B, 720)
        """
        img_feat = self.image_encoder(img)
        lidar_feat = self.lidar_encoder(lidar)
        
        # Concatenate features
        combined = torch.cat([img_feat, lidar_feat], dim=1)
        
        # Fusion layers
        x = F.relu(self.fusion_fc1(combined))
        x = self.dropout1(x)
        x = F.relu(self.fusion_fc2(x))
        x = self.dropout2(x)
        
        # Output
        output = self.head(x)
        return output
