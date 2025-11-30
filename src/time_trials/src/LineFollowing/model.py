#!/usr/bin/env python3

import torch
import torch.nn as nn

class PilotNet(nn.Module):
    """
    Simplified PilotNet for line following.
    Based on NVIDIA's end-to-end driving architecture.
    """
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # Image Branch - Standard CNN feature extractor
        self.features = nn.Sequential(
            # Input: 120x120x3 (YUV)
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # -> 58x58x24
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # -> 27x27x36
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # -> 12x12x48
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),  # -> 10x10x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),  # -> 8x8x64
            nn.ReLU(),
            nn.Flatten()  # -> 4096
        )
        
        # LIDAR Branch - Simple MLP
        self.lidar_branch = nn.Sequential(
            nn.Linear(720, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion and Output
        # 4096 (image) + 32 (lidar) = 4128
        self.head = nn.Sequential(
            nn.Linear(4128, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2)  # v, w (raw output, clamped at inference)
        )

    def forward(self, x_img, x_lidar):
        img_features = self.features(x_img)
        lidar_features = self.lidar_branch(x_lidar)
        combined = torch.cat((img_features, lidar_features), dim=1)
        return self.head(combined)
