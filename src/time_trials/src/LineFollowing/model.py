#!/usr/bin/env python3

import torch
import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # Image Branch
        self.features = nn.Sequential(
            # Input: 120x120x3 (YUV or RGB)
            nn.Conv2d(3, 24, kernel_size=5, stride=2), # Output: 58x58x24
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), # Output: 27x27x36
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), # Output: 12x12x48
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), # Output: 10x10x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 8x8x64
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Image output size: 64 * 8 * 8 = 4096
        
        # LIDAR Branch
        # Input: 720 ranges
        self.lidar_branch = nn.Sequential(
            nn.Linear(720, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion
        # 4096 (Image) + 64 (LIDAR) = 4160
        self.classifier = nn.Sequential(
            nn.Linear(4160, 100),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout added
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout added
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2) # v, w
        )

    def forward(self, x_img, x_lidar):
        img_out = self.features(x_img)
        lidar_out = self.lidar_branch(x_lidar)
        
        # Concatenate
        combined = torch.cat((img_out, lidar_out), dim=1)
        
        output = self.classifier(combined)
        return output
