#!/usr/bin/env python3

import torch
import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # Image Branch (no BatchNorm - matches saved checkpoint)
        self.features = nn.Sequential(
            # Input: 120x120x3 (YUV or RGB)
            nn.Conv2d(3, 24, kernel_size=5, stride=2), # 0: Output: 58x58x24
            nn.ReLU(),                                  # 1
            nn.Conv2d(24, 36, kernel_size=5, stride=2), # 2: Output: 27x27x36
            nn.ReLU(),                                  # 3
            nn.Conv2d(36, 48, kernel_size=5, stride=2), # 4: Output: 12x12x48
            nn.ReLU(),                                  # 5
            nn.Conv2d(48, 64, kernel_size=3),           # 6: Output: 10x10x64
            nn.ReLU(),                                  # 7
            nn.Conv2d(64, 64, kernel_size=3),           # 8: Output: 8x8x64
            nn.ReLU(),                                  # 9
            nn.Flatten()                                # 10
        )
        
        # Image output size: 64 * 8 * 8 = 4096
        
        # LIDAR Branch (no BatchNorm - matches saved checkpoint)
        # Input: 720 ranges
        self.lidar_branch = nn.Sequential(
            nn.Linear(720, 128),  # 0
            nn.ReLU(),            # 1
            nn.Linear(128, 64),   # 2
            nn.ReLU()             # 3
        )
        
        # Fusion (no BatchNorm - matches saved checkpoint)
        # 4096 (Image) + 64 (LIDAR) = 4160
        self.classifier = nn.Sequential(
            nn.Linear(4160, 100), # 0
            nn.ReLU(),            # 1
            nn.Dropout(0.5),      # 2
            nn.Linear(100, 50),   # 3
            nn.ReLU(),            # 4
            nn.Dropout(0.5),      # 5
            nn.Linear(50, 10),    # 6
            nn.ReLU(),            # 7
            nn.Linear(10, 2)      # 8: v, w
        )

    def forward(self, x_img, x_lidar):
        img_out = self.features(x_img)
        lidar_out = self.lidar_branch(x_lidar)
        
        # Concatenate
        combined = torch.cat((img_out, lidar_out), dim=1)
        
        output = self.classifier(combined)
        return output
