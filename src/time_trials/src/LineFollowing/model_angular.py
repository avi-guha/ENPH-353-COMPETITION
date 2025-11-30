#!/usr/bin/env python3

import torch
import torch.nn as nn

class PilotNet_AngularOnly(nn.Module):
    def __init__(self):
        super(PilotNet_AngularOnly, self).__init__()
        
        # Image Branch - Using ELU for smoother gradients
        self.features = nn.Sequential(
            # Input: 120x120x3 (YUV)
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Image output size: 64 * 8 * 8 = 4096
        
        # LIDAR Branch
        self.lidar_branch = nn.Sequential(
            nn.Linear(720, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        
        # Fusion - Simplified classifier for single output
        # 4096 (Image) + 64 (LIDAR) = 4160
        self.classifier = nn.Sequential(
            nn.Linear(4160, 100),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),  # Single output: w (angular velocity)
            nn.Tanh()  # Constrain output to [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_lidar):
        img_out = self.features(x_img)
        lidar_out = self.lidar_branch(x_lidar)
        
        # Concatenate
        combined = torch.cat((img_out, lidar_out), dim=1)
        
        output = self.classifier(combined)
        return output  # Shape: [batch, 1]
