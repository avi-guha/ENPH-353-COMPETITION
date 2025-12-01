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
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), # Output: 27x27x36
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), # Output: 12x12x48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), # Output: 10x10x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 8x8x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Image output size: 64 * 8 * 8 = 4096
        
        # Classifier
        # Input: 4096 (Image only)
        self.classifier = nn.Sequential(
            nn.Linear(4096, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 2) # v, w
        )

    def forward(self, x_img):
        img_out = self.features(x_img)
        output = self.classifier(img_out)
        return output
