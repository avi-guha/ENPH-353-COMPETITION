import torch
import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # Image Branch
        self.features = nn.Sequential(
            # Input: 66x200x3 (YUV or RGB)
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Image output size: 1152
        
        # LIDAR Branch
        # Input: 720 ranges
        self.lidar_branch = nn.Sequential(
            nn.Linear(720, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion
        # 1152 (Image) + 64 (LIDAR) = 1216
        self.classifier = nn.Sequential(
            nn.Linear(1216, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
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
