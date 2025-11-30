import torch
import torch.nn as nn

class PilotNetVision(nn.Module):
    """
    Standard NVIDIA PilotNet architecture (Vision Only).
    No LIDAR, no complex fusion. Just images -> steering.
    """
    def __init__(self):
        super(PilotNetVision, self).__init__()
        
        # Input: 120x120x3 (YUV)
        self.features = nn.Sequential(
            # Normalization is done in preprocessing, but we can add a BN layer here for safety
            nn.BatchNorm2d(3),
            
            # Conv 1: 5x5, stride 2
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            
            # Conv 2: 5x5, stride 2
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            
            # Conv 3: 5x5, stride 2
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            
            # Conv 4: 3x3
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            
            # Conv 5: 3x3
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(),
            
            nn.Flatten()
        )
        
        # Compute flatten size
        # 120 -> 58 -> 27 -> 12 -> 10 -> 8
        # 64 * 8 * 8 = 4096
        
        self.regressor = nn.Sequential(
            nn.Linear(4096, 100),
            nn.ELU(),
            nn.Dropout(0.3), # Standard dropout
            
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.3),
            
            nn.Linear(50, 10),
            nn.ELU(),
            
            nn.Linear(10, 1) # w only
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
