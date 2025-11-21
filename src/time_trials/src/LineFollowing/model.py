import torch
import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.features = nn.Sequential(
            # Normalization is usually done in dataset or preprocessing
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
        
        # Calculate input size for fully connected layer
        # 200x66 -> 98x31 -> 47x14 -> 22x5 -> 20x3 -> 18x1
        # Wait, let's calculate carefully or use a dummy input.
        # 200 (width) / 2 = 99 (kernel 5) -> 98?
        # (200 - 5) / 2 + 1 = 98.5 -> 98
        # (98 - 5) / 2 + 1 = 47.5 -> 47
        # (47 - 5) / 2 + 1 = 22
        # (22 - 3) / 1 + 1 = 20
        # (20 - 3) / 1 + 1 = 18
        # Height:
        # (66 - 5) / 2 + 1 = 31.5 -> 31
        # (31 - 5) / 2 + 1 = 14
        # (14 - 5) / 2 + 1 = 5.5 -> 5
        # (5 - 3) / 1 + 1 = 3
        # (3 - 3) / 1 + 1 = 1
        # So 64 * 18 * 1 = 1152
        
        self.classifier = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2) # v, w
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
