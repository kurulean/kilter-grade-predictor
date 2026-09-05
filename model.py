import torch
import torch.nn as nn

NUM_CLASSES = 14


class KilterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # three conv blocks, each grows channels and shrinks the grid
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # keeps each layer's input on a stable scale, so training is
        # steadier and can support the extra capacity above
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # light dropout in the conv stack, heavier before the final layer
        self.conv_drop = nn.Dropout(0.25)
        self.fc_drop = nn.Dropout(0.5)

        # after 3 pools, 38x47 becomes 4x5, times 128 channels
        # plus 1 for the angle value
        self.fc1 = nn.Linear(128 * 4 * 5 + 1, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, image, angle):
        # conv, relu, pool, then batchnorm, matching the order used
        # in the reference thesis architecture
        x = self.conv_drop(self.bn1(self.pool(self.relu(self.conv1(image)))))
        x = self.conv_drop(self.bn2(self.pool(self.relu(self.conv2(x)))))
        x = self.conv_drop(self.bn3(self.pool(self.relu(self.conv3(x)))))
        x = x.flatten(1)

        # angle is one number per climb, add it as an extra feature
        angle = angle.unsqueeze(1)
        x = torch.cat([x, angle], dim=1)

        x = self.fc_drop(self.relu(self.fc1(x)))
        return self.fc2(x)
