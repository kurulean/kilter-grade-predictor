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


class GateCNN(nn.Module):
    """Binary "is this climb physically plausible" classifier -- a much
    easier task than 14-way grading (spacing/hold-count, not subtle
    difficulty), so this is deliberately smaller than KilterCNN rather
    than a copy of it: fewer channels, two conv blocks instead of three.
    Runs ahead of KilterCNN in serve.py; a climb it rejects never reaches
    the grade model at all. See generate_gate_dataset.py for why this
    exists and how its training data was built.
    """

    NUM_CLASSES = 2

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.drop = nn.Dropout(0.25)

        # after 2 pools, 38x47 becomes 9x11, times 32 channels, plus 1 for angle
        self.fc1 = nn.Linear(32 * 9 * 11 + 1, 64)
        self.fc2 = nn.Linear(64, self.NUM_CLASSES)

    def forward(self, image, angle):
        x = self.drop(self.bn1(self.pool(self.relu(self.conv1(image)))))
        x = self.drop(self.bn2(self.pool(self.relu(self.conv2(x)))))
        x = x.flatten(1)

        angle = angle.unsqueeze(1)
        x = torch.cat([x, angle], dim=1)

        x = self.relu(self.fc1(x))
        return self.fc2(x)
