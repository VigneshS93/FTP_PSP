import torch
import torch.nn as nn
import torch.nn.functional as F


class network(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1)

       
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        return h
