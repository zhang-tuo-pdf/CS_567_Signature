# Siamese Network model Implementation with Contrastive Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3)
        )

        # Assuming input size = 105x105x1, subject to change
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    '''
    Loss = 1/2[ (Y)*EuDist^2 + (1-Y)*max(0, margin-EuDist)^2 ]
        For Y=0 (dissimilar pair), pair with EuDist > margin will not contribute to loss
    '''

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        EuDist = F.pairwise_distance(x0, x1)
        loss = torch.mean(
            (1 - y) * torch.pow(EuDist, 2) + y * torch.pow(torch.clamp(self.margin - EuDist, min=0.0), 2)
        )

        return loss