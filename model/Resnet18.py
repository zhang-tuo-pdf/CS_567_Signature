# Siamese Network model Implementation with Contrastive Loss
# Encoder: ResNet18 with pretrained weights and modified fc layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Use ResNet18 pretrained weights except for newly-designed fc layers
        self.resnet18 = models.resnet18(weights="DEFAULT")

        original_first_layer = self.resnet18.conv1

        # Create a new conv layer with in_channels=1
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize the new conv layer's weights by averaging the pretrained weights across the color channels
        with torch.no_grad():
            self.resnet18.conv1.weight[:] = original_first_layer.weight.mean(dim=1, keepdim=True)

        for param in self.resnet18.parameters():
            param.requires_grad = False

        self.resnet18.fc = nn.Sequential(
            nn.Linear(self.resnet18.fc.in_features, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 2)
        )

    def forward_once(self, x):
        output = self.resnet18(x)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    '''
    Loss = 1/2[ (1-Y)*EuDist^2 + (Y)*max(0, margin-EuDist)^2 ]
        For Y=1 (dissimilar pair), pair with EuDist > margin will not contribute to loss
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
