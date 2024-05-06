# Siamese Network model Implementation + Contrastive Loss
# model encoder choices from ['cnn', 'resnet'], Default='cnn'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNetwork(nn.Module):
    def __init__(self, model='cnn'):
        self.model = model
        super().__init__()

        # Original SigNet CNN encoder
        if self.model == 'cnn':
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

                nn.Linear(128, 2))

        # Resnet18 encoder with pretrained weights
        elif self.model == 'resnet':
            # Use ResNet18 pretrained weights except for newly-designed fc layers
            #self.resnet18 = models.resnet18(weights="DEFAULT")
            self.resnet18 = models.resnet18()
            #original_first_layer = self.resnet18.conv1

            # Create a new conv layer with in_channels=1
            self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Initialize the new conv layer's weights by averaging the pretrained weights across the color channels
            # with torch.no_grad():
            #     self.resnet18.conv1.weight[:] = original_first_layer.weight.mean(dim=1, keepdim=True)

            # for param in self.resnet18.parameters():
            #     param.requires_grad = False

            self.resnet18.fc = nn.Sequential(
                nn.Linear(self.resnet18.fc.in_features, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256, 64),
                nn.ReLU(inplace=True),

                nn.Linear(64, 2)
            )

    def forward_once(self, x):
        if self.model == 'cnn':
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)

        elif self.model == 'resnet':
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
