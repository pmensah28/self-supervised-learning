import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# SimCLR model architecture: An encoder (ResNet18) and a projector.
class SimCLRModel(nn.Module):
    def __init__(self):
        super(SimCLRModel, self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        return z, F.normalize(p, dim=1)