import torch
import torch.nn as nn
import torchvision


class DnCNN(nn.Module):
    def __init__(self, in_channels, depth) -> None:
        super(DnCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        layers = []
        for _ in range(2, depth - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )

        self.layer2 = nn.Sequential(*layers)
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        
    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x + identity

class N3Block(nn.Module):
    def __init__(self, in_channels, feature_dim=None) -> None:
        super(N3Block, self).__init__()
        
        if feature_dim == None:
            feature_dim = in_channels
        
        self.embedding_network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1),
        )
        
        self.temrature_cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=feature_dim, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        
        