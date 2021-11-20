import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim, depth) -> None:
        super(DnCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        layers = []
        for _ in range(depth - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        feature_dim, feature_dim, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                )
            )
        self.layer2 = nn.Sequential(*layers)

        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# {'k': 7, 'patchsize': 10, 'stride': 5, 'temp_opt': {'external_temp': True, 'temp_bias': 0.1, 'distance_bn': True, 'avgpool': True}, 'embedcnn_opt': {'features': 64, 'depth': 3, 'kernel': 3, 'bn': True, 'nplanes_out': 8}}
# {'features': 64, 'depth': 6, 'kernel': 3, 'bn': True, 'residual': True}


class N3Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=8,
        feature_dim=64,
        patch_size=10,
        stride=5,
        K=7,
        match_window=15,
    ) -> None:
        super(N3Block, self).__init__()
        
        self.K = K

        # Embedding CNN
        self.embedding_network = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # Temprature CNN
        self.temrature_cnn = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, y):

        E = self.embedding_network(y)
        T = self.temrature_cnn(y)
        
        # TODO: Insert Logic for nearest neigbors calculation
        
        neighbors = torch.cat([E]*self.K, dim=1)
        Y = torch.cat([E, neighbors], dim=1)
        return Y


class N3Net(nn.Module):
    def __init__(
        self,
        channel_dim=3,
        dncnn_out_channels=8,
        dncnn_feature_dim=64,
        dncnn_blocks=3,
        dncnn_depth=6,
    ) -> None:
        super(N3Net, self).__init__()

        dncnn_in_channels = channel_dim
        layers = []
        for _ in range(dncnn_blocks - 1):
            layers.append(
                nn.Sequential(
                    DnCNN(
                        dncnn_in_channels,
                        dncnn_out_channels,
                        dncnn_feature_dim,
                        dncnn_depth,
                    ),
                    N3Block(dncnn_out_channels),
                )
            )
            dncnn_in_channels = dncnn_feature_dim
        self.n3network = nn.Sequential(*layers)
        
        self.reconstruction_dncnn = DnCNN(dncnn_in_channels, channel_dim, dncnn_feature_dim, dncnn_depth)

    def forward(self, x):
        x = self.n3network(x)
        x = self.reconstruction_dncnn(x)
        return x


if __name__ == "__main__":
    n3net = N3Net()
    img = torch.rand(2, 3, 256, 256)

    out = n3net(img)
    print(out.shape)
