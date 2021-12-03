import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


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
        self.patch_size = patch_size
        self.stride = stride
        self.match_window = match_window

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

    def get_padding(self, x):
        xdim = x.shape[2:]
        pad_v = -(xdim[0] - self.patch_size) % self.stride
        pad_h = -(xdim[1] - self.patch_size) % self.stride

        pad_top = int(math.floor(pad_v / 2.0))
        pad_bottom = int(math.ceil(pad_v / 2.0))
        pad_left = int(math.floor(pad_h / 2.0))
        pad_right = int(math.ceil(pad_h / 2.0))

        return pad_top, pad_bottom, pad_left, pad_right

    def forward(self, x):

        _, _, H, W = x.shape

        E = self.embedding_network(x)
        T = self.temrature_cnn(x)

        # x2col, padding = self.im2col(x)
        E2col, padding = self.im2col(E)

        B, N, C = E2col.shape

        # TODO: Insert Logic for nearest neigbors calculation
        
        ## Distance Metric
        distance = torch.cdist(E2col, E2col)
        _, indices = torch.topk(distance, dim=-1, k=self.K + 1, largest=False)

        indices = indices.view(B, -1, 1).expand(B, N * (self.K + 1), C)

        E_neighbors = E2col.gather(dim=1, index=indices).view(B, N, self.K + 1, C)
        E_neighbors = E_neighbors[:, :, 1:]

        Y = torch.cat([E2col, E_neighbors.view(B, N, -1)], dim=2).transpose(1, 2)
        
        Y = F.fold(
            Y,
            kernel_size=[self.patch_size] * 2,
            stride=[self.stride] * 2,
            output_size=(H, W),
            padding=padding[0],
        )
        return Y

    def im2col(self, x):
        pad_top, pad_bottom, pad_left, pad_right = self.get_padding(x)
        x_pad = F.pad(x, pad=(pad_top, pad_bottom, pad_left, pad_right))

        B, C, H, W = x_pad.shape
        f_h = (H - self.patch_size) // self.stride + 1
        f_w = (W - self.patch_size) // self.stride + 1

        # B, C, K_h, K_w, F_h, F_w
        x2col = F.unfold(
            x_pad,
            kernel_size=[self.patch_size] * 2,
            stride=[self.stride] * 2,
            padding=[0, 0],
        )
        x2col = x2col.transpose(1, 2)
        # x2col = rearrange(
        #     x2col,
        #     "b (c kh kw) (fh fw) -> b c (kh kw) (fh fw)",
        #     c=C,
        #     kw=self.patch_size,
        #     kh=self.patch_size,
        #     fh=f_h,
        #     fw=f_w,
        # )

        return x2col, (pad_top, pad_bottom, pad_left, pad_right)


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

        self.reconstruction_dncnn = DnCNN(
            dncnn_in_channels, channel_dim, dncnn_feature_dim, dncnn_depth
        )

    def forward(self, x):
        x = self.n3network(x)
        x = self.reconstruction_dncnn(x)
        return x


if __name__ == "__main__":
    n3net = N3Net()
    img = torch.rand(2, 3, 256, 256)

    out = n3net(img)
    print(out.shape)
