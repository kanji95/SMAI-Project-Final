import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class DnCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_dim,
        depth,
        residual=False,
        last_block=False,
    ) -> None:
        super(DnCNN, self).__init__()

        self.residual = residual
        self.last_block = last_block
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        layers = []
        for _ in range(depth - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        feature_dim,
                        feature_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.layer2 = nn.Sequential(*layers)

        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if residual:
            self.residual_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
            )

    def forward(self, x):
        shortcut = torch.clone(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.residual:
            x = x + self.residual_block(shortcut)
        if self.last_block:
            x = torch.sigmoid(x)
        return x


class N3Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=8,
        feature_dim=64,
        patch_size=10,
        stride=5,
        K=13,
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
        # self.temperature_cnn = nn.Sequential(
        #     nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(feature_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(feature_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1),
        # )

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

        # B, C, H, W = x.shape

        ## Embedding Network (B, C, H, W)
        E = self.embedding_network(x)

        B, C, H, W = E.shape

        # E2col = rearrange(E, "b c h w -> b (h w) c")
        # B, N, C = E2col.shape

        # distance = -torch.cdist(E2col, E2col)
        # distance_softmax = F.softmax(distance, dim=-1)

        # _, indices = torch.topk(distance_softmax, dim=-1, k=self.K + 1)

        # indices = indices[:, :, 1:]
        # # indices = repeat(indices, 'b n k -> b n k c', c=C)

        # E_neighbors = torch.zeros_like(E2col)
        # E_neighbors = repeat(E_neighbors, "b n c -> b n k c", k=self.K)

        # for batch_idx in range(B):
        #     E_neighbors[batch_idx] = E2col[batch_idx, indices[batch_idx]]

        # E_neighbors = rearrange(E_neighbors, "b n k c -> b n (k c)")

        # Y = torch.cat([E2col, E_neighbors], dim=-1)

        # Y = rearrange(Y, "b (h w) c -> b c h w", h=H, w=W)
        # return Y
    
        Z = torch.zeros_like(E)
        Z = repeat(Z, 'b c h w -> b (repeat c) h w', repeat=self.K)
        
        ### Vectorize this code
        for i in range(H):
            for j in range(W):
                query = E[:, :, i, j] # B, C, 1, 1
                
                tl_x = max(0, i - self.match_window)
                tl_y = max(0, j - self.match_window)
                br_x = min(H, i + self.match_window)
                br_y = min(W, j + self.match_window)
                
                neighbor_patch = E[:, :, tl_x:br_x+1, tl_y:br_y+1] # B, C, Ph, Pw
                
                query = rearrange(query, 'b c -> b 1 c')
                neighbor_patch = rearrange(neighbor_patch, 'b c h w -> b (h w) c')
                
                distance = -torch.cdist(query, neighbor_patch) # b 1 hw
                distance_softmax = F.softmax(distance, dim=-1)
                
                _, indices = torch.topk(distance_softmax, dim=-1, k=self.K + 1) # b 1 k
                indices = repeat(indices.squeeze(1), 'b k -> b k c', c=C)
                neighbors = neighbor_patch.gather(dim=1, index=indices) # b k c
                neighbors = rearrange(neighbors[:, 1:], 'b k c -> b (k c)')
                
                Z[:, :, i, j] = neighbors
                
        Y = torch.cat([E, Z], dim=1)
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
        x2col = rearrange(
            x2col,
            "b (c kh kw) (fh fw) -> b (fh fw) (kh kw) c",
            c=C,
            kw=self.patch_size,
            kh=self.patch_size,
            fh=f_h,
            fw=f_w,
        )

        return x2col, (pad_top, pad_bottom, pad_left, pad_right)


class Baseline(nn.Module):
    def __init__(
        self,
        channel_dim=3,
        dncnn_out_channels=8,
        dncnn_feature_dim=64,
        dncnn_blocks=3,
        dncnn_depth=6,
    ) -> None:
        super(Baseline, self).__init__()

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
                        residual=True,
                    ),
                )
            )
            dncnn_in_channels = dncnn_out_channels

        self.n3network = nn.Sequential(*layers)

        self.reconstruction_cnn = DnCNN(
            dncnn_in_channels,
            channel_dim,
            dncnn_feature_dim,
            dncnn_depth,
            residual=True,
            last_block=True,
        )

    def forward(self, x):
        x = self.n3network(x)
        x = self.reconstruction_cnn(x)
        return x


class N3Net(nn.Module):
    def __init__(
        self,
        channel_dim=3,
        dncnn_out_channels=8,
        dncnn_feature_dim=64,
        dncnn_blocks=3,
        dncnn_depth=6,
        K_neighbors=7,
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
                        residual=True,
                    ),
                    N3Block(dncnn_out_channels, K=K_neighbors),
                )
            )
            dncnn_in_channels = dncnn_out_channels * (K_neighbors + 1)
        self.n3network = nn.Sequential(*layers)

        self.reconstruction_cnn = DnCNN(
            dncnn_in_channels,
            channel_dim,
            dncnn_feature_dim,
            dncnn_depth,
            residual=True,
            last_block=True,
        )

    def forward(self, x):
        x = self.n3network(x)
        x = self.reconstruction_cnn(x)
        return x


if __name__ == "__main__":
    n3net = N3Net()
    img = torch.rand(2, 3, 80, 80)

    out = n3net(img)
    print(out.shape)
        
