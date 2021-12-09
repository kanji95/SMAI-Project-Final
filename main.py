import wandb

import argparse

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Baseline, N3Net
from dataset import *

from train import train_epoch
from evaluate import validate
from utils import RotationTransform


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Neural Nearest Neighbors Networks", add_help=False
    )

    # HYPER Parameters
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

    parser.add_argument("--num_crops", default=512, type=int)
    parser.add_argument("--crop_size", default=80, type=int)
    parser.add_argument("--sigma", default=25, type=int)

    parser.add_argument("--model_dir", type=str, default="./saved_model")

    parser.add_argument(
        "--data_root", type=str, default="/ssd_scratch/cvit/kanishk/n3_dataset"
    )

    ## Model Parameters
    parser.add_argument(
        "--channel_dim", type=int, default=4, help="channels in input image"
    )
    parser.add_argument(
        "--dncnn_out_feat", type=int, default=8, help="dncnn output features"
    )
    parser.add_argument(
        "--dncnn_feat_dim", type=int, default=64, help="dncnn features dim"
    )
    parser.add_argument("--dncnn_blocks", type=int, default=3, help="dncnn num blocks")
    parser.add_argument("--dncnn_depth", type=int, default=6, help="dncnn depth")

    parser.add_argument("--patch_size", type=int, default=10, help="n3block patch_size")
    parser.add_argument("--stride", type=int, default=5, help="n3block stride")
    parser.add_argument("--K", type=int, default=7, help="K nearest neigbors")
    parser.add_argument(
        "--match_window", type=int, default=15, help="size of matching window"
    )

    return parser


def main(args):

    experiment = wandb.init(project="N3Net", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"{device} being used with {n_gpu} GPUs!!")

    ## Dataset Augmentations
    transform = transforms.Compose(
        [
            transforms.RandomCrop(size=(args.crop_size, args.crop_size)),
            # transforms.RandomRotation(degrees=[0, 90, 180, 270]),
            RotationTransform(angles=[0, 90, 180, 270]),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p=0.3),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.RandomCrop(size=(args.crop_size, args.crop_size)),
            transforms.ToTensor(),
        ]
    )

    train_data = BSDDataset(args.data_root, split="train", transform=transform)
    # val_data = BSDDataset(args.data_root, split="val", transform=transform)
    val_data = Urban100(args.data_root, transform=val_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ## Model Definition
    """
    n3_net = N3Net(
        channel_dim=args.channel_dim,
        dncnn_out_channels=args.dncnn_out_feat,
        dncnn_feature_dim=args.dncnn_feat_dim,
        dncnn_blocks=args.dncnn_blocks,
        dncnn_depth=args.dncnn_depth,
        K_neighbors=args.K,
    )
    """
    # """
    n3_net = Baseline(
        channel_dim=args.channel_dim,
        dncnn_out_channels=args.dncnn_out_feat,
        dncnn_feature_dim=args.dncnn_feat_dim,
        dncnn_blocks=args.dncnn_blocks,
        dncnn_depth=args.dncnn_depth,
    )
    # """
    
    if n_gpu > 1:
        n3_net = nn.DataParallel(n3_net)
    n3_net.to(device)

    wandb.watch(n3_net, log="all")

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        n3_net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    for epoch in range(args.epochs):
        train_epoch(n3_net, train_loader, optimizer, epoch, args)
        scheduler.step()

        if epoch % 4 == 0:
            validate(n3_net, val_loader, epoch, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("N3 Networks", parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    main(args)
