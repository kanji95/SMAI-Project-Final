import wandb

import torch
import torch.nn as nn

from utils import grad_check, data_preprocessing, criterion

from statistics import mean
from metrics import psnr, ssim

from torchvision.utils import save_image, make_grid


def train_epoch(network, loader, optimizer, epoch, args, log_interval=50):

    network.train()

    # criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')
    # criterion = nn.SmoothL1Loss(reduction='sum')

    total_loss = []
    total_ssim = []
    total_psnr = []

    sigma = args.sigma
    
    for step, batch in enumerate(loader):
        
        x, target = data_preprocessing(batch, sigma)
        
        x = x.cuda()
        target = target.cuda()

        x_recon = network(x)

        # print(x.shape, x_recon.shape)
        loss = criterion(x_recon, x)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(network.parameters(), 100)

        if step % 20 == 0:
            grad_check(network.named_parameters())

        optimizer.step()

        ssim_ = ssim(x_recon, target)
        psnr_ = psnr(x_recon, target, maxval=1).mean()

        total_loss += [loss.item()]
        total_ssim += [ssim_.item()]
        total_psnr += [psnr_.item()]

    wandb.log(
        {
            "Train Loss": mean(total_loss),
            "Train PSNR": mean(total_psnr),
            "Train SSIM": mean(total_ssim),
        }
    )
    print(
        f"Training Epoch: {epoch}, Loss: {mean(total_loss)}, PSNR: {mean(total_psnr)}, SSIM: {mean(total_ssim)}"
    )
