import wandb

import torch
import torch.nn as nn

from utils import grad_check

from statistics import mean
from metrics import psnr, ssim

from torchvision.utils import save_image, make_grid


def train_epoch(network, loader, optimizer, epoch, log_interval=50):

    network.train()

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()

    total_loss = []
    total_ssim = []
    total_psnr = []

    for step, batch in enumerate(loader):
        x = batch.cuda()

        x_recon = network(x)

        loss = criterion(x_recon, x)

        optimizer.zero_grad()
        loss.backward()

        if step % 20 == 0:
            grad_check(network.named_parameters())

        optimizer.step()

        ssim_ = ssim(x_recon, x)
        psnr_ = psnr(x_recon, x)

        total_loss += [loss.item()]
        total_ssim += [ssim_.item()]
        total_psnr += [psnr_.item()]

        if step % log_interval == 0:
            wandb.log(
                {
                    "Orig Images": wandb.Image(make_grid(x.detach().cpu())),
                    "Recon Images": wandb.Image(make_grid(x_recon.detach().cpu())),
                }
            )
        #     print(f'Epoch: {epoch}, training_loss: {mean(total_loss)}')

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
