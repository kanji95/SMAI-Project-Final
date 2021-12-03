import wandb

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from statistics import mean
from metrics import psnr, ssim


@torch.no_grad()
def validate(network, loader, epoch, log_interval=50):

    network.eval()

    criterion = nn.MSELoss()

    total_loss = []
    total_ssim = []
    total_psnr = []

    for step, batch in enumerate(loader):
        x = batch.cuda()

        x_recon = network(x)

        loss = criterion(x_recon, x)

        total_loss += [loss.item()]

        ssim_ = ssim(x_recon, x)
        psnr_ = psnr(x_recon, x)

        total_ssim += [ssim_.item()]
        total_psnr += [psnr_.item()]

        if step % log_interval == 0:
            print(f"Epoch: {epoch}, validation_loss: {mean(total_loss)}")
            if epoch % 10 == 0:
                wandb.log(
                    {
                        "Orig Images": wandb.Image(make_grid(x.detach().cpu())),
                        "Recon Images": wandb.Image(make_grid(x_recon.detach().cpu())),
                    }
                )
                # save_image(x.detach().cpu(), f"./results/orig_epoch_{epoch}_images.png")
                # save_image(x_recon.detach().cpu(), f"./results/recon_epoch_{epoch}_images.png")
    wandb.log(
        {
            "Val Loss": mean(total_loss),
            "Val PSNR": {mean(total_psnr)},
            "Val SSIM": {mean(total_ssim)},
        }
    )
    print(
        f"Validation Epoch: {epoch}, PSNR: {mean(total_psnr)}, SSIM: {mean(total_ssim)}"
    )
