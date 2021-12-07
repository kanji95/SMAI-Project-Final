import wandb

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from statistics import mean
from metrics import psnr, ssim

from utils import data_preprocessing, criterion


@torch.no_grad()
def validate(network, loader, epoch, args, log_interval=50):

    network.eval()

    # criterion = nn.MSELoss()
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

        loss = criterion(x_recon, target)

        total_loss += [loss.item()]

        ssim_ = ssim(x_recon, target)
        psnr_ = psnr(x_recon, target, maxval=1).mean()

        total_ssim += [ssim_.item()]
        total_psnr += [psnr_.item()]

        if step % log_interval == 0:
            print(f"Epoch: {epoch}, validation_loss: {mean(total_loss)}")
            wandb.log(
                {
                    "Orig Images": wandb.Image(make_grid(target.detach().cpu())),
                    "Noisy Images": wandb.Image(make_grid(x.detach().cpu())),
                    "Recon Images": wandb.Image(make_grid(x_recon.detach().cpu())),
                }
            )

    wandb.log(
        {
            "Val Loss": mean(total_loss),
            "Val PSNR": mean(total_psnr),
            "Val SSIM": mean(total_ssim),
        }
    )
    print(
        f"Validation Epoch: {epoch}, PSNR: {mean(total_psnr)}, SSIM: {mean(total_ssim)}"
    )
