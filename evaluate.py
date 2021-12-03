import torch
import torch.nn as nn
from torchvision.utils import save_image

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
        
        total_ssim += [ssim_]
        total_psnr += [psnr_]
        
        if step % log_interval == 0:
            print(f'Epoch: {epoch}, validation_loss: {mean(total_loss)}')
            if epoch % 10 == 0:
                save_image(x.detach().cpu(), f"orig_epoch_{epoch}_images.png")
                save_image(x_recon.detach().cpu(), f"recon_epoch_{epoch}_images.png")
    print(f'Validation Epoch: {epoch}, PSNR: {mean(total_psnr)}, SSIM: {mean(total_ssim)}')