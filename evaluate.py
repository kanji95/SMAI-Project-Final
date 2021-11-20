import torch
import torch.nn as nn

@torch.no_grad()
def validate(network, loader, epoch, log_interval=50):
    
    network.eval()
    
    criterion = nn.MSELoss()
    
    total_loss = 0
    
    for step, batch in enumerate(loader):
        x = batch.cuda()
        
        x_recon = network(x)
        
        loss = criterion(x_recon, x)
    
        total_loss += loss.item()
        
        if step % log_interval == 0:
            print(f'Epoch: {epoch}, validation_loss: {total_loss/(step+1)}')
    