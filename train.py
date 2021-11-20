import torch
import torch.nn as nn

def train_epoch(network, loader, optimizer, epoch, log_interval=50):
    
    network.train()
    
    criterion = nn.MSELoss()
    
    total_loss = 0
    
    for step, batch in enumerate(loader):
        x = batch.cuda()
        
        x_recon = network(x)
        
        loss = criterion(x_recon, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % log_interval == 0:
            print(f'Epoch: {epoch}, training_loss: {total_loss/(step+1)}')