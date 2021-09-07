import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from quant_gan import TCN, TemporalBlock, Generator, Discriminator
from torch.utils.data import DataLoader
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.manual_seed(42)

class StockDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = np.expand_dims(self.data[index:index+self.window], -1)
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.data) - self.window

def train_quantgan(data_log, clip_value = 0.01, lr = 0.0002, num_epochs = 50, nz = 3, batch_size = 30, seq_len = 127,
                   celery_task = None):

    netG = Generator(nz, 1).to(device)
    netD = Discriminator(1, 1).to(device)
    optD = optim.RMSprop(netD.parameters(), lr=lr)
    optG = optim.RMSprop(netG.parameters(), lr=lr)
    
    data_mean = np.mean(data_log)
    data_norm = data_log - data_mean
    params = igmm(data_norm)
    data_processed = W_delta((data_norm - params[0]) / params[1], params[2])
    data_max = np.max(np.abs(data_processed))
    data_processed /= data_max
    
    dataset = StockDataset(data_processed, 127)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            noise = torch.randn(batch_size, seq_len, nz, device=device)
            fake = netG(noise).detach()

            lossD = -torch.mean(netD(real)) + torch.mean(netD(fake))
            lossD.backward()
            optD.step()

            for p in netD.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if i % 5 == 0:
                netG.zero_grad()
                lossG = -torch.mean(netD(netG(noise)))
                lossG.backward()
                optG.step()
            print(f'Batch {i} out of {len(dataloader)}')
        if celery_task:
            celery_task.update_state(state='PROGRESS', meta={'done': epoch + 1, 'total': num_epochs})
        
       
    netG.preprocess_params = {
        "data_mean": data_mean,
        "params": params,
        "data_max": data_max,
    }
    
    return netG.cpu()