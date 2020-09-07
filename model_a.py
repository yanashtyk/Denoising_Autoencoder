# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:40:36 2020

@author: Administrator
"""
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True), 
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,6,kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=3),
            nn.BatchNorm2d(1),
            nn.ReLU(True))
        
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    