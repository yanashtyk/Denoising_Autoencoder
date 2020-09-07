# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:46:47 2020

@author: Administrator
"""
import torch 
import argparse
from model_a import autoencoder
from Codes import Codes
import torch.nn as nn
parser=argparse.ArgumentParser(description='Train the model')
parser.add_argument('weight_decay', type=float, help='Regularization')
parser.add_argument('lr', type=float, help='learning rate')
parser.add_argument('org_dir', type=str, help='directory of originals')
parser.add_argument('modf_dir', type=str,help='directory of modified')
args=parser.parse_args()



org_dir=args.org_dir #in what directory are the originals
modf_dir=args.modf_dir #in what directory are the modified

code=Codes(org_dir, modf_dir)

train_set, test_set=torch.utils.data.random_split(code, [int(len(code)*0.7), int(len(code)*0.3)])
    
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
num_epochs=5

for epoch in range(num_epochs):
    for data in train_loader:
        model.train()
        img, lab = data
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, lab)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))



loss=0
num=0
for data in test_loader:
    model.eval()
    img, lab=data
    
    output=model(img)
    
    loss+=criterion(output, lab)
    num+=1
    
loss/=float(num)    
print('test_loss:{:.4f}'.format(loss))


torch.save(model.state_dict(), './conv_autoencoder005.pth')