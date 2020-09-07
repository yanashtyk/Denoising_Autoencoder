# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:14:01 2020

@author: Administrator
"""

import torch
import numpy as np
import argparse
from Codes_test import Codes_test
from model_a import autoencoder
from torchvision.utils import save_image

parser=argparse.ArgumentParser(description='Test the model')
parser.add_argument('org_dir', type=str, help='directory of originals')
parser.add_argument('modf_dir', type=str,help='directory of modified')
args=parser.parse_args()

org_dir=args.org_dir #in what directory are the originals
modf_dir=args.modf_dir #in what directory are the modified

def to_image384(tlist, num_bl):
    t=[]
    for i in range(num_bl):
        row=torch.cat([tlist[j, 0, :, :] for j in range (i*num_bl, (i+1)*num_bl)], 1)
        t.append(row)
        
    res=torch.cat(t)

    return res.unsqueeze(0)
    



def to_binary(img):
    num_pixl=6
    col=int(img.shape[1]/num_pixl)
    row=int(img.shape[2]/num_pixl)
    binr = np.zeros(img.shape)
    for i in range( row):
        for j in range (col):
            s=0
            
            s=sum(img[0, k, m] for k in range (i*num_pixl, (i+1)*num_pixl) for m in range(j*num_pixl, (j+1)*num_pixl))
            
            if s>num_pixl*num_pixl/2.0:
                binr[0,i*num_pixl: (i+1)*num_pixl , j*num_pixl: (j+1)*num_pixl]=1
    return torch.tensor(binr, dtype=torch.float)




def Ham_dst(cd1, cd2,sym_siz ):
    
    num_sym=int(cd1.shape[1]/sym_siz)
    
    dist=sum((int(cd1[0, i*sym_siz, j*sym_siz])^int(cd2[0, i*sym_siz, j*sym_siz])) for i in range (num_sym) for j in range (num_sym))
    
    return dist
model = autoencoder()
model.load_state_dict(torch.load('./conv_autoencoder005.pth'))
model.eval()


test_set=Codes_test(org_dir, modf_dir)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
out_list=[]
loss=0
count=0

for data in test_loader:
    img=data
    
    output=model(img)
    
    res=to_image384(output, 8)
    res=to_binary(res)
    
    org=test_set.original(count)
    num_sym=(384/6)**2
    loss+=Ham_dst(res, org, 6)/num_sym
    count +=1
    out_list.append(res)
print('loss:{:.4f}'.format(loss/len(out_list)))



