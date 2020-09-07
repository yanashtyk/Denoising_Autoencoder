# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:42:46 2020

@author: Administrator
"""

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
LEN_BL=48
class Codes(Dataset): #create dataset class
    
    def __init__(self, org_dir, modf_dir ):
        
        self.org_dir=org_dir #in what directory are the originals
        self.modf_dir=modf_dir #in what directory are the modified
        self.transform=transforms.ToTensor() #what transform will be used
        
        self.org_names=os.listdir(org_dir) #List of all files in originals folder
        self.modf_names=os.listdir(modf_dir) #List of all files in modified folder
        self.org_names=[os.path.join(org_dir, org_name) for org_name in self.org_names] #join folder and file name for originals
        self.modf_names=[os.path.join(modf_dir, modf_name) for modf_name in self.modf_names] #join folder and file name for modified
        
        self.block_len=8*6 #size of blocks
        
    def bl_idx(self, index):
         #find index of image and block
        num_block=384/self.block_len 
        idx_cod=int(index//(num_block**2))
        index-=idx_cod*(num_block**2) 
        return idx_cod, index
        
        
    def get_x(self, idx_cod, index):
        
         #prepare the modified
        modf_name=self.modf_names[idx_cod]
        modf=Image.open(modf_name)
        modf=self.transform(modf)
        
         #extract the block
        x = modf.unfold(1, self.block_len, self.block_len).unfold(2, self.block_len, self.block_len)
        x = x.reshape(1, -1, LEN_BL, LEN_BL)
        return x[:, int(index),  :, :]
    
    
    def get_y(self, idx_cod, index):
        
         #prepare the original
        org_name=self.org_names[idx_cod]
        org=Image.open(org_name)
        org=self.transform(org)
        
        #extract the block
        
        y=org.unfold(1, self.block_len, self.block_len).unfold(2, self.block_len, self.block_len)
        y=y.reshape(1, -1, LEN_BL, LEN_BL)
    
        return y[:, int(index),:, :]
        
        
        
        
    def __getitem__(self, index):
            
        idx_cod, index=self.bl_idx(index)
       
        return self.get_x( idx_cod, index), self.get_y( idx_cod, index)
        

  
    def __len__(self):
        
        num_bl=int(384/self.block_len)
        
        return len(self.org_names)*num_bl**2
            
            
        