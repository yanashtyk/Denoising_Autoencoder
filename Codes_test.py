# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:08:58 2020

@author: Administrator
"""
from Codes import Codes
from PIL import Image

class Codes_test(Codes): #create dataset class
    
    def __init__(self, org_dir, modf_dir):
        super().__init__(org_dir, modf_dir)

            
        
        
    def __getitem__(self, index):
            
        idx_cod, index=self.bl_idx(index)
       
        return self.get_x( idx_cod, index)
    
    def original(self, index):
        
        org_name=self.org_names[index]
        org=Image.open(org_name)
        org=self.transform(org)
        
        return org
        

            