# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn


class D(nn.Module):
    def __init__(self , channels , features_d):
        super(D, self).__init__()
        
        self.blk_1 = nn.Sequential(
            nn.Conv2d(channels ,features_d ,4 , 2 , 1 )
            ,nn.BatchNorm2d(features_d)
            ,nn.LeakyReLU(.2)
            )
        
        self.blk_2 = nn.Sequential(
            nn.Conv2d(features_d , features_d*2 , 4 , 2 , 1 )
            ,nn.BatchNorm2d( features_d*2 )
            ,nn.LeakyReLU(.25)
            )
        
        self.blk_3 = nn.Sequential(
            nn.Conv2d(features_d*2 , features_d*3 , 4, 2,1 )
            ,nn.BatchNorm2d(features_d*3)
            ,nn.LeakyReLU(.25)
            )
        
        self.blk_4 = nn.Sequential(
            nn.Conv2d(features_d*3 , features_d*4 , 4 ,2 ,1 )
            ,nn.BatchNorm2d( features_d*4 )
            ,nn.LeakyReLU(.25)
            )
        
        self.blk_5 = nn.Sequential(
            ## Batch_size x 1 x 1 x1 
            nn.Conv2d( features_d*4 , 1 , 4 , 2 ,0)
            ,nn.Sigmoid()
            )
        
    def forward(self,x):
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        x = self.blk_4(x)
        x = self.blk_5(x)
        
        return x
    

class G(nn.Module):
    def __init__(self , latent_noise ,channels , features_g):
        super(G, self).__init__()
        
        self.blk_1 = nn.Sequential(
            nn.ConvTranspose2d( latent_noise , features_g*4  ,4 , 2 ,0 )
            ,nn.BatchNorm2d( features_g*4 )
            ,nn.ReLU(.25)
            )
        
        
        self.blk_2 = nn.Sequential(
            nn.ConvTranspose2d( features_g*4 , features_g*3 , 4 , 2 , 1)
            ,nn.BatchNorm2d(features_g*3)
            ,nn.ReLU(.25)
            )
        
        self.blk_3 = nn.Sequential(
            nn.ConvTranspose2d( features_g*3 , features_g*2 , 4 , 2, 1)
            ,nn.BatchNorm2d(features_g*2)
            ,nn.ReLU(.25)
            )
        
        self.blk_4 = nn.Sequential(
            nn.ConvTranspose2d( features_g*2 , features_g*1 ,4 , 2 , 1 )
            ,nn.BatchNorm2d(features_g*1)
            ,nn.ReLU(.25)
            )
        
        self.blk_5 = nn.Sequential(
            nn.ConvTranspose2d(features_g*1 , channels , 4 , 2 , 1 )
            # batch_size x 1 x 64 x 64 
            ,nn.Tanh()
            )
        
    def forward(self,x):
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        x = self.blk_4(x)
        x = self.blk_5(x)
        
        return x
        
































        
            

