# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
from torchvision import datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import numpy as np



batch_size = 32
n_epoch = 25
latent_dim = 100
n_classes = 10
img_shape = 28*28
lr = .0001

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

feat , label = next(iter(dataloader))

class Discriminator(nn.Module):
    def __init__(self , n_classes , img_shape):
        super(Discriminator, self).__init__()
        
        self.label_embed = nn.Embedding( n_classes , np.prod(img_shape) )
        
        self.blk_1 = nn.Sequential(
            nn.Conv2d( 2 , 32 , 4 , 2 , 1)
            ,nn.BatchNorm2d(32)
            ,nn.LeakyReLU(.2)
            )
            
        self.blk_2 = nn.Sequential(    
            nn.Conv2d( 32 , 32*2 , 4 , 2 , 1)
            ,nn.BatchNorm2d( 32*2 )
            ,nn.LeakyReLU( .2 )
            )
        self.blk_3 = nn.Sequential(
            nn.Conv2d( 32*2 , 32*3 , 4 , 2 , 1)
            ,nn.BatchNorm2d( 32*3 )
            ,nn.LeakyReLU( .2 )
            )
            
        self.blk_4 = nn.Sequential(
            nn.Conv2d( 32*3, 1 , 4 , 2 , 1)
            ,nn.Sigmoid()
            )
        
      
        
    def forward(self , lab , img):
        
        label = self.label_embed( lab )
        label = label.view(-1 , 1 , 28 ,28 )
        x = torch.cat([ label , img ] , 1 )
        x = self.blk_1(x)
        x = self.blk_2(x) 
        x = self.blk_3(x)
        x = self.blk_4(x)        
        
        return x
    
label_embed = nn.Embedding( n_classes , np.prod(img_shape) )
labd = label_embed(label)
labd = labd.view(-1 , 1 , 28 ,28 )

torch.cat([ labd,feat ],1)

class Generator(nn.Module):
    def __init__(self , n_classes , embed_size , latent_noise ):
        super(Generator, self).__init__()
        
           
        self.label_embed = nn.Embedding(n_classes,embed_size)
        self.blk_1 =  nn.Sequential(nn.ConvTranspose2d( latent_noise + embed_size  , 32 *4  ,4 , 2 ,0 )
                                    ,nn.BatchNorm2d( 32*4 )
                                    ,nn.ReLU(.25)
                                    )
        
        self.blk_2 = nn.Sequential(
            nn.ConvTranspose2d( 32*4 , 32*3 , 4 , 2 , 1)
            ,nn.BatchNorm2d(32*3)
            ,nn.ReLU(.25)
            )
        
        self.blk_3 = nn.Sequential(
            nn.ConvTranspose2d( 32*3 , 32*2 , 4 , 2, 1)
            ,nn.BatchNorm2d(32*2)
            ,nn.ReLU(.25)
            )
        
        self.blk_4 = nn.Sequential(
            nn.ConvTranspose2d( 32*2 , 1 ,4 , 2, 1 )
            ,nn.Tanh()
            )
        self.adaptive = nn.AdaptiveAvgPool2d((28,28))
  
    def forward(self , lab , img):
        
        embed = self.label_embed(lab).unsqueeze(2).unsqueeze(3) ## bs X Noise x 1 x 1
        x = torch.cat([embed , img ],1)
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        x = self.blk_4(x)
        x = self.adaptive(x)
        
        return x
        

n_classes = 10
embed_size = 28*28
latent_noise = 100
## Model
model_d = Discriminator( n_classes , img_shape ).to(device)
model_g = Generator(n_classes , embed_size , latent_noise).to(device)

print('Trainning is same as Conditional GAN')