# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 01:00:40 2020

@author: epocxlabs
"""

import torch
import torchvision.utils as vutils
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import   DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Models import D,G
### https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## Hyperparameter

lr = .0002
batch_size = 32
latent_noise = 100
channels = 1
features_d = 32
features_g = 32
num_epoch = 10

my_transforms = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=my_transforms, download=True
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## building the models

model_d = D(channels , features_d).to(device)
model_g = G(latent_noise ,channels , features_g ).to(device)

## optimizer

optimizerD = optim.Adam(model_d.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(model_g.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

real_label = 1
real_fake  = 0

fixed_noise = torch.rand(batch_size , latent_noise , 1 , 1 ).to(device)
img_list = []

print('Start Trainning -----')
'''
for epoch  in range(num_epoch):
    

        if idx % 50 == 0:
                print(
                f"Epoch [{epoch}/{num_epoch}] Batch {idx}/{len(dataloader)} \
                  Loss D: {total_loss:.4f}, loss G: {lossG:.4f} D(x): {d_mean:.4f}"
                                                                             )
                with torch.no_grad():
                    fake = model_g(fixed_noise)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
'''                   
    
counter = 0   
for epoch  in range(num_epoch):
    
    for idx,data in enumerate(dataloader):
         feat , label = data
         #### Train Discriminator with real label
         #### Update D network: maximize log(D(x)) + log(1 - D(G(z)))
         if True:
             feat = feat.to(device)
             label = label.to(device)
         y_real =torch.ones(batch_size,device=device)*.95 ## Label Smoothing
         
         model_d.zero_grad()
         real_output = model_d(feat).squeeze()
         real_loss = criterion(real_output,y_real)
         d_mean = real_output.mean().item()
         
         #### Train Discriminator with Fake label    
         noise = torch.rand(batch_size , latent_noise , 1 , 1 ).to(device)
         y_fake =torch.zeros(batch_size,device=device) * .05 
         fake_output = model_d(model_g(noise)).squeeze()
         fake_loss = criterion( fake_output , y_fake )
         g_mean = fake_output.mean().item()
         
         total_loss = real_loss + fake_loss
         total_loss.backward()
         optimizerD.step()
         
         
         #### Train Generator
         #### Update G network: maximize log(D(G(z)))
         model_g.zero_grad()
         label = torch.ones(batch_size,device=device)
         output = model_d(model_g(noise)).squeeze()
         lossG = criterion( output , label )
         lossG.backward()
         optimizerG.step()
         
         if idx %100 == 0:
             
             
             print(f"Epoch [{epoch}/{num_epoch}] Batch {idx}/{len(dataloader)} \
                  Loss D: {total_loss:.4f}, loss G: {lossG:.4f} ")
                  
             with torch.no_grad():
                 fake = model_g(fixed_noise)
                 
             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
         counter +=1
         
         
         
## Animation starts here
import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i.detach().cpu(),(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
                
             
             
         
     
    
    




































