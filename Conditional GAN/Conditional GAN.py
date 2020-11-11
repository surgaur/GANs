# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
from torchvision import datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim 
import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt


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
        
        self.label_embed = nn.Embedding(n_classes,n_classes)
        self.blk = nn.Sequential(
            nn.Linear(int(np.prod(img_shape))   + n_classes , 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear( 512 , 512 ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(.35),
            
            nn.Linear( 512 , 256 ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(.35),
            
            nn.Linear( 256 , 1 ),
            nn.Sigmoid()          
            )
        
    def forward(self , lab , img):
        
        label = self.label_embed(lab)
        x = torch.cat([ img.view(img.size(0),-1) , label ],1)
        x  = self.blk(x)
        
        return x

class Generator(nn.Module):
    def __init__(self , n_classes , img_shape):
        super(Generator, self).__init__()
        
        input_dim = 100
        
        self.label_embed = nn.Embedding(n_classes,n_classes)
        
        self.blk = nn.Sequential(
            nn.Linear( input_dim + n_classes , 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear( 256 , 512 ),
            nn.LeakyReLU(0.2),
            
            nn.Linear( 512 , int(np.prod(img_shape))),
            nn.Tanh()
            )
        
    def forward(self , label ,img ):
        
        lab =self.label_embed(label)
        x = torch.cat([ img , lab ], 1)
        x = self.blk(x)
        
        return x


## Model
model_d = Discriminator( n_classes , img_shape ).to(device)
model_g = Generator( n_classes , img_shape ).to(device)


## optimizer

optimizerD = optim.Adam(model_d.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(model_g.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()      



#noise = torch.randn(batch_size, latent_dim ).to(device)

#torch.random.normal(0, 1, (batch_size, latent_dim))

for epoch in range(n_epoch):
    
    for idx,( img , label ) in enumerate(dataloader):
        ### Running Discriminator 
        
        ### Discriminator with True labels
        img   = img.view(batch_size,-1).to(device)
        label = label.to(device)
        
        true_labels = torch.ones(batch_size).to(device)
        optimizerD.zero_grad()
        
        outputd  = model_d( label ,img )
        true_loss = criterion( outputd.squeeze() ,true_labels )
        
        ### Generator/Discriminator with fake labels
        
        noise = torch.randn(batch_size, latent_dim ).to(device)
        
        #fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        
        fake_labels = torch.zeros(batch_size).to(device)
        outputg = model_d( label, model_g( label , noise ))
        fake_loss = criterion( outputg.squeeze() , fake_labels )
        
        
        total_loss = true_loss + fake_loss
        
        total_loss.backward()
        optimizerD.step()
        
        ### Running Generator
        
        model_g.zero_grad()
        fake_true_labels = torch.ones(batch_size).to(device)
        output = model_d( label, model_g( label , noise ))
        lossG = criterion( output.squeeze() , fake_true_labels )
        
        lossG.backward()
        optimizerG.step()
        if idx %100 == 0:
            
            print(f"Epoch [{epoch}/{n_epoch}] Batch {idx}/{len(dataloader)} \
                  Loss D: {total_loss:.4f}, loss G: {lossG:.4f} ")
                  


### Model Eval                
with torch.no_grad():
    noise = torch.randn(batch_size,100).to(device)
    fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
    generated_data = model_g(fake_labels ,noise )
    generated_data = generated_data.view(batch_size,28,28)
        
    for x in range(len(generated_data)):
        
        plt.imshow(generated_data[x].detach().cpu().numpy(), interpolation='nearest',cmap='gray')
        plt.ylabel(fake_labels.detach().cpu().numpy()[x])
        plt.show()
        
print('Completed')