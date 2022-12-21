import torch
from torch import nn

import ipdb

class SAE(torch.nn.Module):
    def __init__(self, input_size:int, dropout:float, latent_size: int, hidden_sizes: list):
        super(SAE, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        # put every size layer in a list
        self.list_sizes = [input_size] + hidden_sizes + [latent_size]

        # Appending layers int a nn.Sequential list by using for loop
        modules= []
        for hs in range(1,len(self.list_sizes)):
            modules.append(nn.Dropout(self.dropout))
            modules.append(nn.Linear(self.list_sizes[hs-1], self.list_sizes[hs]))
            if hs<len(self.list_sizes)-1:
                modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules)

        # Reverse the list and made the corresponding modifications
        modules= []
        for hs in range(len(self.list_sizes)-1,0, -1):
            modules.append(nn.Linear(self.list_sizes[hs], self.list_sizes[hs-1]))
            if hs>1:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout())
        self.decoder = nn.Sequential(*modules)    

        # Create the model
        self.model = nn.Sequential(self.encoder, 
                                nn.Dropout(self.dropout),
                                self.decoder)

    def encode(self,x):
        return self.encoder(x)

    def forward(self, x):
        return self.model(x)