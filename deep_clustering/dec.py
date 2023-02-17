import torch
from torch import nn
from torch.nn import functional as F
import math

from deep_clustering.cluster import ClusterLayer, FairoidLayer
# from deep_clustering.sae import SAE
from typing import Optional, List

import ipdb

class SAE(nn.Module):
    def __init__(self, input_size:int, 
                dropout:float, 
                latent_size: int, 
                hidden_sizes: list):
        super(SAE, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        # put every size layer in a list
        self.list_sizes = hidden_sizes + [latent_size]

        # Appending layers int a nn.Sequential list by using for loop
        modules= []
        modules.append(nn.Linear(input_size, self.list_sizes[0]))
        for hs in range(len(self.list_sizes)-1):
            modules.append(nn.Dropout(self.dropout))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(self.list_sizes[hs], self.list_sizes[hs+1]))
        self.encoder = nn.Sequential(*modules)

        # Reverse the list and made the corresponding modifications
        modules= []
        for hs in range(len(self.list_sizes)-1,0, -1):
            modules.append(nn.Linear(self.list_sizes[hs], self.list_sizes[hs-1]))
            modules.append(nn.Dropout())
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.list_sizes[0], input_size))
        self.decoder = nn.Sequential(*modules)    

        self.init_weights()
    
    def init_weights(self):
        def func(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.00)

        self.encoder.apply(func)
        self.decoder.apply(func)


    def get_encoder(self,x):
        return self.encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class clustering(nn.Module):
    def __init__(self, 
                n_clusters:int, 
                input_shape:int, 
                alpha: float=1.0, 
                cluster_centers: Optional[torch.Tensor]=None)->None:
        super(clustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_clusters, self.input_shape, dtype=torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        '''
        Forward pass

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        # q = 1.0 / (1.0 + (torch.sum((inputs[:, None, :] - self.clustcenters[None, :, :]) ** 2, dim=2) / self.alpha))
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, axis=1) - self.cluster_centers), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
        return q
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

class DEC(nn.Module):
    def __init__(self, 
                n_clusters:int, 
                input_size:int,
                latent_size_sae:int, 
                hidden_sizes_sae:list, 
                alpha:float,
                beta:float,
                dropout:float,
                cluster_centers=None,
                fairoid_centers=None, 
                autoencoder=None,
                p_norm=2):
        super(DEC, self).__init__()
        self.beta = beta

        # self.clustering_layer = ClusterLayer(n_clusters, latent_size_sae, cluster_centers, alpha, p_norm)
        self.clustering_layer = clustering(n_clusters, latent_size_sae)
        # self.fairoid_layer = FairoidLayer(n_clusters, latent_size_sae, fairoid_centers, alpha, p_norm)
        
        self.__null_autoencoder = autoencoder==None
        if self.__null_autoencoder:
            self.autoencoder = SAE(input_size, 
                                    dropout, 
                                    latent_size_sae, 
                                    hidden_sizes_sae)
        else:
            self.autoencoder = autoencoder

        self.model = nn.Sequential(
            self.autoencoder.encoder,
            self.clustering_layer
        )

    # def target_distribution_phi(self, phi):
    #     #add noise to phi
    #     # ipdb.set_trace()
    #     noise = 1e-5
    #     phi_hat = (phi+noise)**(1/self.beta)
    #     weight = phi_hat / torch.sum( phi, 0).t()
    #     return (weight.t()/torch.sum(weight, 1)).t().float()
    
    # def project(self, x):
    #     x_proj = self.autoencoder.encode(x) 
    #     return x_proj

    # def soft_assignment(self, x_proj):
    #     q = self.clustering_layer(x_proj)
    #     return q
    
    # def cond_prob(self, q, x_proj):
    #     p = self.target_distribution_p(q)
    #     matrix_p = torch.linalg.inv(torch.mm(p.t(), p))
    #     centroids = torch.mm(matrix_p, torch.mm(p.t(), x_proj))

    #     cond_prob_group = self.fairoid_layer(centroids)

    #     return cond_prob_group

    def forward(self, x):
        X = self.model(x)
        return X

