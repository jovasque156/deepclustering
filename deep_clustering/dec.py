import torch
from torch import nn
from torch.nn import functional as F

from deep_clustering.cluster import ClusterLayer, FairoidLayer
from deep_clustering.sae import SAE

import ipdb

class DEC(nn.Module):
    def __init__(self, 
                n_clusters:int, 
                latent_size_sae:int, 
                hidden_sizes_sae:list, 
                cluster_centers,
                fairoid_centers, 
                alpha:float,
                beta:float,
                dropout:float,
                autoencoder=None,
                p_norm=2):
        super(DEC, self).__init__()
        self.beta = beta

        self.clustering_layer = ClusterLayer(n_clusters, latent_size_sae, cluster_centers, alpha, p_norm)
        self.fairoid_layer = FairoidLayer(n_clusters, latent_size_sae, fairoid_centers, alpha, p_norm)
        
        self.__null_autoencoder = autoencoder==None
        if self.__null_autoencoder:
            self.autoencoder = SAE(latent_size_sae, dropout, latent_size_sae, hidden_sizes_sae)
        else:
            self.autoencoder = autoencoder

    def target_distribution_p(self, q):
        weight = (q**2) / torch.sum(q, 0)
        return (weight.t()/torch.sum(weight, 1)).t().float()

    def target_distribution_phi(self, phi):
        #add noise to phi
        # ipdb.set_trace()
        noise = 1e-5
        phi_hat = (phi+noise)**(1/self.beta)
        weight = phi_hat / torch.sum( phi, 0).t()
        return (weight.t()/torch.sum(weight, 1)).t().float()
    
    def project(self, x):
        x_proj = self.autoencoder.encode(x) 
        return x_proj

    def soft_assignment(self, x_proj):
        q = self.clustering_layer(x_proj)
        return q
    
    def cond_prob(self, q, x_proj):
        p = self.target_distribution_p(q)
        matrix_p = torch.linalg.inv(torch.mm(p.t(), p))
        centroids = torch.mm(matrix_p, torch.mm(p.t(), x_proj))

        cond_prob_group = self.fairoid_layer(centroids)

        return cond_prob_group

