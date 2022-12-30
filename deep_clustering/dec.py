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

    def target_distribution_p(self, q_):
        weight = (q_**2) / torch.sum(q_, 0)
        return (weight.t()/torch.sum(weight, 1)).t().float()

    def target_distribution_phi(self, phi_):
        #add noise to phi
        # ipdb.set_trace()
        noise = 1e-5
        phi_hat = (phi_+noise)**(1/self.beta)
        weight = phi_hat / torch.sum( phi_, 0).t()
        return (weight.t()/torch.sum(weight, 1)).t().float()

    def forward(self, x):
        # ipdb.set_trace()
        x_proj = self.autoencoder.encode(x)
        soft_assignment = self.clustering_layer(x_proj)

        # get vector multiplication of soft_assignement and soft_assignment.t()
        # to get the conditional probability of each group

        p_ = self.target_distribution_p(soft_assignment)
        matrix_p = torch.linalg.inv(torch.mm(p_.t(), p_))
        centroids = torch.mm(matrix_p, torch.mm(p_.t(), x_proj))
        
        # fairoid_projected = self.autoencoder.encode(self.fairoid_layer.fairoid_centers).detach()
        # cond_prob_group = self.fairoid_layer(centroids, fairoid_projected)
        cond_prob_group = self.fairoid_layer(centroids)

        return soft_assignment.float(), cond_prob_group.float()
