import torch
from torch import nn

import ipdb

class ClusterLayer(nn.Module):
    def __init__(self, 
                 n_clusters:int, 
                 latent_size:int, 
                 cluster_centers:torch.Tensor, 
                 alpha:float, 
                 p_norm=2):
        super(ClusterLayer, self).__init__()
        self.n_clusters = n_clusters
        self.latent_size = latent_size
        self.p_norm = p_norm
        
        if cluster_centers==None:
            initial_cluster_centers = torch.zeros(self.n_clusters, self.latent_size, dtype=torch.float64).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        self.cluster_centers = nn.Parameter(initial_cluster_centers)

        self.alpha = alpha

    def forward(self, x):
        #compute soft assignment q
        distance_centers = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, dim=2)
        numerator_q = (1.0 + (distance_centers / self.alpha))**(-float(self.alpha+1)/2)
        denominator_q = torch.sum(numerator_q, dim=1)
        q_ij = numerator_q.t() / denominator_q
        
        return q_ij.t().float()

class FairoidLayer(nn.Module):
    def __init__(self, n_clusters:int, latent_size:int, fairoid_centers:torch.Tensor, alpha:float, p_norm=2):
        super(FairoidLayer, self).__init__()
        self.n_clusters = n_clusters
        self.latent_size = latent_size
        self.p_norm = p_norm
        
        if fairoid_centers==None:
            initial_fairoid_centers = torch.zeros(self.n_clusters, self.latent_size, dtype=torch.float64).cuda()
            nn.init.xavier_uniform_(initial_fairoid_centers)
        else:
            initial_fairoid_centers = fairoid_centers

        # self.fairoid_centers = initial_fairoid_centers
        self.fairoid_centers = nn.Parameter(initial_fairoid_centers)

        self.alpha = alpha

    def forward(self, cluster_centers):
    # def forward(self, cluster_centers, fairoid_centers):
        # ipdb.set_trace()
        # distance_centers = torch.sum((cluster_centers.unsqueeze(1) - fairoid_centers)**2, dim=2)
        distance_centers = torch.sum((cluster_centers.unsqueeze(1) - self.fairoid_centers)**2, dim=2)
        numerator_phi = (1.0 + (distance_centers / self.alpha))**(-float(self.alpha+1)/2)
        denominator_phi = torch.sum(numerator_phi, dim=1)
        phi_jt = numerator_phi.t() / denominator_phi
        
        return phi_jt.t().float()