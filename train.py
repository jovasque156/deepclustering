#Basics
import os
import numpy as np
import argparse
from utils import (
    save_checkpoint, 
    visualize, 
    load_data, 
    balance,
    similarity_matrix
)

import warnings
warnings.filterwarnings('ignore')

#Torch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import MSELoss, KLDivLoss
import torch.nn.functional as F

# resulting_clusterings
from deep_clustering.sae import SAE
from deep_clustering.dec import DEC
from sklearn.cluster import KMeans

# Debugging
import ipdb

DATA = ['census_income', 'compas', 'dutch_census', 'german_data']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ContrastiveLoss(points, sensitive_attribute, t):
    '''
    Compute the contrastive loss

    Input:
        points: points to compute the loss on
        sensitive_attribute: sensitive attribute of the points
        m: margin

    Output:
        loss_c: contrastive loss
    '''
    # Compute the similarity matrix
    
    # ipdb.set_trace()

    # sensitive_attribute = sensitive_attribute.contiguous().view(-1, 1)
    # mask = torch.eq(sensitive_attribute, sensitive_attribute.T).float().to(DEVICE)
    
    loss = torch.zeros(points.shape[0]).to(DEVICE)

    for i, (p, s) in enumerate(zip(points, sensitive_attribute)):
        positives = points[sensitive_attribute!=s]
        anchor_dot_contrast = p*positives/t
        logit_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logit_max.detach()
        
        # phi_p = torch.exp(torch.sum(logits, dim=-1))
        # phi_p /= phi_p.sum()
        # phi_p = torch.log(phi_p).sum()
        
        exp_logits_fair = torch.exp(logits)
        exp_logits_sum = exp_logits_fair.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))
        
        
        mean_log_prob = log_prob.sum(1)/positives.shape[0]
        
        loss[i] = -mean_log_prob.mean()

    return loss.mean()

def ClusteringLoss(q_, target_q):
    '''
    Compute the clustering loss function

    Input:
        q_: predicted q
        target_q: target q
        phi: predicted phi
        target_phi: target phi

    Output:
        loss_cl: loss of q
    '''
    kl = KLDivLoss(size_average=False)
    loss_cl = kl(q_, target_q) 

    return loss_cl

def FairLoss(phi, target_phi):
    '''
    Compute the fair loss function
    Input:
        phi: predicted phi
        target_phi: target phi
    Output:
        loss_fr: loss of phi weighted by gamma
    '''
    # ipdb.set_trace()
    kl = KLDivLoss(size_average=False)
    loss_fr = kl(phi, target_phi)

    return loss_fr

# def ContrastiveLoss(points, sensitive_attribute, m):
#     '''
#     Compute the contrastive loss

#     Input:
#         points: points to compute the loss on
#         sensitive_attribute: sensitive attribute of the points
#         m: margin

#     Output:
#         loss_c: contrastive loss
#     '''
#     # Compute the similarity matrix
#     sim = similarity_matrix(points)

#     # Construct the 3-tuple with positive and negative points
#     positives = []
#     negatives = []

#     for i in range(sim.shape[0]):
#         distances = sim[i,:]

#         #Positive candidates
#         positive_candidates = 1e10*((sensitive_attribute==sensitive_attribute[i]) + (distances<=m))
#         # Negative candidates
#         negative_candidates = -1e10*((sensitive_attribute!=sensitive_attribute[i]) + (distances>m))
#         negative_candidates[i] = -1e10
        
#         # Select the positive and negative points
#         positive = torch.argmin(distances+positive_candidates) if sum(positive_candidates==0) > 0 else 0
#         negative = torch.argmax(distances+negative_candidates) if sum(negative_candidates==0) > 0 else 0

#         positives.append(points[positive,:].detach().cpu())
#         negatives.append(points[negative,:].detach().cpu())

#     # cast the points
#     # ipdb.set_trace()
#     positives = torch.cat(positives).to(DEVICE)
#     negatives = torch.cat(negatives).to(DEVICE)

#     # Compute the loss
#     relu = torch.nn.ReLU()
#     loss_c = torch.mean(relu(torch.sum((positives.reshape(sim.shape[0],-1)-points)**2,dim=1).sqrt() - torch.sum((negatives.reshape(sim.shape[0],-1)-points)**2,dim=1).sqrt()))
    
#     return loss_c

def pretrainSAE(args):
    '''
    Pretrain the SAE
    Input:
        args: arguments from argparse

    Output:
        sae: pre-trained SAE
    '''
    # Create a dataset from torch by using a tuple of numpys and pandas
    data = load_data(args, merged=True)[0]
    
    data_train_loader = DataLoader(dataset = data, batch_size=args.batch_size, shuffle=True)
    
    # Pretrain the autoencoder
    sae = SAE(input_size= data.X.shape[1], dropout=args.dropout, latent_size=args.latent_size_sae, hidden_sizes=args.hidden_sizes_sae).to(DEVICE)
    
    # optimizer = SGD(sae.parameters(), lr=args.lr_pretrain, momentum=0.9)
    optimizer = Adam(sae.parameters(), lr=args.lr_pretrain, weight_decay=1e-5)

    loss_epochs = []
    best = float('inf')
    sae.train()
    for epoch in range(args.num_epochs):
        loss = 0
        batches = 1
        for b_x, _, _ in data_train_loader:
            optimizer.zero_grad()
            b_x = b_x.to(DEVICE)
            # Forward pass
            output = sae(b_x)

            # Compute loss
            loss_batch = MSELoss()(output, b_x)

            # Backward pass
            loss_batch.backward()
            optimizer.step()

            loss += loss_batch.item()
            batches += 1

        # Print loss
        
        print(f"Epoch: {epoch}, Loss: {loss/batches: .4f}")
        
        loss_epochs.append(loss/batches)
        # Save sae if there is an improvement
        if args.save_checkpoints:
            save_checkpoint({'epoch': epoch,
                            'state_dict': sae.state_dict(),
                            'optimizer': optimizer.state_dict(), 
                            'loss': loss_epochs, 
                            'args': args}, 
                            f"resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt", 
                            loss/batches<best)

        best = loss/batches if loss/batches<best else best

def trainDEC(args):
    '''
    Train the DEC
    Input:
        args: arguments from argparse

    Output:
        dec: trained DEC

    '''
    # Load data
    # data_train, _ = load_data(args, train=True, test=True)
    data_train = load_data(args, merged=True)[0]

    # Create data loaders
    data_train_loader = DataLoader(dataset = data_train, batch_size=args.batch_size, shuffle=True)
    # data_test_loader = DataLoader(dataset = data_test, batch_size=args.batch_size, shuffle=True)

    # Initializing DEC
    # 1. Loading SAE
    assert os.path.isfile(f'resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt'), 'Pre-training not found. Pre-training SAE by using --pretrain'

    print('Loading SAE')
    # ipdb.set_trace()
    if torch.cuda.is_available():
        best_sae = torch.load(f"resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt")
    else: 
        best_sae = torch.load(f"resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt", map_location=torch.device('cpu'))
    sae = SAE(input_size= data_train.X.shape[1], 
            dropout=best_sae['args'].dropout, 
            latent_size=best_sae['args'].latent_size_sae, 
            hidden_sizes=best_sae['args'].hidden_sizes_sae).to(DEVICE)
    sae.load_state_dict(best_sae['state_dict'])

    # 2. Find initial cluster centers
    # Pass the data through the autoencoder
    sae.eval()
    features = []
    sensitives = []
    for b_x, b_s, _ in data_train_loader:
        b_x = b_x.to(DEVICE)
        b_s = b_s.to(DEVICE)
        features.append(sae.encode(b_x).detach().cpu())
        sensitives.append(b_s.detach().cpu())
    
    features = torch.cat(features)
    sensitives = torch.cat(sensitives)
    
    #=====K-means clustering=====
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=1)
    cluster_centers = kmeans.fit(features).cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True).to(DEVICE)
    cluster_centers = cluster_centers.cuda(non_blocking=True) if 'cuda' in DEVICE.type else cluster_centers

    #=====Fairoid centers=====
    unique_t, _ = torch.unique(data_train.S, return_counts=True)
    # fairoid_centers = torch.zeros(len(unique_t), data_train.X.shape[1], dtype=torch.float, requires_grad=False).to(DEVICE)
    fairoid_centers = torch.zeros(len(unique_t), args.latent_size_sae, dtype=torch.float, requires_grad=False).to(DEVICE)
    
    for t in range(len(unique_t)):
        # fairoid_centers[t] = torch.mean(data_train.X[data_train.S==unique_t[t]], dim=0)
        fairoid_centers[t] = torch.mean(features[sensitives==unique_t[t]], dim=0)
    
    # fairoid_centers = torch.tensor(fairoid_centers, dtype=torch.float, requires_grad=False).to(DEVICE)
    fairoid_centers = torch.tensor(fairoid_centers, dtype=torch.float, requires_grad=True).to(DEVICE)
    fairoid_centers = fairoid_centers.cuda(non_blocking=True) if 'cuda' in DEVICE.type else fairoid_centers
    
    # 3.1 Initialize DEC model
    dec = DEC(n_clusters = args.n_clusters,
                latent_size_sae=args.latent_size_sae, 
                hidden_sizes_sae=args.hidden_sizes_sae, 
                cluster_centers=cluster_centers, 
                fairoid_centers=fairoid_centers, 
                alpha= args.alpha, 
                beta = args.beta, 
                dropout= args.dropout, 
                autoencoder= sae, 
                p_norm=2).to(DEVICE)
    
    # 3.2 Initialize optimizer
    optimizer = SGD(dec.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = Adam(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 4. Initialize cluster centroids
    # with torch.no_grad():
    #     dec.state_dict()['clustering_layer.cluster_centers'].copy_(cluster_centers)
    #     dec.state_dict()['fairoid_layer.fairoid_centers'].copy_(fairoid_centers)
    # dec.fairoid_layer.fairoid_centers = fairoid_centers
    
    # Initial assignments
    #Visualization
    dec.eval()
    if args.plot: visualize(args.data, 0, data_train.X, data_train.Y, data_train.S, dec, args)
    assignments_prev = dec.soft_assignment(dec.project(data_train.X.to(DEVICE)))
    assignments_prev = assignments_prev.detach().argmax(dim=1).cpu().numpy()

    print('Training DEC')
    loss_iterations = []
    balance_iterations = []
    iterations = 0
    best = float('inf')
    best_balance = 0

    criterion = F.kl_div
    criterionFair = F.kl_div
    while True:
        dec.train()
        loss = 0
        # batches = 1
        iterations+=1
        # ipdb.set_trace()
        for b_x, b_s, _ in data_train_loader:
            
            b_x, b_s = b_x.to(DEVICE), b_s.to(DEVICE)

            # Forward pass
            b_x = b_x.cuda(non_blocking=True) if 'cuda' in DEVICE.type else b_x
            b_s = b_s.cuda(non_blocking=True) if 'cuda' in DEVICE.type else b_s
            x_proj = dec.project(b_x)
            q = dec.soft_assignment(x_proj)
            phi = dec.cond_prob(q, x_proj)

            # soft_assignment_q, _, x_proj = dec(b_x)
            target_q = dec.target_distribution_p(q).detach()
            target_phi = dec.target_distribution_phi(phi).detach()

            # Compute loss
            # ipdb.set_trace()
            fair_loss = criterionFair(phi.log(), target_phi, reduction='sum')
            # cluster_loss = ClusteringLoss(soft_assignment_q.log(), target_q)
            cluster_loss = criterion(q.log(), target_q, reduction='batchmean')
            # contrastive_loss_batch = ContrastiveLoss(x_proj.detach(), b_s.detach(), args.margin)
            # contrastive_loss_batch = ContrastiveLoss(x_proj.clone().detach(), b_s.detach(), args.temp)
            
            
            # ipdb.set_trace()
            # loss_batch = cluster_loss/b_x.shape[0] + args.rho*contrastive_loss_batch + args.gamma*fair_loss 
            # loss_batch = cluster_loss + args.rho*contrastive_loss_batch
            # loss_batch = cluster_loss
            loss_batch = cluster_loss + args.gamma*fair_loss

            # Backward pass
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
#           ipdb.set_trace()

            # loss += loss_batch.item()
            # batches += 1

        # Save plot every args.plot_iter
        if iterations%args.plot_iter==0 and args.plot:
            print('Plotting results so far')
            visualize(args.data, 
                    iterations, 
                    data_train.X.detach(), 
                    data_train.Y.detach(), 
                    data_train.S.detach(), 
                    dec,
                    args)

        # Compute balance
        dec.eval()
        x_proj_iter = dec.project(data_train.X.to(DEVICE))
        q_iter = dec.soft_assignment(x_proj_iter)
        target_q_iter = dec.target_distribution_p(q_iter).detach()
        
        phi_iter = dec.cond_prob(q_iter, x_proj_iter)
        target_phi_iter = dec.target_distribution_phi(phi_iter).detach()


        fair_loss = criterionFair(phi_iter.log(), target_phi_iter, reduction='sum')
        cluster_loss = criterion(q_iter.log(), target_q_iter, reduction='batchmean')
        loss_iter = cluster_loss + args.gamma*fair_loss
        loss_iterations.append(loss_iter)
        
        assignment_iter = q_iter.detach().argmax(dim=1).cpu().numpy()
        b = balance(data_train.S.cpu().numpy(), assignment_iter, args.n_clusters)
        balance_iterations.append(b)
        # Print loss
        # print(f'centroids: {dec.clustering_layer.cluster_centers}')
        # print(f'fairoids: {dec.fairoid_layer.fairoid_centers}')
        print(f"Iteration: {iterations}, Loss: {loss_iter: .4f}, Balances: {b}")

        # Save dec if there is an improvement
        if args.save_checkpoints:
            save_checkpoint({'iterations': iterations,
                            'state_dict': dec.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss_iterations': loss_iterations,
                            'balance_iterations': balance_iterations,
                            'args': args}, 
                            f"resulting_clusterings/{args.data}/best_balance_dec_nclusters{args.n_clusters}_g{args.gamma}_rho{args.rho}_run{args.run_id}.pt", 
                            b.min()>best_balance)
        
        if args.save_checkpoints:
            save_checkpoint({'iterations': iterations,
                            'state_dict': dec.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss_iterations': loss_iterations,
                            'balance_iterations': balance_iterations,
                            'args': args}, 
                            f"resulting_clusterings/{args.data}/best_loss_dec_nclusters{args.n_clusters}_g{args.gamma}_rho{args.rho}_run{args.run_id}.pt", 
                            best>loss_iter)

        # Save loss if there is improvement
        best = loss_iter if loss_iter<best else best
        best_balance = b.min() if b.min()>best_balance else best_balance

        # Stop if iterations are greater than the limit or the loss is less than the tolerance
        total_dif = np.count_nonzero(assignment_iter-assignments_prev)
        print(f'Changes: {total_dif/assignment_iter.shape[0]: .2%}')
        if total_dif/assignment_iter.shape[0]<=args.tolerance or args.limit_it<=iterations:
            break
        
        assignments_prev = assignment_iter

    if args.save_checkpoints:
        save_checkpoint({'iterations': iterations,
                        'state_dict': dec.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss_iterations': loss_iterations,
                        'balance_iterations': balance_iterations,
                        'args': args}, 
                        f"resulting_clusterings/{args.data}/last_dec_nclusters{args.n_clusters}_g{args.gamma}_rho{args.rho}_run{args.run_id}.pt", 
                        True)


if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, help='Id running of the experiment')
    parser.add_argument('--data', type=str, default='census_income')
    parser.add_argument('--sampled', type=int, default=4000, help='number of sampling for plots')
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0, help='Parameter of q distribution, which is the assignment function')
    parser.add_argument('--beta', type=float, default=1000.0, help = 'Parameter for phi distribution, which is the conditional distribution of sensitive attribute given the centroid')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for fairness loss')
    parser.add_argument('--rho', type=float, default=10.0, help='Weight for Contrastive loss')
    parser.add_argument('--margin', type=float, default=1.0, help='Parameter to define positive and negative in contrastive loss')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature parameter')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--p_norm', type=int, default=2)
    parser.add_argument('--plot_iter', type=int, default=1, help='Number of iterations to pass to plot result so far')
    parser.add_argument('--tolerance', type=float, default=0.0001, help='Tolerance for early stopping DEC training')
    parser.add_argument('--limit_it', type=int, default= 75, help='Number of iterations for stopping DEC training')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train the SAE')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training the SAE and DEC')
    parser.add_argument('--lr_pretrain', type=float, default=0.0001, help='learning rate for pretraining SAE')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training DEC')
    parser.add_argument('--pretrain_sae', action = 'store_true', help = 'Use this flag to pretrain the autoencoder')
    parser.add_argument('--hidden_sizes_sae', type=list, nargs='+', default=[500, 500, 2000], help='List of hidden layer sizes for the autoencoder')
    parser.add_argument('--latent_size_sae', type=int, default=5, help='Latent size for the autoencoder')
    parser.add_argument('--save_checkpoints', action='store_true', help='Set to store checkpoints')
    parser.add_argument('--plot', action='store_true', help='Set to plot across iterations')
    
    args = parser.parse_args()
    print('=====================')    
    print("Running the following configuration")
    print(f'data: {args.data}, gamma: {args.gamma}, tolerance: {args.tolerance}, latent_size_sae: {args.latent_size_sae}, id_run: {args.run_id}, ')
    print(f'limit_it: {args.limit_it}, lr: {args.lr}, latent size: {args.latent_size_sae}')
    print()
    
    # Check if the dataset is valid
    assert args.data in DATA, f"Invalid dataset. Please choose from {DATA}"
    
    # Create a folder to save the model
    if not os.path.exists(f"resulting_clusterings/{args.data}"):
        os.makedirs(f"resulting_clusterings/{args.data}")

    # Create a folder to save the plots
    if not os.path.exists(f"plots/{args.data}"):
        os.makedirs(f"plots/{args.data}")

    # Pretrain autoencoder if specified
    if args.pretrain_sae:
        if not os.path.exists(f'resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt'):
            print('Pretraining SAE')
            pretrainSAE(args)
    
    print('=====================')
    print()

    # Train DEC
    print('DEC')
    trainDEC(args)