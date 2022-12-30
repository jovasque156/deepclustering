#Basics
import os
import numpy as np
import argparse
from utils import (
    save_checkpoint, 
    visualize, 
    load_data, 
    balance
)

import warnings
warnings.filterwarnings('ignore')

#Torch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import MSELoss, KLDivLoss

# resulting_clusterings
from deep_clustering.sae import SAE
from deep_clustering.dec import DEC
from sklearn.cluster import KMeans

# Debugging
import ipdb

#TODO: train by using the new implementation of cluster and dec, these have not been copied to hopper.
# Run the experiments on hopper.

DATA = ['mnist', 'fashion_mnist', 'cifar10', 'census_income', 'compas']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def FairLossFunction(q_,target_q, phi, target_phi, gamma=1.0):
    '''
    Compute the fair loss function
    Input:
        q_: predicted q
        target_q: target q
        phi: predicted phi
        target_phi: target phi
        gamma: weight of phi loss
    Output:
        loss_cl: loss of q
        loss_fr: loss of phi weighted by gamma
    '''
    # ipdb.set_trace()
    kl_cl = KLDivLoss(size_average=False)
    kl_fr = KLDivLoss(size_average=False)

    return kl_cl(q_, target_q)/q_.shape[0]+gamma*kl_fr(phi, target_phi)

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
    cluster_centers = cluster_centers.cuda(non_blocking=True)  if torch.cuda.is_available() else cluster_centers

    #=====Fairoid centers=====
    unique_t, _ = torch.unique(data_train.S, return_counts=True)
    # fairoid_centers = torch.zeros(len(unique_t), data_train.X.shape[1], dtype=torch.float, requires_grad=False).to(DEVICE)
    fairoid_centers = torch.zeros(len(unique_t), args.latent_size_sae, dtype=torch.float, requires_grad=False).to(DEVICE)
    
    for t in range(len(unique_t)):
        # fairoid_centers[t] = torch.mean(data_train.X[data_train.S==unique_t[t]], dim=0)
        fairoid_centers[t] = torch.mean(features[sensitives==unique_t[t]], dim=0)
    
    # fairoid_centers = torch.tensor(fairoid_centers, dtype=torch.float, requires_grad=False).to(DEVICE)
    fairoid_centers = torch.tensor(fairoid_centers, dtype=torch.float, requires_grad=True).to(DEVICE)
    fairoid_centers = fairoid_centers.cuda(non_blocking=True) if torch.cuda.is_available() else fairoid_centers
    
    # 3.1 Initialize DEC model
    dec = DEC(n_clusters = args.n_clusters, latent_size_sae=args.latent_size_sae, hidden_sizes_sae=args.hidden_sizes_sae, cluster_centers=cluster_centers, fairoid_centers=fairoid_centers, alpha= args.alpha, beta = args.beta, dropout= args.dropout, autoencoder= sae, p_norm=2).to(DEVICE)
    
    # 3.2 Initialize optimizer
    # optimizer = SGD(dec.parameters(), lr=args.lr, momentum=0.9)
    optimizer = Adam(dec.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 4. Initialize cluster centroids
    with torch.no_grad():
        dec.state_dict()['clustering_layer.cluster_centers'].copy_(cluster_centers)
        dec.state_dict()['fairoid_layer.fairoid_centers'].copy_(fairoid_centers)
    # dec.fairoid_layer.fairoid_centers = fairoid_centers
    
    # Initial assignments
    #Visualization
    dec.eval()
    visualize(args.data, 0, data_train.X, data_train.Y, data_train.S, dec, args)
    assignments_prev,_ = dec(data_train.X)
    assignments_prev = assignments_prev.detach().argmax(dim=1).cpu().numpy()

    print('Training DEC')
    loss_iterations = []
    balance_iterations = []
    dec.train()
    iterations = 0
    best = float('inf')
    while True:
        loss = 0
        batches = 1
        iterations+=1
        # ipdb.set_trace()
        for b_x, _, _ in data_train_loader:
            b_x = b_x.to(DEVICE)

            # Forward pass
            b_x = b_x.cuda(non_blocking=True) if torch.cuda.is_available() else b_x
            soft_assignment_q, cond_prob_group_phi = dec(b_x)
            target_q = dec.target_distribution_p(soft_assignment_q).detach()
            target_phi = dec.target_distribution_phi(cond_prob_group_phi).detach()

            # Compute loss
            loss_batch = FairLossFunction(soft_assignment_q.log(), target_q,
                                        cond_prob_group_phi.log(), target_phi,
                                        gamma=args.gamma)
            
            # Backward pass
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step(closure=None)
            
#           ipdb.set_trace()

            loss += loss_batch.item()
            batches += 1
        
        loss_iterations.append(loss/batches)
        
        # Save plot every args.plot_iter
        if iterations%args.plot_iter==0:
            print('Plotting results so far')
            visualize(args.data, 
                    iterations, 
                    data_train.X, 
                    data_train.Y, 
                    data_train.S, 
                    dec,
                    args)

        # Compute balance
        assignments,_ = dec(data_train.X)
        assignments = assignments.detach().argmax(dim=1).cpu().numpy()
        b = balance(data_train.S.cpu().numpy(), assignments, args.n_clusters)
        balance_iterations.append(b)

        # Print loss
        print(f"Iteration: {iterations}, Loss: {loss/batches: .4f}, Balances: {b}")

        # Save dec if there is an improvement
        save_checkpoint({'iterations': iterations,
                        'state_dict': dec.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss_iterations': loss_iterations,
                        'balance_iterations': balance_iterations,
                        'args': args}, 
                        f"resulting_clusterings/{args.data}/last_dec_nclusters{args.n_clusters}_run{args.run_id}.pt", 
                        best>loss/batches)
        
        # Save loss if there is improvement
        best = loss/batches if loss/batches<best else best

        # Stop if iterations are greater than the limit or the loss is less than the tolerance
        total_dif = np.count_nonzero(assignments-assignments_prev)
        print(f'Changes: {total_dif/assignments.shape[0]: .2%}')
        if total_dif/assignments.shape[0]<=args.tolerance:
            break
        
        assignments_prev = assignments


if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, help='Id running of the experiment')
    parser.add_argument('--data', type=str, default='census_income')
    parser.add_argument('--sampled', type=int, default=4000, help='number of sampling for plots')
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1000.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--p_norm', type=int, default=2)
    parser.add_argument('--plot_iter', type=int, default=1, help='Number of iterations to pass to plot result so far')
    parser.add_argument('--tolerance', type=float, default=0.001, help='Tolerance for early stopping DEC training')
    parser.add_argument('--limit_it', type=int, default= 75, help='Number of iterations for stopping DEC training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the SAE')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training the SAE and DEC')
    parser.add_argument('--lr_pretrain', type=float, default=0.0001, help='learning rate for pretraining SAE')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training DEC')
    parser.add_argument('--pretrain_sae', action = 'store_true', help = 'Use this flag to pretrain the autoencoder')
    parser.add_argument('--hidden_sizes_sae', type=list, nargs='+', default=[500, 500, 2000], help='List of hidden layer sizes for the autoencoder')
    parser.add_argument('--latent_size_sae', type=int, default=10, help='Latent size for the autoencoder')
    
    args = parser.parse_args()

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
        if not os.path.exists(f'results/{args.data}/sae_losses_{args.latent_size_sae}.pt'):
            print('Pretraining SAE')
            pretrainSAE(args)
    
    print('=====================')
    print()

    # Train DEC
    print('DEC')
    trainDEC(args)