#Basics
import os
import numpy as np
import argparse
from utils import (
    save_checkpoint, 
    visualize, 
    load_data, 
    balance,
    similarity_matrix,
    acc,
    nmi,
    ari
)
from typing import List

import warnings
warnings.filterwarnings('ignore')

#Torch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import MSELoss, KLDivLoss
import torch.nn.functional as F

# resulting_clusterings
# from deep_clustering.sae import SAE
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

def pretrainSAE(data_path:str, 
                model:torch.nn.Module, 
                savepath:str,
                num_epochs:int,
                lr:float,
                batch_size:int):
    '''
    Pretrain the SAE
    Input:
        args: arguments from argparse

    Output:
        sae: pre-trained SAE
    '''
    # Create a dataset from torch by using a tuple of numpys and pandas
    data = load_data(data_path, merged=True)[0]
    data_train_loader = DataLoader(dataset = data, batch_size=batch_size, shuffle=True)
    
    # Pretrain the autoencoder
    # sae = SAE(input_size= data.X.shape[1], dropout=args.dropout, latent_size=args.latent_size_sae, hidden_sizes=args.hidden_sizes_sae).to(DEVICE)
    
    optimizer = SGD(model.autoencoder.parameters(), lr=.1, momentum=0.9)
    # optimizer = Adam(model.autoencoder.parameters(), lr=1e-4, weight_decay=1e-3)

    criterion = MSELoss()

    loss_epochs = []
    best = float('inf')
    # sae.train()

    for epoch in range(num_epochs):
        loss = 0
        batches = 1
        model.autoencoder.train()

        for b_x, _, _, _ in data_train_loader:
            b_x = b_x.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model.autoencoder(b_x)
                loss_batch = criterion(outputs, b_x)

                loss_batch.backward()
                optimizer.step()

            # # Forward pass
            # output = sae(b_x)

            # # Compute loss
            # loss_batch = MSELoss()(output, b_x)

            # # Backward pass
            # loss_batch.backward()
            # optimizer.step()

            loss += loss_batch.item()
            batches += 1

        # Print loss
        
        print(f"Epoch: {epoch+1}, Loss: {loss/batches: .4f}")
        
        loss_epochs.append(loss/batches)
        # Save sae if there is an improvement
        # if args.save_checkpoints:

        save_checkpoint({'epoch': epoch+1,
                        'state_dict': model.autoencoder.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        'loss': loss_epochs, 
                        'args': args}, 
                        savepath,
                        # f"resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt", 
                        loss/batches<best)

        best = loss/batches if loss/batches<best else best

def trainDEC(
        model:torch.nn.Module, 
        x:torch.tensor,
        s: torch.tensor,
        optimizer:torch.optim, 
        criterion:torch.nn, 
        assignment_last:float,
        y:torch.tensor=None,
        batch_size:int=256, 
        update_interval:int=30,
        update_freq:bool=False
        ):
        
    index_array = np.arange(x.shape[0])
    index = 0
    loss = 0
    count = 0
    for i in range(int(np.ceil(x.shape[0]/batch_size))):
        if i % update_interval == 0:
            with torch.no_grad():
                q = model(x)
                p = model.clustering_layer.target_distribution(q)  # update the auxiliary target distribution p
                assignment = q.argmax(1)
                
                bal = balance(s.cpu().numpy(), assignment.clone().detach().cpu().numpy(), model.clustering_layer.n_clusters)

                if update_freq and i != 0 :
                    if y is not None:
                        acc_ = np.round(acc(y.clone().detach().cpu().numpy(), assignment.clone().detach().cpu().numpy()), 5)
                        nmi_ = np.round(nmi(y.clone().detach().cpu().numpy().squeeze(), assignment.clone().detach().cpu().numpy()), 5)
                        loss = np.round(loss/count, 5)
                        print(f'iter {i}: ; acc: {acc_: .4f}, loss={loss: .4f}, balance = {bal}')
                    else:
                        loss = np.round(loss/count, 5)
                        nmi_ = np.round(nmi(assignment_last, assignment.clone().detach().cpu().numpy()), 5)
                        print(f'iter {i}: loss={loss: .4f}, balance = {bal}')

                assignment_last = assignment.detach().clone().cpu().numpy()
            
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]

            trainx = x[idx]
            trainassignment = p[idx]

            trainx = trainx.to(DEVICE)
            trainassignment = trainassignment.to(DEVICE)

            outputs = model(trainx)
            index = index + 1 if (index + 1) * batch_size < x.shape[0] else 0

            train_loss = criterion(outputs.log(), trainassignment)

            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            count +=1

    return loss/count
    


def train(
        sae_path:str,
        data_path:str,
        n_clusters:int,
        latent_size_sae:int,
        hidden_sizes_sae: List[int],
        alpha:float,
        beta:float,
        dropout:float,
        pretrain_sae:bool,
        lr:float,
        batch_size:int,
        num_epochs_sae:int,
        num_epochs_dec:int,
        ):
    
    # Load data
    # data_train, _ = load_data(args, train=True, test=True)
    data_train = load_data(data_path=data_path, 
                            train=False,
                            test=False,
                            merged=True)[0]

    # Create data loaders
    # data_train_loader = DataLoader(dataset = data_train, batch_size=args.batch_size, shuffle=True)
    # data_test_loader = DataLoader(dataset = data_test, batch_size=args.batch_size, shuffle=True)

    # Initialize DEC
    # ipdb.set_trace()
    dec = DEC(n_clusters = n_clusters,
            input_size=data_train.X.shape[-1],
            hidden_sizes_sae = hidden_sizes_sae,
            latent_size_sae = latent_size_sae,
            alpha = alpha,
            beta = beta,
            dropout = dropout
            )
    
    dec.to(DEVICE)

    # Initializing DEC
    # 1. Training/Loading SAE
    # assert os.path.isfile(f'resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt'), 'Pre-training not found. Pre-training SAE by using --pretrain'
    if pretrain_sae:
        print('Pre-training SAE')
        pretrainSAE(data_path=data_path, 
                    model=dec, 
                    savepath=sae_path,
                    num_epochs=num_epochs_sae,
                    lr=lr,
                    batch_size=batch_size)
        
    assert os.path.isfile(sae_path), 'Pre-training not found. Pre-training SAE by using --pretrain'
    
    if not pretrain_sae: print('Loading SAE')
    # ipdb.set_trace()
    dec.autoencoder.load_state_dict(torch.load(sae_path)['state_dict'])

    # 2. Find initial cluster centers
    # Pass the data through the autoencoder
    print('Initializing cluster centers')
    
    features = []
    sensitives = []
    for x, s, _, _ in data_train:
        x, s = x.to(DEVICE), s.to(DEVICE)
        features.append(dec.autoencoder.encoder(x).clone().detach().cpu())
        sensitives.append(s.clone().detach().cpu())
    
    #=====K-means clustering=====
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    assignment_last = kmeans.fit_predict(torch.cat(features).reshape(data_train.X.shape[0],-1))
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True).to(DEVICE)
    cluster_centers = cluster_centers.cuda(non_blocking=True) if 'cuda' in DEVICE.type else cluster_centers

    dec.state_dict()['clustering_layer.cluster_centers'].copy_(cluster_centers)

    # #=====Fairoid centers=====
    # unique_t, _ = torch.unique(data_train.S, return_counts=True)
    # # fairoid_centers = torch.zeros(len(unique_t), data_train.X.shape[1], dtype=torch.float, requires_grad=False).to(DEVICE)
    # fairoid_centers = torch.zeros(len(unique_t), args.latent_size_sae, dtype=torch.float, requires_grad=False).to(DEVICE)
    
    # for t in range(len(unique_t)):
    #     # fairoid_centers[t] = torch.mean(data_train.X[data_train.S==unique_t[t]], dim=0)
    #     fairoid_centers[t] = torch.mean(features[sensitives==unique_t[t]], dim=0)
    
    # # fairoid_centers = torch.tensor(fairoid_centers, dtype=torch.float, requires_grad=False).to(DEVICE)
    # fairoid_centers = torch.tensor(fairoid_centers, dtype=torch.float, requires_grad=True).to(DEVICE)
    # fairoid_centers = fairoid_centers.cuda(non_blocking=True) if 'cuda' in DEVICE.type else fairoid_centers
    
    # # 3.1 Initialize DEC model
    # dec = DEC(n_clusters = args.n_clusters,
    #             latent_size_sae=args.latent_size_sae, 
    #             hidden_sizes_sae=args.hidden_sizes_sae, 
    #             cluster_centers=cluster_centers, 
    #             fairoid_centers=fairoid_centers, 
    #             alpha= args.alpha, 
    #             beta = args.beta, 
    #             dropout= args.dropout, 
    #             autoencoder= sae, 
    #             p_norm=2).to(DEVICE)
    
    # 4. Initialize optimizer and criterion
    optimizer = SGD(dec.parameters(), lr=lr, momentum=0.9) #, weight_decay=1e-4)
    # optimizer = Adam(dec.model.parameters(), lr=1e-2, weight_decay=1e-3)
    criterion = F.kl_div
    criterion_fair = F.kl_div

    # optimizer = Adam(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 4. Initialize cluster centroids
    # with torch.no_grad():
    #     dec.state_dict()['clustering_layer.cluster_centers'].copy_(cluster_centers)
    #     dec.state_dict()['fairoid_layer.fairoid_centers'].copy_(fairoid_centers)
    # dec.fairoid_layer.fairoid_centers = fairoid_centers
    
    # Initial assignments
    #Visualization
    # dec.eval()
    # if args.plot: visualize(args.data, 0, data_train.X, data_train.Y, data_train.S, dec, args)
    # assignments_prev = dec.soft_assignment(dec.project(data_train.X.to(DEVICE)))
    # assignments_prev = assignments_prev.detach().argmax(dim=1).cpu().numpy()

    print('Training DEC')
    # loss_iterations = []
    # balance_iterations = []
    # iterations = 0
    # best = float('inf')
    # best_balance = 0

    # criterion = F.kl_div
    # criterionFair = F.kl_div
    # while True:

    # ipdb.set_trace()
    epochs = 0
    update_interval = data_train.X.shape[0]
    update_freq = False
    for e in range(num_epochs_dec):
        dec.train()
        loss = 0
        # batches = 1
        epochs+=1
        # ipdb.set_trace()
        
        
        x = data_train.X.to(DEVICE)
        s = data_train.S.to(DEVICE)
        y = data_train.Y.to(DEVICE)
        index_array = np.arange(x.shape[0])
        index = 0
        loss = 0
        count = 0

        for i in range(int(np.ceil(x.shape[0]/batch_size))):
            if i % update_interval == 0:
                with torch.no_grad():
                    x_proj = dec.autoencoder.encoder(x)
                    q, _ = dec(x)
                    p = dec.clustering_layer.target_distribution(q)  # update the auxiliary target distribution p
                    assignment = q.argmax(1)
                    
                    bal = balance(s.cpu().numpy(), assignment.clone().detach().cpu().numpy(), dec.clustering_layer.n_clusters)

                    if update_freq and i != 0:
                        if y is not None:
                            acc_ = np.round(acc(s.clone().detach().cpu().numpy(), assignment.clone().detach().cpu().numpy()), 5)
                            nmi_ = np.round(nmi(s.clone().detach().cpu().numpy().squeeze(), assignment.clone().detach().cpu().numpy()), 5)
                            loss = np.round(loss/count, 5)
                            print(f'iter {i+1}: ; acc: {acc_: .4f}, loss={loss: .4f}, balance = {bal}')
                        else:
                            loss = np.round(loss/count, 5)
                            nmi_ = np.round(nmi(assignment_last, assignment.clone().detach().cpu().numpy()), 5)
                            print(f'iter {i+1}: loss={loss: .4f}, balance = {bal}')


                    centroids = torch.zeros_like(dec.clustering_layer.cluster_centers)
                    for i in range(centroids.shape[0]):
                        # Select only the data points that belong to cluster i
                        cluster_data = x_proj[assignment == i]
                        # Compute the mean along each dimension
                        centroids[i] = torch.mean(cluster_data, dim=0)
                    
                    cluster_centers = torch.tensor(centroids, dtype=torch.float, requires_grad=True).to(DEVICE)
                    cluster_centers = cluster_centers.cuda(non_blocking=True) if 'cuda' in DEVICE.type else cluster_centers

                    dec.state_dict()['clustering_layer.cluster_centers'].copy_(cluster_centers)
                    # assignment_last = assignment.detach().clone().cpu().numpy()
                
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # ipdb.set_trace()
                idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
                
                trainx = x[idx]
                train_target = p[idx]

                trainx = trainx.to(DEVICE)
                train_target = train_target.to(DEVICE).detach()

                outputs = dec(trainx)
                cond_outputs = (train_target, dec.autoencoder.encoder(trainx))
                index = index + 1 if (index + 1) * batch_size < x.shape[0] else 0

                train_loss = criterion(outputs.log(), train_target)
                train_loss_fair = criterion_fair(cond_outputs.log(), )

                train_loss.backward()
                optimizer.step()

                loss += train_loss.item()
                count +=1
            
#           ipdb.set_trace()

        with torch.no_grad():
            q_eval = dec(x)
            p_eval = dec.clustering_layer.target_distribution(q_eval)  # update the auxiliary target distribution p
            assignment_eval = q.argmax(1)

            loss = criterion(q_eval.log(), p_eval)
            bal = balance(s.clone().detach().cpu().numpy(), assignment_eval.clone().detach().cpu().numpy(), dec.clustering_layer.n_clusters)
            acc_ = np.round(acc(y.clone().detach().cpu().numpy(), assignment_eval.clone().detach().cpu().numpy()), 5)
            nmi_ = np.round(nmi(y.clone().detach().cpu().numpy(), assignment_eval.clone().detach().cpu().numpy()), 5)

            print(f'epoch {e+1}: acc: {acc_: .4f}, nmi: {nmi_: .4f} loss={loss: .4f}, balance = {bal}')

        assignment_last = assignment.detach().clone().cpu().numpy()

        # Save plot every args.plot_iter
        # if iterations%args.plot_iter==0 and args.plot:
        #     print('Plotting results so far')
        #     visualize(args.data, 
        #             iterations, 
        #             data_train.X.detach(), 
        #             data_train.Y.detach(), 
        #             data_train.S.detach(), 
        #             dec,
        #             args)

    #     # Compute balance
    #     dec.eval()
    #     x_proj_iter = dec.project(data_train.X.to(DEVICE))
    #     q_iter = dec.soft_assignment(x_proj_iter)
    #     target_q_iter = dec.target_distribution_p(q_iter).detach()
        
    #     phi_iter = dec.cond_prob(q_iter, x_proj_iter)
    #     target_phi_iter = dec.target_distribution_phi(phi_iter).detach()


    #     fair_loss = criterionFair(phi_iter.log(), target_phi_iter, reduction='sum')
    #     cluster_loss = criterion(q_iter.log(), target_q_iter, reduction='batchmean')
    #     loss_iter = cluster_loss + args.gamma*fair_loss
    #     loss_iterations.append(loss_iter)
        
    #     assignment_iter = q_iter.detach().argmax(dim=1).cpu().numpy()
    #     b = balance(data_train.S.cpu().numpy(), assignment_iter, args.n_clusters)
    #     balance_iterations.append(b)
    #     # Print loss
    #     # print(f'centroids: {dec.clustering_layer.cluster_centers}')
    #     # print(f'fairoids: {dec.fairoid_layer.fairoid_centers}')
    #     print(f"Iteration: {iterations}, Loss: {loss_iter: .4f}, Balances: {b}")

    #     # Save dec if there is an improvement
    #     if args.save_checkpoints:
    #         save_checkpoint({'iterations': iterations,
    #                         'state_dict': dec.state_dict(),
    #                         'optimizer': optimizer.state_dict(),
    #                         'loss_iterations': loss_iterations,
    #                         'balance_iterations': balance_iterations,
    #                         'args': args}, 
    #                         f"resulting_clusterings/{args.data}/best_balance_dec_nclusters{args.n_clusters}_g{args.gamma}_rho{args.rho}_run{args.run_id}.pt", 
    #                         b.min()>best_balance)
        
    #     if args.save_checkpoints:
    #         save_checkpoint({'iterations': iterations,
    #                         'state_dict': dec.state_dict(),
    #                         'optimizer': optimizer.state_dict(),
    #                         'loss_iterations': loss_iterations,
    #                         'balance_iterations': balance_iterations,
    #                         'args': args}, 
    #                         f"resulting_clusterings/{args.data}/best_loss_dec_nclusters{args.n_clusters}_g{args.gamma}_rho{args.rho}_run{args.run_id}.pt", 
    #                         best>loss_iter)

    #     # Save loss if there is improvement
    #     best = loss_iter if loss_iter<best else best
    #     best_balance = b.min() if b.min()>best_balance else best_balance

    #     # Stop if iterations are greater than the limit or the loss is less than the tolerance
    #     total_dif = np.count_nonzero(assignment_iter-assignments_prev)
    #     print(f'Changes: {total_dif/assignment_iter.shape[0]: .2%}')
    #     if total_dif/assignment_iter.shape[0]<=args.tolerance or args.limit_it<=iterations:
    #         break
        
    #     assignments_prev = assignment_iter

    # if args.save_checkpoints:
    #     save_checkpoint({'iterations': iterations,
    #                     'state_dict': dec.state_dict(),
    #                     'optimizer': optimizer.state_dict(),
    #                     'loss_iterations': loss_iterations,
    #                     'balance_iterations': balance_iterations,
    #                     'args': args}, 
    #                     f"resulting_clusterings/{args.data}/last_dec_nclusters{args.n_clusters}_g{args.gamma}_rho{args.rho}_run{args.run_id}.pt", 
    #                     True)


if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='census_income')
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0, help='Parameter of q distribution, which is the assignment function')
    parser.add_argument('--beta', type=float, default=1000.0, help = 'Parameter for phi distribution, which is the conditional distribution of sensitive attribute given the centroid')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_epochs_sae', type=int, default=150, help='Number of epochs to train the SAE')
    parser.add_argument('--num_epochs_dec', type=int, default=150, help='Number of epochs to train the DEC')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training the SAE and DEC')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training DEC')
    parser.add_argument('--pretrain_sae', action = 'store_true', help = 'Use this flag to pretrain the autoencoder')
    parser.add_argument('--hidden_sizes_sae', type=list, nargs='+', default=[500, 500, 2000], help='List of hidden layer sizes for the autoencoder')
    parser.add_argument('--latent_size_sae', type=int, default=5, help='Latent size for the autoencoder')
    
    # parser.add_argument('--sampled', type=int, default=4000, help='number of sampling for plots')
    # parser.add_argument('--gamma', type=float, default=1.0, help='Weight for fairness loss')
    # parser.add_argument('--rho', type=float, default=10.0, help='Weight for Contrastive loss')
    # parser.add_argument('--margin', type=float, default=1.0, help='Parameter to define positive and negative in contrastive loss')
    # parser.add_argument('--temp', type=float, default=1.0, help='Temperature parameter')
    # parser.add_argument('--p_norm', type=int, default=2)
    # parser.add_argument('--plot_iter', type=int, default=1, help='Number of iterations to pass to plot result so far')
    # parser.add_argument('--tolerance', type=float, default=0.0001, help='Tolerance for early stopping DEC training')
    # parser.add_argument('--limit_it', type=int, default= 75, help='Number of iterations for stopping DEC training')
    # parser.add_argument('--lr_pretrain', type=float, default=0.0001, help='learning rate for pretraining SAE')
    
    # parser.add_argument('--save_checkpoints', action='store_true', help='Set to store checkpoints')
    # parser.add_argument('--plot', action='store_true', help='Set to plot across iterations')
    args = parser.parse_args()
    print('=====================')    
    print("Running the following configuration")
    # print(f'data: {args.data}, gamma: {args.gamma}, tolerance: {args.tolerance}, latent_size_sae: {args.latent_size_sae}, id_run: {args.run_id}, ')
    # print(f'limit_it: {args.limit_it}, lr: {args.lr}, latent size: {args.latent_size_sae}')
    print(f'{args}')
    print()


    # Check if the dataset is valid
    assert args.data in DATA, f"Invalid dataset. Please choose from {DATA}"
    
    # Create a folder to save the model
    if not os.path.exists(f"resulting_clusterings/{args.data}"):
        os.makedirs(f"resulting_clusterings/{args.data}")

    # Create a folder to save the plots
    if not os.path.exists(f"plots/{args.data}"):
        os.makedirs(f"plots/{args.data}")

    sae_path = f'resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt'
    data_path = f'datasets/{args.data}/{args.data}.pt'
    
    # Pretrain autoencoder if specified
    # if args.pretrain_sae:
    #     if not os.path.exists(f'resulting_clusterings/{args.data}/best_sae_{args.latent_size_sae}.pt'):
    #         print('Pretraining SAE')
    #         pretrainSAE(args)
    
    print('=====================')
    print()

    # Train DEC
    print('DEC')
    train(sae_path = sae_path,
        data_path = data_path,
        n_clusters=args.n_clusters,
        latent_size_sae=args.latent_size_sae,
        hidden_sizes_sae=args.hidden_sizes_sae,
        alpha=args.alpha,
        beta=args.beta,
        dropout=args.dropout,
        pretrain_sae=args.pretrain_sae,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs_sae=args.num_epochs_sae,
        num_epochs_dec=args.num_epochs_dec
        )