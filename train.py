#Basics
import os
import numpy as np
import argparse
from utils import save_checkpoint, visualize

import warnings
warnings.filterwarnings('ignore')


#Torch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import MSELoss, KLDivLoss, Parameter

# Models
from datasets.dataset import CustomDataset
from deep_clustering.sae import SAE
from deep_clustering.dec import DEC
from sklearn.cluster import KMeans

# Debugging
import ipdb

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
	kl_cl = KLDivLoss(size_average=False)
	kl_fr = KLDivLoss(size_average=False)

	return kl_cl(q_, target_q)+gamma*kl_fr(phi, target_phi)

def load_data(args, train=True, test=True):
	'''
	Load data from datasets/ folder
	Input:
		args: arguments from argparse
		train: load train data
		test: load test data
	
	Output:
		data_train: train data
		data_test: test data
	'''
	#Load data from datasets/ folder
	checkpoint = torch.load(f"datasets/{args.data}/{args.data}.pt")
	
	#Load train data	
	if train:
		X_train, S_train, Y_train = checkpoint['train']
		data_train = CustomDataset(X_train, S_train, Y_train)

	#Load test data
	if test:
		X_test, S_test, Y_test = checkpoint['test']
		data_test = CustomDataset(X_test, S_test, Y_test)

	if train and test:
		return data_train, data_test
	elif train:
		return data_train
	elif test:
		return data_test

def pretrainSAE(args):
	'''
	Pretrain the autoencoder
	'''
    # Create a dataset from torch by using a tuple of numpys and pandas
	data_train = load_data(args, train=True, test=False)
	
	data_train_loader = DataLoader(dataset = data_train, batch_size=args.batch_size, shuffle=True)
	
	# Pretrain the autoencoder
	sae = SAE(input_size= data_train.X.shape[1], 
			dropout=args.dropout, 
			latent_size=args.latent_size_sae, 
			hidden_sizes=args.hidden_sizes_sae).to(DEVICE)
	optimizer = Adam(sae.parameters(), lr=args.lr, weight_decay=1e-5)

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

		# Save sae if there is an improvement
		save_checkpoint({'state_dict': sae.state_dict(),
						'optimizer': optimizer.state_dict(),
						'loss': loss/batches,
						'epoch': epoch,
						'args': args}, 
						f"models/{args.data}/best_sae.pt", 
						loss/batches<best)

		best = loss/batches if loss/batches<best else best

		loss_epochs.append(loss/batches)


	# Save the results
	torch.save(loss_epochs, f"results/{args.data}/sae_losses.pt")

def trainDEC(args):
	# Load data
	data_train, _ = load_data(args, train=True, test=True)

	# Create data loaders
	data_train_loader = DataLoader(dataset = data_train, batch_size=args.batch_size, shuffle=True)
	# data_test_loader = DataLoader(dataset = data_test, batch_size=args.batch_size, shuffle=True)

	# Initializing DEC
	# 1. Loading SAE
	assert os.path.isfile(f'models/{args.data}/best_sae.pt'), 'Pre-training not found. Pre-training SAE by using --pretrain'

	print('Loading SAE')
	best_sae = torch.load(f"models/{args.data}/best_sae.pt")
	sae = SAE(input_size= data_train.X.shape[1], 
			dropout=best_sae['args'].dropout, 
			latent_size=best_sae['args'].latent_size_sae, 
			hidden_sizes=best_sae['args'].hidden_sizes_sae).to(DEVICE)
	sae.load_state_dict(best_sae['state_dict'])

	# 2. Find initial cluster centers
	# Pass the data through the autoencoder
	sae.eval()
	features = []
	for b_x, _, _ in data_train_loader:
		b_x = b_x.to(DEVICE)
		features.append(sae.encode(b_x).detach().cpu())
	features = torch.cat(features)
	#=====K-means clustering=====
	kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
	cluster_centers = kmeans.fit(features).cluster_centers_
	cluster_centers = torch.from_numpy(cluster_centers).to(DEVICE)

	#=====Fairoid centers=====
	unique_t, _ = torch.unique(data_train.S, return_counts=True)
	fairoid_centers = torch.zeros(len(unique_t), args.latent_size_sae).to(DEVICE)
	for t in range(len(unique_t)):
		fairoid_centers[t] = torch.mean(features[data_train.S==unique_t[t]], dim=0)

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
	optimizer = Adam(sae.parameters(), lr=args.lr, weight_decay=1e-5)
	# loss_func = KLDivLoss(size_average=False)

	print('Training DEC')
	loss_iterations = []
	prev_loss = float('inf')
	dec.train()
	iterations = 0
	while True:
		loss = 0
		batches = 1
		iterations+=1
		for b_x, _, _ in data_train_loader:
			optimizer.zero_grad()
			b_x = b_x.to(DEVICE)
			# Forward pass
			soft_assignment_q, cond_prob_group_phi = dec(b_x)
			target_q = dec.target_distribution_p(soft_assignment_q).detach()
			target_phi = dec.target_distribution_phi(cond_prob_group_phi).detach()

			# Compute loss
			loss_batch = FairLossFunction(soft_assignment_q.log(), target_q,
										cond_prob_group_phi.log(), target_phi,
										gamma=args.gamma)

			loss_batch /=soft_assignment_q.shape[0]
			
			# Backward pass
			loss_batch.backward()
			optimizer.step()

			loss += loss_batch.item()
			batches += 1
		
		loss_iterations.append(loss/batches)
		
		# Save plot every 2 epochs
		if iterations%2==0:
			print('Plotting results so far')
			visualize(args.data, 
					iterations, 
					data_train.X, 
					data_train.Y, 
					data_train.S, 
					dec,
					num_clusters=args.n_clusters)
			
		# Print loss
		print(f"Epoch: {iterations}, Loss: {loss/batches: .4f}")

		# Save dec if there is an improvement
		save_checkpoint({'state_dict': dec.state_dict(),
						'optimizer': optimizer.state_dict(),
						'loss_iterations': loss/batches,
						'iterations': iterations,
						'args': args}, 
						f"models/{args.data}/last_dec.pt", 
						True)

		if iterations>=args.limit_it or abs((loss/batches-prev_loss))/prev_loss<args.tolerance:
			break
		
		prev_loss = loss/batches

	# Save the results
	torch.save(loss_iterations, f"results/{args.data}/dec_losses.pt")


if __name__=='__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='census_income')
	parser.add_argument('--n_clusters', type=int, default=10)
	parser.add_argument('--alpha', type=float, default=1.0)
	parser.add_argument('--beta', type=float, default=2.0)
	parser.add_argument('--gamma', type=float, default=1.0)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--p_norm', type=int, default=2)
	parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for early stopping DEC training')
	parser.add_argument('--limit_it', type=int, default= 50, help='Number of iterations for stopping DEC training')
	parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the SAE')
	parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training the SAE and DEC')
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--pretrain_sae', action = 'store_true', help = 'Use this flag to pretrain the autoencoder')
	parser.add_argument('--hidden_sizes_sae', type=list, nargs='+', default=[500, 500, 2000], help='List of hidden layer sizes for the autoencoder')
	parser.add_argument('--latent_size_sae', type=int, default=10, help='Latent size for the autoencoder')
	
	args = parser.parse_args()

	# Check if the dataset is valid
	assert args.data in DATA, f"Invalid dataset. Please choose from {DATA}"
    
	# Create a folder to save the model
	if not os.path.exists(f"models/{args.data}"):
		os.makedirs(f"models/{args.data}")

	# Create a folder to save the results
	if not os.path.exists(f"results/{args.data}"):
		os.makedirs(f"results/{args.data}")

	# Create a folder to save the plots
	if not os.path.exists(f"plots/{args.data}"):
		os.makedirs(f"plots/{args.data}")

	# Pretrain autoencoder if specified
	if args.pretrain_sae:
		print('Pretraining SAE')
		pretrainSAE(args)
	
	print('=====================')
	print()

	# Train DEC
	print('DEC')
	trainDEC(args)