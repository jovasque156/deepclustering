import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
	def __init__(self, X, S, Y):
		'''
		Args:
			X: sparse matrix, representing the features
			S: numpy, representing the sensitive attribute. Assuming binary
			Y: numpy, representing the label.
		'''
		self.X = torch.FloatTensor(X.toarray()).to(DEVICE)
		self.S = torch.tensor(S).to(DEVICE)
		self.Y = torch.tensor(Y).to(DEVICE)

	def __len__(self):
		'''
		Returns:
			int: length of the dataset
		'''
		return len(self.X)

	def __getitem__(self, idx):
		'''
		Args:
			idx: int, index of the data point
		Returns:
			tuple: (X, S, Y), representing the features, sensitive attribute, and label
		'''
		return self.X[idx], self.S[idx], idx, self.Y[idx]