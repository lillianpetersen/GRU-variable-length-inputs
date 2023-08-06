import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

############################################
# Dataset
############################################

def create_Tensor_list(X):
	X_tensor_list = []
	for i in range(len(X)):
		X_tensor_list.append(torch.Tensor(X[i]))
	return X_tensor_list

class TextDataset(Dataset):
	def __init__(self, X, y):
		self.X = create_Tensor_list(X)  # store X as a list of Tensors
		self.y = torch.Tensor(np.array(y))  # store y as a pytorch Tensor
		self.len=len(self.X)				

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len


############################################
# Sort batches by length to minimize padding
############################################

def collate_batch(batch):
	''' Pads each X in a batch to the same length '''
	X_list, y = zip(*batch)
	X_tensor = torch.nn.utils.rnn.pad_sequence(X_list, batch_first=True, padding_value=0)
	y = torch.Tensor(y).reshape(-1,1)
	return X_tensor, y

class BatchSamplerSimilarLength(Sampler):
	'''
	Batch X into batches of similar length to minimize padding
	Addapted from https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71
	'''
	def __init__(self, dataset, batch_size, indices=None, shuffle=True):
		self.batch_size = batch_size
		self.shuffle = shuffle
		# get the indices and length
		self.indices = [(i, len(s[0])) for i, s in enumerate(dataset)]
		# if indices are passed, then use only the ones passed (for ddp)
		if indices is not None:
			self.indices = torch.tensor(self.indices)[indices].tolist()

	def __iter__(self):
		if self.shuffle:
			random.shuffle(self.indices)

		pooled_indices = []
		# create pool of indices with similar lengths
		for i in range(0, len(self.indices), self.batch_size * 100):
			pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
		self.pooled_indices = [x[0] for x in pooled_indices]

		# yield indices for current batch
		batches = [self.pooled_indices[i:i+self.batch_size] for i in range(0, len(self.pooled_indices), self.batch_size)]

		if self.shuffle:
			random.shuffle(batches)
		for batch in batches:
			yield batch

	def __len__(self):
		return len(self.pooled_indices) // self.batch_size

############################################
# GRU
############################################

class GRUNet(nn.Module):
	def __init__(self, vocab_size, hidden_dim, n_layers, device, dropout=0.2):
		super(GRUNet, self).__init__()
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.device = device
		
		self.gru = nn.GRU(vocab_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(hidden_dim, 1)
		self.relu = nn.ReLU()
		
	def forward(self, x, h):
		out, h = self.gru(x, h)
		out = self.fc(self.relu(out[:,-1]))
		return out, h
	
	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
		return hidden


