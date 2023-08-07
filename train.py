import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
import classes
import base
import benchmark

############################################
# train model, given hyperparameters
############################################

def train_model(X_train, X_val, y_train, y_val, device, lr, hidden_dim, dropout, n_layers, vocab_size, batch_size, modeldir, save_descript):
	'''
	Inputs:
		X_train: list of numpy arrays (each seq_length x vocab_length)
		X_val: list of numpy arrays (each seq_length x vocab_length)
		y_train: list-like, continuous target for X_train (this code is made for regression)
		y_val: list-like, continuous target for X_val
		device: str, eg 'cuda:0'
		lr: learning rate
		hidden_dim: int, hidden dimension of GRU
		dropout: float between 0 and 1. dropout percentage
		n_layers: int, number of layers in the GRU
		vocab_size: int, vocab size
		batch_size: int, batch size
		model_dir: str, directory to save best model to
		save_descript: str, name to save best model under
	No output, just saves best model
	'''

	train_data = classes.TextDataset(X_train, y_train)
	val_data = classes.TextDataset(X_val, y_val)
	train_dataloader = DataLoader(train_data, 
				batch_sampler=classes.BatchSamplerSimilarLength(dataset=train_data, batch_size=batch_size), 
				collate_fn=classes.collate_batch) 
	val_dataloader = DataLoader(val_data, 
				batch_sampler=classes.BatchSamplerSimilarLength(dataset=val_data, batch_size=batch_size), 
				collate_fn=classes.collate_batch) 
	
	model = classes.GRUNet(vocab_size, hidden_dim, n_layers, device, dropout)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	val_loss_curve = []
	train_loss_curve = []

	best_loss = 3
	best_epoch = 0
	epoch = -1
	while(True):
		epoch += 1
		# Train model on training data
		epoch_loss = base.train(model, train_dataloader, optimizer, device=device, GRU=GRU)
		
		# Validate on validation data 
		val_loss = base.validate(model, val_dataloader, device=device, GRU=GRU) 
		
		# Record train and loss performance 
		train_loss_curve.append(epoch_loss)
		val_loss_curve.append(val_loss)

		if val_loss < best_loss:
			best_loss = val_loss
			torch.save(model.state_dict(), modeldir+'/best_'+save_descript+'.pt')
			best_epoch = epoch
			print(best_epoch, best_loss)
		
		if epoch > best_epoch+50: break

############################################
# apply trained model to val set
############################################

def apply_model(X_test, y_test, device, hidden_dim, dropout, n_layers, vocab_size, batch_size, modeldir, figdir, save_descript, title):
	'''
	Inputs:
		X_test: list of numpy arrays (each seq_length x vocab_length)
		y_test: list-like, continuous target for X_test (this code is made for regression)
		device: str, eg 'cuda:0'
		hidden_dim: int, hidden dimension of GRU
		dropout: float between 0 and 1. dropout percentage
		n_layers: int, number of layers in the GRU
		vocab_size: int, vocab size
		batch_size: int, batch size
		model_dir: str, directory best model was saved to
		save_descript: str, name best model was saved under
		title: str, graph title
	Output: RMSE, spearman correlation
	'''

	test_data = classes.TextDataset(X_test, y_test)
	test_dataloader = DataLoader(test_data, 
				batch_sampler=classes.BatchSamplerSimilarLength(dataset=test_data, batch_size=batch_size), 
				collate_fn=classes.collate_batch) 
	
	model = classes.GRUNet(vocab_size, hidden_dim, n_layers, device, dropout)

	model.load_state_dict(torch.load(modeldir+'/best_'+save_descript+'.pt'))
	model.eval()

	model = model.to(device)
	preds = []
	y_true = []
	with torch.no_grad(): 
		for batch in test_dataloader:
			X_batch, y_batch = batch
			X_batch = X_batch.to(device)

			h = model.init_hidden(X_batch.shape[0])
			h = h.data
			y_pred, h = model(X_batch, h)

			preds.extend(y_pred.tolist())
			y_true.extend(y_batch.tolist())
	rmse_test, corr_test = benchmark.benchmark(np.array(y_true), np.array(preds), title, save_descript, figdir)

	return rmse_test, corr_test


