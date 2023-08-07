import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random

import classes
import base

############################################
# hyperparameter search objective
############################################

def objective(trial, X_train, X_val, y_train, y_val, device, vocab_size,  batch_size=None):
	# Generate the optimizers.
	if batch_size is None:
		batch_size = trial.suggest_categorical("batch_size", [16,32,64])
	lr = trial.suggest_float("lr", 1e-6, 2e-3, log=True)
	hidden_dim = trial.suggest_int("hidden_dim", 256, 2048)
	n_layers = trial.suggest_int("n_layers", 1, 10)
	dropout = trial.suggest_float("dropout", 0.1, 0.5)
	model = classes.GRUNet(vocab_size, hidden_dim, n_layers, device, dropout)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# data
	train_data = classes.TextDataset(X_train, y_train)
	val_data = classes.TextDataset(X_val, y_val)
	train_dataloader = DataLoader(train_data, 
				batch_sampler=classes.BatchSamplerSimilarLength(dataset=train_data, batch_size=batch_size), 
				collate_fn=classes.collate_batch) 
	val_dataloader = DataLoader(val_data, 
				batch_sampler=classes.BatchSamplerSimilarLength(dataset=val_data, batch_size=batch_size), 
				collate_fn=classes.collate_batch) 

	val_loss_curve = []
	train_loss_curve = []

	for epoch in range(80):
		# Train
		epoch_loss = base.train(model, train_dataloader, optimizer, device=device)
		# Validate
		val_loss = base.validate(model, val_dataloader, device=device) 

		# Record train and loss performance 
		train_loss_curve.append(epoch_loss)
		val_loss_curve.append(val_loss)

		trial.report(np.amin(val_loss), epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			return 2

	return np.amin(val_loss)

############################################
# code for optimizing over multiple GPUs (single node)
############################################

def get_num_gpus():
	"""Returns the number of GPUs available"""
	from pycuda import driver
	driver.init()
	num_gpus = driver.Device.count()
	return num_gpus

class MultiGPUObjective:
	def __init__(self, gpu_queue, X_train, X_val, y_train, y_val, batch_size, vocab_size):
		# Shared queue to manage GPU IDs.
		self.gpu_queue = gpu_queue
		self.X_train = X_train
		self.X_val = X_val
		self.y_train = y_train
		self.y_val = y_val
		self.batch_size = batch_size
		self.vocab_size = vocab_size

	def __call__(self, trial):
		# Fetch GPU ID for this trial.
		gpu_id = self.gpu_queue.get()

		# objective function
		loss = objective(trial, self.X_train, self.X_val, self.y_train, self.y_val, device='cuda:'+str(gpu_id), batch_size=self.batch_size, vocab_size=self.vocab_size)

		# Return GPU ID to the queue.
		self.gpu_queue.put(gpu_id)

		# GPU ID is stored as an objective value.
		return loss

