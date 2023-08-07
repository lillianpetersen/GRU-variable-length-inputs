import pandas as pd
import numpy as np

import optuna
from multiprocessing import Manager
from joblib import parallel_backend

import hyperparams
import train


batch_size = 64
device = 'cuda:0'

## For you to define
X_train = None
X_val = None
X_test = None
y_train = None
y_val = None
y_test = None
save_descript = None
title = None
vocab_size = None

print('Optimizing hyperparameters...')
study = optuna.create_study(direction="minimize")
n_gpu = get_num_gpus()
print('GPUs:', n_gpu)
if n_gpu>1:
	with Manager() as manager:
		gpu_queue = manager.Queue()
		for i in range(n_gpu):
			gpu_queue.put(i)
		with parallel_backend("multiprocessing", n_jobs=n_gpu):
			study.optimize(hyperparams.MultiGPUObjective(gpu_queue, X_train, X_val, y_train, y_val, batch_size=batch_size, vocab_size=vocab_size), n_trials=100, n_jobs=n_gpu)
else:
	func = lambda trial: hyperparams.objective(trial, X_train, X_val, y_train, y_val, device=device, vocab_size=vocab_size, batch_size=batch_size)
	study.optimize(func, n_trials=100)

## Re-train using best hyperparameters
print('Training using best hyperparameters')
best_trial = study.best_trial
print(best_trial.params)
params_df = pd.DataFrame(best_trial.params, index=[1])
params_df.to_csv(modeldir+'/best_'+save_descript+'.csv')

train.train_model(X_train, X_val, y_train, y_val, 
		device=device, lr=best_trial.params['lr'], batch_size=batch_size, 
		n_layers=best_trial.params['n_layers'],
		hidden_dim=best_trial.params['hidden_dim'], dropout=best_trial.params['dropout'],
		vocab_size=vocab_size,
		modeldir=modeldir, save_descript=save_descript)

## Test on test set
rmse, corr = train.apply_model(X_test, y_test, 
		device=device, batch_size=batch_size,
		hidden_dim=best_trial.params['hidden_dim'], 
		dropout=best_trial.params['dropout'], n_layers=best_trial.params['n_layers'],
		vocab_size=vocab_size,
		modeldir=modeldir, figdir=figdir,
		save_descript=save_descript, title=title)

