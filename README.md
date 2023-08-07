# GRU-variable-length-inputs

Functions for training a GRU (gated recurrent unit) that can work with variable length inputs. Hyperparameters are tuned via optuna, and the hyperparameter search is parallelized over multiple GPUs. 

See main.py for examples on how to run the hyperparameter search and how to train the GRU.

The function for batching input sequences by length to reduce padding was adapted from https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71
