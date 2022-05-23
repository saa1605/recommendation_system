import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import gzip
import pandas as pd
import numpy as np
import math
from urllib.request import urlopen
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import config
from process_data import load_data_into_df, split_dataframe, create_vocab, RecSysDataset
from model_training import train, test, engine
from rating_model import MFModel, MFNeuralNetwork
from recommendation_list import get_product_ratings
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

datafile = 'Video_Games_5.json.gz'

def print_metrics(test_ratings, test_predictions):
    print(metrics.recall_score(np.rint(test_ratings), np.rint(test_predictions), average='macro'))
    print(metrics.precision_score(np.rint(test_ratings), np.rint(test_predictions), average='macro'))
    print(metrics.f1_score(np.rint(test_ratings), np.rint(test_predictions), average='macro'))

def main(datafile, model_name='mf'):
    df = load_data_into_df(datafile)

    product2int, int2product = create_vocab(df, 'asin')
    reviewer2int, int2reviewer = create_vocab(df, 'reviewerID')

    df_train, df_test = split_dataframe(df)

    train_dataset = RecSysDataset(df_train, product2int, reviewer2int)
    test_dataset = RecSysDataset(df_test, product2int, reviewer2int)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)
    
    if model_name == 'mf':
        model = MFModel(len(reviewer2int), len(product2int), config.emb_sz, config.sparse)
        model = model.to(config.device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=config.learning_rate)

    else:
        model = MFNeuralNetwork(len(reviewer2int), len(product2int), config.emb_sz, config.sparse)
        model = model.to(config.device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer_reviewer = torch.optim.SparseAdam([model.reviewer_embeddings.weight, model.reviewer_biases.weight], lr=config.learning_rate)
        optimizer_product = torch.optim.SparseAdam([model.product_embeddings.weight, model.product_biases.weight], lr=config.learning_rate)
    
    train_losses, train_maes, test_losses, test_maes, best_test_loss, best_test_mae = engine(train_loader, test_loader, model, loss_fn, optimizer, config.epochs, config.device)

    top10product_ratings_user = get_product_ratings()


    if model_name == 'mf':
        test_ratings = np.load('ratings_mf.npy')
        test_predictions = np.load('best_predictions_mf')
        print_metrics(test_ratings, test_predictions)
    else:
        test_ratings = np.load('ratings_nn.npy')
        test_predictions = np.load('best_predictions_nn.npy')
        print_metrics(test_ratings, test_predictions) 