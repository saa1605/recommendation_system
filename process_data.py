import pandas as pd
import numpy as np
import gzip
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


def load_data_into_df(datafile):
    data = []
    with gzip.open('Video_Games_5.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))

    print(len(data))
    df = pd.DataFrame(data)
    del data
    return df


def split_dataframe(df):
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['reviewerID'])

    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    return df_train, df_test


def create_vocab(df, field):
    id2int = {}
    int2id = {}
    for i, id in enumerate(df[field].unique()):
        id2int[id] = i
        int2id[i] = id

    return id2int, int2id


class RecSysDataset(Dataset):
    def __init__(self, df, product2int, reviwer2int):
        self.df = df
        self.ratings = df['overall'].values
        self.product_ids = df['asin'].values
        self.reviewer_ids = df['reviewerID'].values
        self.product2int = product2int
        self.reviewer2int = reviwer2int

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):

        rating = torch.tensor(self.ratings[idx] - 1, dtype=torch.float)
        product_id = torch.tensor(
            self.product2int[self.product_ids[idx]], dtype=torch.long)
        reviewer_id = torch.tensor(
            self.reviewer2int[self.reviewer_ids[idx]], dtype=torch.long)

        return {
            'rating': rating,
            'product_id': product_id,
            'reviewer_id': reviewer_id
        }
