import torch
import torch.nn as nn
import torch.nn.functional as F


class MFModel(nn.Module):
    def __init__(self, num_reviewers, num_products, emb_sz, sparse):
        super().__init__()
        self.reviewer_embeddings = nn.Embedding(
            num_reviewers, emb_sz, sparse=sparse)
        self.product_embeddings = nn.Embedding(
            num_products, emb_sz, sparse=sparse)

        self.reviewer_biases = nn.Embedding(num_reviewers, 1, sparse=sparse)
        self.product_biases = nn.Embedding(num_products, 1, sparse=sparse)

        torch.nn.init.xavier_uniform_(self.reviewer_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.product_embeddings.weight)

    def forward(self, product_id, reviewer_id):
        bias = self.reviewer_biases(reviewer_id) + \
            self.product_biases(product_id)
        pred = bias + (
            (self.reviewer_embeddings(reviewer_id)
             * self.product_embeddings(product_id))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()


class MFNeuralNetwork(nn.Module):
    def __init__(self, num_reviewers, num_products, emb_sz, sparse):
        super().__init__()
        self.reviewer_embeddings = nn.Embedding(num_reviewers, emb_sz)
        self.product_embeddings = nn.Embedding(num_products, emb_sz)
        self.linear1 = nn.Linear(2*emb_sz, 64)
        self.linear2 = nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.reviewer_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.product_embeddings.weight)

    def forward(self, product_id, reviewer_id):

        self.emb_out = torch.cat([self.reviewer_embeddings(
            reviewer_id), self.product_embeddings(product_id)], dim=1)
        self.out1 = F.relu(self.linear1(self.emb_out))
        self.out2 = self.linear2(self.out1).squeeze(-1)

        return self.out2
