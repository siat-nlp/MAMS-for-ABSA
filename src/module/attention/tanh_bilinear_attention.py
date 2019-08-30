import torch
from torch import nn
from torch.nn import init
from src.module.attention.attention import Attention

class TanhBilinearAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(TanhBilinearAttention, self).__init__(dropout)
        self.weights = nn.Parameter(torch.FloatTensor(query_size, key_size))
        init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(1))

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        score = torch.tanh(query.matmul(self.weights).matmul(key.transpose(1, 2)) + self.bias)
        return score