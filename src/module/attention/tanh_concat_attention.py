import torch
from torch import nn
from torch.nn import init
from src.module.attention.attention import Attention

class TanhConcatAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(TanhConcatAttention, self).__init__(dropout)
        self.query_weights = nn.Parameter(torch.Tensor(query_size, 1))
        self.key_weights = nn.Parameter(torch.Tensor(key_size, 1))
        init.xavier_uniform_(self.query_weights)
        init.xavier_uniform_(self.key_weights)

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        batch_size, num_queries, time_step = query.size(0), query.size(1), key.size(1)
        query = query.matmul(self.query_weights).expand(batch_size, num_queries, time_step)
        key = key.matmul(self.key_weights).transpose(1, 2).expand(batch_size, num_queries, time_step)
        score = query + key
        score = torch.tanh(score)
        return score