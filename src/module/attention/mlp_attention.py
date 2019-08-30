import torch
from torch import nn
from torch.nn import init
from src.module.attention.attention import Attention

class MlpAttention(Attention):

    def __init__(self, query_size, key_size, out_size=100, dropout=0):
        super(MlpAttention, self).__init__(dropout)
        self.query_projection = nn.Linear(query_size, out_size)
        self.key_projection = nn.Linear(key_size, out_size)
        self.v = nn.Parameter(torch.FloatTensor(out_size, 1))
        init.xavier_uniform_(self.v)

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        batch_size, num_queries, time_step, out_size = query.size(0), query.size(1), key.size(1), self.v.size(0)
        query = self.query_projection(query).unsqueeze(-1).expand(batch_size, num_queries, time_step, out_size)
        key = self.key_projection(key).unsqueeze(1).expand(batch_size, num_queries, time_step, out_size)
        score = torch.tanh(query + key).matmul(self.v).squeeze(-1)
        return score