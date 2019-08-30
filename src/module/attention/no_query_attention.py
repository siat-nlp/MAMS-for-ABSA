import torch
from torch import nn
from torch.nn import init

class NoQueryAttention(nn.Module):

    def __init__(self, query_size, attention):
        super(NoQueryAttention, self).__init__()
        self.query_size = query_size
        self.query = nn.Parameter(torch.Tensor(1, query_size))
        init.xavier_uniform_(self.query)
        self.attention = attention

    def forward(self, key, value, mask=None):
        batch_size = key.size(0)
        query = self.query.expand(batch_size, self.query_size)
        return self.attention(query, key, value, mask)