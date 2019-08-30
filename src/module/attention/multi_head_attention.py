from torch import nn
from torch.nn import init
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, attention, num_heads, hidden_size, key_size='default', value_size='default', out_size='default'):
        key_size = hidden_size // num_heads if key_size == 'default' else key_size
        value_size = hidden_size // num_heads if value_size == 'default' else value_size
        out_size = hidden_size if out_size == 'default' else out_size
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.query_projection = nn.Linear(hidden_size, num_heads * key_size)
        self.key_projection = nn.Linear(hidden_size, num_heads * key_size)
        self.value_projection = nn.Linear(hidden_size, num_heads * value_size)
        init.normal_(self.query_projection.weight, mean=0, std=math.sqrt(2.0 / (hidden_size + key_size)))
        init.normal_(self.key_projection.weight, mean=0, std=math.sqrt(2.0 / (hidden_size + key_size)))
        init.normal_(self.value_projection.weight, mean=0, std=math.sqrt(2.0 / (hidden_size + value_size)))
        self.output_projection = nn.Linear(num_heads * value_size, out_size)
        init.xavier_normal_(self.output_projection.weight)
        self.attention = attention

    def forward(self, query, key, value, mask=None):
        """
        query: FloatTensor (batch_size, hidden_size) or (batch_size, num_queries, hidden_size)
        key: FloatTensor (batch_size, time_step, hidden_size)
        value: FloatTensor (batch_size, time_step, hidden_size)
        mask: ByteTensor (batch_size, time_step) or ByteTensor (batch_size, num_queries, time_step)
        subsequent_mask: ByteTensor (num_queries, time_step)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        num_heads, key_size, value_size = self.num_heads, self.key_size, self.value_size
        batch_size, num_queries, time_step = query.size(0), query.size(1), key.size(1)
        query = self.query_projection(query).view(batch_size, num_queries, num_heads, key_size)
        key = self.key_projection(key).view(batch_size, time_step, num_heads, key_size)
        value = self.value_projection(value).view(batch_size, time_step, num_heads, value_size)
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(0).repeat(num_heads, 1, 1).view(-1, time_step)
            else:
                mask = mask.unsqueeze(0).repeat(num_heads, 1, 1, 1).view(-1, num_queries, time_step)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, num_queries, key_size)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, time_step, key_size)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, time_step, value_size)
        output = self.attention(query, key, value, mask)
        output = output.view(num_heads, batch_size, num_queries, value_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, num_queries, -1)
        output = self.output_projection(output)
        if single_query:
            output = output.squeeze(1)
        return output