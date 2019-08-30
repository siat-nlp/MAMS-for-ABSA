from src.module.attention.attention import Attention

class DotAttention(Attention):

    def __init__(self, dropout=0):
        super(DotAttention, self).__init__(dropout)

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        assert query.size(2) == key.size(2)
        return query.matmul(key.transpose(1, 2))