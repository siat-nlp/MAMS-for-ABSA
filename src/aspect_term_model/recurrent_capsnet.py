import torch
from torch import nn
import torch.nn.functional as F
from src.aspect_term_model.capsnet import CapsuleNetwork

class RecurrentCapsuleNetwork(CapsuleNetwork):

    def __init__(self, embedding, num_layers, bidirectional, capsule_size, dropout, num_categories):
        super(RecurrentCapsuleNetwork, self).__init__(
            embedding=embedding,
            hidden_size=embedding.embedding_dim * (2 if bidirectional else 1),
            capsule_size=capsule_size,
            dropout=dropout,
            num_categories=num_categories
        )
        embed_size = embedding.embedding_dim
        self.rnn = nn.GRU(
            input_size=embed_size * 2,
            hidden_size=embed_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.bidirectional = bidirectional

    def _sentence_encode(self, sentence, aspect, mask=None):
        batch_size, time_step, embed_size = sentence.size()
        aspect_aware_sentence = torch.cat((
            sentence, aspect.unsqueeze(1).expand(batch_size, time_step, embed_size)
        ), dim=-1)
        output, _ = self.rnn(aspect_aware_sentence)
        if self.bidirectional:
            sentence = sentence.unsqueeze(-1).expand(batch_size, time_step, embed_size, 2)
            sentence = sentence.contiguous().view(batch_size, time_step, embed_size * 2)
        output = output + sentence
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output