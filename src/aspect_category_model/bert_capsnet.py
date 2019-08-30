import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from src.module.utils.constants import PAD_INDEX, INF
from src.module.utils.sentence_clip import sentence_clip
from src.module.attention.dot_attention import DotAttention
from src.module.attention.scaled_dot_attention import ScaledDotAttention
from src.module.attention.bilinear_attention import BilinearAttention
from src.module.attention.tanh_bilinear_attention import TanhBilinearAttention
from src.module.attention.concat_attention import ConcatAttention
from src.module.attention.tanh_concat_attention import TanhConcatAttention
from src.module.attention.mlp_attention import MlpAttention
from src.aspect_term_model.capsnet import squash
import numpy as np

class BertCapsuleNetwork(nn.Module):

    def __init__(self, bert, bert_size, capsule_size, dropout, num_categories):
        super(BertCapsuleNetwork, self).__init__()
        self.bert = bert
        self.bert_size = bert_size
        self.capsule_size = capsule_size
        self.aspect_transform = nn.Sequential(
            nn.Linear(bert_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.sentence_transform = nn.Sequential(
            nn.Linear(bert_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.norm_attention = BilinearAttention(capsule_size, capsule_size)
        self.guide_capsule = nn.Parameter(
            torch.Tensor(num_categories, capsule_size)
        )
        self.guide_weight = nn.Parameter(
            torch.Tensor(capsule_size, capsule_size)
        )
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.capsule_projection = nn.Linear(bert_size, bert_size * num_categories)
        self.dropout = dropout
        self.num_categories = num_categories
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.guide_capsule)
        init.xavier_uniform_(self.guide_weight)

    def load_sentiment(self, path):
        sentiment = np.load(path)
        e1 = np.mean(sentiment)
        d1 = np.std(sentiment)
        e2 = 0
        d2 = np.sqrt(2.0 / (sentiment.shape[0] + sentiment.shape[1]))
        sentiment = (sentiment - e1) / d1 * d2 + e2
        self.guide_capsule.data.copy_(torch.tensor(sentiment))

    def forward(self, bert_token, bert_segment):
        # BERT encoding
        encoder_layer, _ = self.bert(bert_token, bert_segment, output_all_encoded_layers=False)
        batch_size, segment_len = bert_segment.size()
        max_segment_len = bert_segment.argmax(dim=-1, keepdim=True)
        batch_arrange = torch.arange(segment_len).unsqueeze(0).expand(batch_size, segment_len).to(bert_segment.device)
        segment_mask = batch_arrange <= max_segment_len
        sentence_mask = segment_mask & (1 - bert_segment).byte()
        aspect_mask = bert_segment
        sentence_lens = sentence_mask.long().sum(dim=1, keepdim=True)
        aspect_lens = aspect_mask.long().sum(dim=1, keepdim=True)
        aspect = encoder_layer.masked_fill(aspect_mask.unsqueeze(-1) == 0, 0)
        aspect = aspect.sum(dim=1, keepdim=False) / aspect_lens.float()
        # sentence encode layer
        max_len = sentence_lens.max().item()
        sentence = encoder_layer[:, 0: max_len].contiguous()
        sentence_mask = sentence_mask[:, 0: max_len].contiguous()
        sentence = sentence.masked_fill(sentence_mask.unsqueeze(-1) == 0, 0)
        # primary capsule layer
        sentence = self.sentence_transform(sentence)
        primary_capsule = squash(sentence, dim=-1)
        # aspect capsule layer
        aspect = self.aspect_transform(aspect)
        aspect_capsule = squash(aspect, dim=-1)
        # aspect aware normalization
        norm_weight = self.norm_attention.get_attention_weights(aspect_capsule, primary_capsule, sentence_mask)
        # capsule guided routing
        category_capsule = self._capsule_guided_routing(primary_capsule, norm_weight)
        category_capsule_norm = torch.sqrt(torch.sum(category_capsule * category_capsule, dim=-1, keepdim=False))
        return category_capsule_norm

    def _capsule_guided_routing(self, primary_capsule, norm_weight):
        guide_capsule = squash(self.guide_capsule)
        guide_matrix = primary_capsule.matmul(self.guide_weight).matmul(guide_capsule.transpose(0, 1))
        guide_matrix = F.softmax(guide_matrix, dim=-1)
        guide_matrix = guide_matrix * norm_weight.unsqueeze(-1) * self.scale  # (batch_size, time_step, num_categories)
        category_capsule = guide_matrix.transpose(1, 2).matmul(primary_capsule)
        category_capsule = F.dropout(category_capsule, p=self.dropout, training=self.training)
        category_capsule = squash(category_capsule)
        return category_capsule