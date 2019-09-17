import torch
from torch import nn
import numpy as np
import os
import yaml
from pytorch_pretrained_bert import BertModel
from src.aspect_category_model.recurrent_capsnet import RecurrentCapsuleNetwork
from src.aspect_category_model.bert_capsnet import BertCapsuleNetwork

def make_model(config):
    model_type = config['aspect_category_model']['type']
    if model_type == 'recurrent_capsnet':
        return make_recurrent_capsule_network(config)
    elif model_type == 'bert_capsnet':
        return make_bert_capsule_network(config)
    else:
        raise ValueError('No Supporting.')

def make_bert_capsule_network(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BertCapsuleNetwork(
        bert=bert,
        bert_size=config['bert_size'],
        capsule_size=config['capsule_size'],
        dropout=config['dropout'],
        num_categories=log['num_categories']
    )
    model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix.npy'))
    return model

def make_recurrent_capsule_network(config):
    embedding = make_embedding(config)
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    aspect_embedding = nn.Embedding(num_embeddings=8, embedding_dim=config['embed_size'])
    model = RecurrentCapsuleNetwork(
        embedding=embedding,
        aspect_embedding=aspect_embedding,
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        capsule_size=config['capsule_size'],
        dropout=config['dropout'],
        num_categories=log['num_categories']
    )
    model.load_sentiment(os.path.join(base_path, 'processed/sentiment_matrix.npy'))
    return model

def make_embedding(config):
    base_path = os.path.join(config['base_path'])
    log_path = os.path.join(base_path, 'log/log.yml')
    log = yaml.safe_load(open(log_path))
    vocab_size = log['vocab_size']
    config = config['aspect_category_model'][config['aspect_category_model']['type']]
    embed_size = config['embed_size']
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
    glove = np.load(os.path.join(base_path, 'processed/glove.npy'))
    embedding.weight.data.copy_(torch.tensor(glove))
    return embedding