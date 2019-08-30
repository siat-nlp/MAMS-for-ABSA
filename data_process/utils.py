import os
import numpy as np
import random
from xml.etree.ElementTree import parse
from pytorch_pretrained_bert import BertModel, BertTokenizer
from data_process.vocab import Vocab
from src.module.utils.constants import UNK, PAD_INDEX, ASPECT_INDEX
import spacy
import re
import json

url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')

def check(x):
    return len(x) >= 1 and not x.isspace()

def tokenizer(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def parse_sentence_term(path, lowercase=False):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        if lowercase:
            text = text.lower()
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            if lowercase:
                term = term.lower()
            polarity = aspectTerm.get('polarity')
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end
            data.append(piece)
    return data

def parse_sentence_category(path, lowercase=False):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        if lowercase:
            text = text.lower()
        aspectCategories = sentence.find('aspectCategories')
        if aspectCategories is None:
            continue
        for aspectCategory in aspectCategories:
            category = aspectCategory.get('category')
            polarity = aspectCategory.get('polarity')
            piece = text + split_char + category + split_char + polarity
            data.append(piece)
    return data

def category_filter(data, remove_list):
    remove_set = set(remove_list)
    filtered_data = []
    for text in data:
        if not text.split('__split__')[2] in remove_set:
            filtered_data.append(text)
    return filtered_data

def build_vocab(data, max_size, min_freq):
    if max_size == 'None':
        max_size = None
    vocab = Vocab()
    for piece in data:
        text = piece.split('__split__')[0]
        text = tokenizer(text)
        vocab.add_list(text)
    return vocab.get_vocab(max_size=max_size, min_freq=min_freq)

def save_term_data(data, word2index, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence = []
    aspect = []
    label = []
    context = []
    bert_token = []
    bert_segment = []
    td_left = []
    td_right = []
    f = lambda x: word2index[x] if x in word2index else word2index[UNK]
    g = lambda x: list(map(f, tokenizer(x)))
    d = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
        'conflict': 3
    }
    for piece in data:
        text, term, polarity, start, end = piece.split('__split__')
        start, end = int(start), int(end)
        assert text[start: end] == term
        sentence.append(g(text))
        aspect.append(g(term))
        label.append(d[polarity])
        left_part = g(text[:start])
        right_part = g(text[end:])
        context.append(left_part + [ASPECT_INDEX] + right_part)
        bert_sentence = bert_tokenizer.tokenize(text)
        bert_aspect = bert_tokenizer.tokenize(term)
        bert_token.append(bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'] + bert_aspect + ['[SEP]']))
        bert_segment.append([0] * (len(bert_sentence) + 2) + [1] * (len(bert_aspect) + 1))
        td_left.append(g(text[:end]))
        td_right.append(g(text[start:])[::-1])
        assert len(bert_token[-1]) == len(bert_segment[-1])
    max_length = lambda x: max([len(y) for y in x])
    sentence_max_len = max_length(sentence)
    aspect_max_len = max_length(aspect)
    context_max_len = max_length(context)
    bert_max_len = max_length(bert_token)
    td_left_max_len = max_length(td_left)
    td_right_max_len = max_length(td_right)
    num = len(data)
    for i in range(num):
        sentence[i].extend([0] * (sentence_max_len - len(sentence[i])))
        aspect[i].extend([0] * (aspect_max_len - len(aspect[i])))
        context[i].extend([0] * (context_max_len - len(context[i])))
        bert_token[i].extend([0] * (bert_max_len - len(bert_token[i])))
        bert_segment[i].extend([0] * (bert_max_len - len(bert_segment[i])))
        td_left[i].extend([0] * (td_left_max_len - len(td_left[i])))
        td_right[i].extend([0] * (td_right_max_len - len(td_right[i])))
    sentence = np.asarray(sentence, dtype=np.int32)
    aspect = np.asarray(aspect, dtype=np.int32)
    label = np.asarray(label, dtype=np.int32)
    context = np.asarray(context, dtype=np.int32)
    bert_token = np.asarray(bert_token, dtype=np.int32)
    bert_segment = np.asarray(bert_segment, dtype=np.int32)
    td_left = np.asarray(td_left, dtype=np.int32)
    td_right = np.asarray(td_right, dtype=np.int32)
    np.savez(path, sentence=sentence, aspect=aspect, label=label, context=context, bert_token=bert_token, bert_segment=bert_segment,
             td_left=td_left, td_right=td_right)

def save_category_data(data, word2index, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence = []
    aspect = []
    label = []
    bert_token = []
    bert_segment = []
    f = lambda x: word2index[x] if x in word2index else word2index[UNK]
    g = lambda x: list(map(f, tokenizer(x)))
    d = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
        'conflict': 3
    }
    cd = {
        'food': 0,
        'service': 1,
        'staff': 2,
        'price': 3,
        'ambience': 4,
        'menu': 5,
        'place': 6,
        'miscellaneous': 7
    }
    for piece in data:
        text, category, polarity = piece.split('__split__')
        sentence.append(g(text))
        aspect.append(cd[category])
        label.append(d[polarity])
        bert_sentence = bert_tokenizer.tokenize(text)
        bert_aspect = bert_tokenizer.tokenize(category)
        bert_token.append(bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'] + bert_aspect + ['[SEP]']))
        bert_segment.append([0] * (len(bert_sentence) + 2) + [1] * (len(bert_aspect) + 1))
        assert len(bert_token[-1]) == len(bert_segment[-1])
    max_length = lambda x: max([len(y) for y in x])
    sentence_max_len = max_length(sentence)
    bert_max_len = max_length(bert_token)
    num = len(data)
    for i in range(num):
        sentence[i].extend([0] * (sentence_max_len - len(sentence[i])))
        bert_token[i].extend([0] * (bert_max_len - len(bert_token[i])))
        bert_segment[i].extend([0] * (bert_max_len - len(bert_segment[i])))
    sentence = np.asarray(sentence, dtype=np.int32)
    aspect = np.asarray(aspect, dtype=np.int32)
    label = np.asarray(label, dtype=np.int32)
    bert_token = np.asarray(bert_token, dtype=np.int32)
    bert_segment = np.asarray(bert_segment, dtype=np.int32)
    np.savez(path, sentence=sentence, aspect=aspect, label=label, bert_token=bert_token, bert_segment=bert_segment)

def analyze_term(data):
    num = len(data)
    sentence_lens = []
    aspect_lens = []
    log = {'total': num}
    for piece in data:
        text, term, polarity, _, _ = piece.split('__split__')
        sentence_lens.append(len(tokenizer(text)))
        aspect_lens.append(len(tokenizer(term)))
        if not polarity in log:
            log[polarity] = 0
        log[polarity] += 1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    log['aspect_max_len'] = max(aspect_lens)
    log['aspect_avg_len'] = sum(aspect_lens) / len(aspect_lens)
    return log

def analyze_category(data):
    num = len(data)
    sentence_lens = []
    log = {'total': num}
    for piece in data:
        text, category, polarity = piece.split('__split__')
        sentence_lens.append(len(tokenizer(text)))
        if not polarity in log:
            log[polarity] = 0
        log[polarity] += 1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    return log

def load_glove(path, vocab_size, word2index):
    if not os.path.isfile(path):
        raise IOError('Not a file', path)
    glove = np.random.uniform(-0.01, 0.01, [vocab_size, 300])
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                glove[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    glove[PAD_INDEX, :] = 0
    return glove

def load_sentiment_matrix(glove_path, sentiment_path):
    sentiment_matrix = np.zeros((3, 300), dtype=np.float32)
    sd = json.load(open(sentiment_path, 'r', encoding='utf-8'))
    sd['positive'] = set(sd['positive'])
    sd['negative'] = set(sd['negative'])
    sd['neutral'] = set(sd['neutral'])
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            word = content[0]
            vec = np.array(list(map(float, content[1:])))
            if word in sd['positive']:
                sentiment_matrix[0] += vec
            elif word in sd['negative']:
                sentiment_matrix[1] += vec
            elif word in sd['neutral']:
                sentiment_matrix[2] += vec
    sentiment_matrix -= sentiment_matrix.mean()
    sentiment_matrix = sentiment_matrix / sentiment_matrix.std() * np.sqrt(2.0 / (300.0 + 3.0))
    return sentiment_matrix