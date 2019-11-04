# MAMS-for-ABSA

This repository contains the data and code for the paper "A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis", EMNLP-IJCNLP 2019, [[paper]](https://www.aclweb.org/anthology/D19-1654.pdf).

## MAMS

MAMS is a challenge dataset for aspect-based sentiment analysis (ABSA), in which each sentences contain at least two aspects with different sentiment polarities. MAMS dataset contains two versions: one for aspect-term sentiment analysis (ATSA) and one for aspect-category sentiment analysis (ACSA).

## Requirements

```
pytorch==1.1.0
spacy==2.1.8
pytorch-pretrained-bert==0.6.2
adabound==0.0.5
pyyaml==5.1.2
numpy==1.17.2
scikit-learn==0.21.3
scipy==1.3.1
```

## Quick Start

Put the pretrained GloVe(http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) file `glove.840B.300d.txt` in folder `./data`.
Modify `config.py` to select task, model and hyper-parameters. When mode is set to `term`, base_path should point to an ATSA dataset. When mode is set to `category`, base_path should point to an ACSA dataset.

### Preprocessing

```
python preprocess.py
```

### Train

```
python train.py
```

### Test

```
python test.py
```

## Acknowledgement

The BERT model pretrained by huggingface(https://github.com/huggingface/pytorch-transformers) is used in our experiments.