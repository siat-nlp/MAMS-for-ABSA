# MAMS-for-ABSA

This repository contains the data and code for the paper "A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis", EMNLP-IJCNLP 2019.

## MAMS

MAMS is a challenge dataset for aspect-based sentiment analysis (ABSA), in which each sentences contain at least two aspects with different sentiment polarities. MAMS dataset contains two versions: one for aspect-term sentiment analysis (ATSA) and one for aspect-category sentiment analysis (ACSA).

## Quick Start

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