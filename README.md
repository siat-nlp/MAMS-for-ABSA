# MAMS-for-ABSA
 
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