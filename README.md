# Influencer or Observer — User-level Multimodal Stacking

Predict a Twitter user's social role — **Influencer (1)** or **Observer (0)** — from their tweets
and user metadata. This repository reproduces the technical pipeline of the 88% winning solution
 and serves as the basis for introducing graph-based features (future work).

## Method Overview

A three-stage multimodal stacking ensemble:

```
        ┌─ Stage 1: LightGBM (1373 user-level structured features) ──→ prob A ─┐
user ──→ │                                                                     ├─→ Stage 3: HGB meta-model ─→ 0/1
        └─ Stage 2: CamemBERTa-v2 fine-tuned (user-card text) ─────────→ prob B ┘
```

Key ideas:
- **User reconstruction**: no user ID is provided, so `user.created_at` is used as an implicit ID to group tweets into users.
- **Feature engineering**: 343 tweet-level features → aggregated per user (mean/max/min/std) → 1373 user-level features.
- **Transductive recovery**: recover the deleted `followers/friends` counts from `quoted_status.user.*` across train and test.
- **User cards + multi-card**: structured info is written as French text and concatenated with tweets to bypass the 512-token limit and act as light augmentation.
- **Leakage control**: user-level cross-validation; OOF probabilities feed the meta-model.

## Repository Structure

```
.
├── src/
│   ├── config.py         # global config and hyperparameters
│   ├── features.py       # user reconstruction + feature engineering + aggregation
│   ├── stage1_lgbm.py    # LightGBM structured model
│   ├── stage2_bert.py    # CamemBERTa-v2 fine-tuning 
│   ├── stage3_stack.py   # HGB meta-model + submission
│   └── run_all.py        # one-command pipeline
├── data/                 # place train.jsonl / kaggle_test.jsonl here 
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

Place the official files under `data/`:
- `data/train.jsonl` (with `label`)
- `data/kaggle_test.jsonl` (no label; used to generate the submission)

## Usage

```bash
pip install -r requirements.txt

# No GPU: validate the tree pipeline first (skip BERT)
python src/run_all.py --skip-bert

# With GPU: full three-stage pipeline
python src/run_all.py
```

Each stage can also be run individually:
```bash
python src/features.py
python src/stage1_lgbm.py
python src/stage2_bert.py
python src/stage3_stack.py
```

The submission is written to `submissions/submission.csv` (two columns: `ID, Prediction`).


## License

MIT — see [LICENSE](LICENSE).
