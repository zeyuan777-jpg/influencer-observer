"""
config.py - Global configuration
Organized along the 7-step pipeline.
"""
from pathlib import Path

# ---------- Paths ----------
DATA_DIR = Path("./data")                       # put train.jsonl / kaggle_test.jsonl here
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TEST_JSONL = DATA_DIR / "kaggle_test.jsonl"
WORK_DIR = Path("./work")                        # intermediate artifacts (feature tables / probs)
WORK_DIR.mkdir(exist_ok=True, parents=True)
SUB_DIR = Path("./submissions")
SUB_DIR.mkdir(exist_ok=True, parents=True)

# Intermediate artifacts
TRAIN_USER_FEATS = WORK_DIR / "train_user_features.parquet"
TEST_USER_FEATS = WORK_DIR / "test_user_features.parquet"
OOF_LGBM = WORK_DIR / "oof_lgbm.npy"             # LightGBM OOF probabilities on train
TEST_LGBM = WORK_DIR / "test_lgbm.npy"           # LightGBM probabilities on test
OOF_BERT = WORK_DIR / "oof_bert.npy"
TEST_BERT = WORK_DIR / "test_bert.npy"

# ---------- User reconstruction ----------
USER_KEY = "user.created_at"                     # implicit user ID (constant per user)
# Composite key: add screen_name to avoid second-level collisions (falls back to single key if absent)
USER_KEY_BACKUP = "user.screen_name"

# ---------- Cross-validation ----------
N_FOLDS = 5
SEED = 42

# ---------- LightGBM hyperparameters (from report) ----------
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_error",
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    reg_alpha=0.1,         # L1
    reg_lambda=1.0,        # L2
    subsample=0.8,         # row subsampling
    colsample_bytree=0.8,  # column subsampling
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

# ---------- CamemBERTa-v2 fine-tuning hyperparameters (from report) ----------
BERT_MODEL = "almanach/camembertav2-base"        # CamemBERTa-v2 base (768-dim)
BERT_LR = 2.32e-5
BERT_BATCH = 48
BERT_WEIGHT_DECAY = 0.019
BERT_EPOCHS = 2
BERT_LABEL_SMOOTHING = 0.005
BERT_WARMUP_RATIO = 0.10
BERT_MAX_LEN = 512
MULTI_CARD_THRESHOLD = 5      # users with more tweets than this get 2 cards

# ---------- HGB meta-model hyperparameters (from report) ----------
HGB_PARAMS = dict(
    learning_rate=0.03,
    max_depth=6,
    l2_regularization=0.5,
    max_iter=800,
    early_stopping=True,
    random_state=SEED,
)

LABEL_COL = "label"
ID_COL = "challenge_id"
