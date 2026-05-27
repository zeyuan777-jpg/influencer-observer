"""
stage2_bert.py - Stage 2: fine-tune CamemBERTa-v2 text model.

Key ideas:
  - User cards: write structured info as French text + concatenate tweets
  - Multi-card: users with >5 tweets get 2 cards (different subsets); augment at
    train time, average at predict time
  - 5-fold fine-tuning, produces OOF probabilities
Requires GPU. The base model runs on a Colab/Kaggle T4.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import config as C
from features import load_jsonl, build_user_key, parse_twitter_time, get_col

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Build user cards ----------
def build_user_cards(df, n_cards_for_rich=2):
    """
    Return {user_key: [card_text_1, (card_text_2)]}.
    Card template: structured fields first (French), tweets after, [SEP]-separated,
    missing fields filled with 'inconnu'.
    """
    df = df.copy()
    df["_uk"] = build_user_key(df)
    text = get_col(df, "extended_tweet.full_text")
    text = text.where(~text.isna(), get_col(df, "text")).fillna("").astype(str)
    df["_text"] = text
    df["_desc"] = get_col(df, "user.description").fillna("inconnu").astype(str)
    df["_age"] = (get_col(df, "created_at").apply(parse_twitter_time)
                  - get_col(df, "user.created_at").apply(parse_twitter_time)
                  ).dt.total_seconds() / 86400.0
    df["_statuses"] = get_col(df, "user.statuses_count")
    df["_listed"] = get_col(df, "user.listed_count")
    df["_is_reply"] = (~get_col(df, "in_reply_to_status_id").isna()).astype(int)

    cards = {}
    rng = np.random.RandomState(C.SEED)
    for uk, g in df.groupby("_uk"):
        desc = g["_desc"].iloc[0]
        age = g["_age"].mean()
        listed = g["_listed"].iloc[0]
        statuses = g["_statuses"].iloc[0]
        reply_ratio = g["_is_reply"].mean()
        header = (
            f"Biographie: {desc} | "
            f"Age du compte: {age:.0f} jours | "
            f"Listes: {listed} | Statuts: {statuses} | "
            f"Ratio reponses: {reply_ratio:.2f} | Tweets: "
        )
        tweets = g["_text"].tolist()
        n = len(tweets)
        if n > C.MULTI_CARD_THRESHOLD:
            # multi-card: sample two different subsets
            card_list = []
            for _ in range(n_cards_for_rich):
                idx = rng.choice(n, size=min(C.MULTI_CARD_THRESHOLD, n), replace=False)
                body = " [SEP] ".join(tweets[i] for i in idx)
                card_list.append(header + body)
            cards[uk] = card_list
        else:
            cards[uk] = [header + " [SEP] ".join(tweets)]
    return cards


# ---------- Dataset ----------
class CardDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, max_length=C.BERT_MAX_LEN,
                             padding="max_length", return_tensors="pt")
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def cards_to_flat(cards, label_map=None):
    """Flatten into (user_key, card_text, label) lists. A user may have multiple cards."""
    uks, txts, labs = [], [], []
    for uk, card_list in cards.items():
        for c in card_list:
            uks.append(uk); txts.append(c)
            labs.append(label_map[uk] if label_map is not None else 0)
    return uks, txts, labs


def run():
    train = load_jsonl(C.TRAIN_JSONL)
    test = load_jsonl(C.TEST_JSONL)

    # user-level labels
    lab = pd.to_numeric(get_col(train, C.LABEL_COL), errors="coerce")
    user_label = pd.DataFrame({"uk": build_user_key(train), "y": lab.values}).groupby("uk")["y"].first()
    user_keys = user_label.index.values
    y = user_label.values.astype(int)

    print("Building train user cards...")
    train_cards = build_user_cards(train)
    print("Building test user cards...")
    test_cards = build_user_cards(test)

    tokenizer = AutoTokenizer.from_pretrained(C.BERT_MODEL)

    oof = np.zeros(len(user_keys))
    uk_to_idx = {uk: i for i, uk in enumerate(user_keys)}
    label_map = {uk: int(y[i]) for i, uk in enumerate(user_keys)}

    test_uks = list(test_cards.keys())
    test_uk_to_idx = {uk: i for i, uk in enumerate(test_uks)}
    test_prob = np.zeros(len(test_uks))

    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    for fold, (tr, va) in enumerate(skf.split(user_keys, y)):
        print(f"\n===== Fold {fold} =====")
        tr_uks = set(user_keys[tr]); va_uks = set(user_keys[va])
        tr_cards = {uk: train_cards[uk] for uk in tr_uks if uk in train_cards}
        va_cards = {uk: train_cards[uk] for uk in va_uks if uk in train_cards}

        tr_u, tr_t, tr_l = cards_to_flat(tr_cards, label_map)
        va_u, va_t, va_l = cards_to_flat(va_cards, label_map)

        model = AutoModelForSequenceClassification.from_pretrained(C.BERT_MODEL, num_labels=2).to(DEVICE)
        args = TrainingArguments(
            output_dir=f"./work/bert_fold{fold}",
            learning_rate=C.BERT_LR,
            per_device_train_batch_size=C.BERT_BATCH,
            per_device_eval_batch_size=C.BERT_BATCH,
            num_train_epochs=C.BERT_EPOCHS,
            weight_decay=C.BERT_WEIGHT_DECAY,
            label_smoothing_factor=C.BERT_LABEL_SMOOTHING,
            warmup_ratio=C.BERT_WARMUP_RATIO,
            lr_scheduler_type="linear",
            logging_steps=200, save_strategy="no", report_to=[],
            fp16=(DEVICE == "cuda"),
        )
        trainer = Trainer(model=model, args=args,
                          train_dataset=CardDataset(tr_t, tr_l, tokenizer))
        trainer.train()

        # --- validation: average multi-card probabilities per user ---
        va_logits = trainer.predict(CardDataset(va_t, va_l, tokenizer)).predictions
        va_prob = torch.softmax(torch.tensor(va_logits), dim=1)[:, 1].numpy()
        tmp = {}
        for uk, p in zip(va_u, va_prob):
            tmp.setdefault(uk, []).append(p)
        for uk, ps in tmp.items():
            oof[uk_to_idx[uk]] = np.mean(ps)

        # --- test: this fold's model also predicts test, accumulate across folds ---
        te_u, te_t, te_l = cards_to_flat(test_cards, None)
        te_logits = trainer.predict(CardDataset(te_t, te_l, tokenizer)).predictions
        te_prob = torch.softmax(torch.tensor(te_logits), dim=1)[:, 1].numpy()
        tmp = {}
        for uk, p in zip(te_u, te_prob):
            tmp.setdefault(uk, []).append(p)
        for uk, ps in tmp.items():
            test_prob[test_uk_to_idx[uk]] += np.mean(ps) / C.N_FOLDS

        acc = accuracy_score(y[va], (oof[va] > 0.5).astype(int))
        print(f"[BERT] fold{fold} acc={acc:.4f}")
        del model, trainer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    oof_acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"\n[BERT] OOF acc = {oof_acc:.4f}  (report: 87.5%)")

    np.save(C.OOF_BERT, oof)
    pd.DataFrame({"user_key": user_keys, "bert_oof": oof}).to_parquet(C.WORK_DIR / "bert_oof.parquet")
    pd.DataFrame({"user_key": test_uks, "bert_test": test_prob}).to_parquet(C.WORK_DIR / "bert_test.parquet")
    print("BERT probabilities saved.")
    return oof_acc


if __name__ == "__main__":
    run()
