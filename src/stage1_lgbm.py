"""
stage1_lgbm.py - Stage 1: LightGBM on structured features.

Produces OOF probabilities (for stacking) and test probabilities.
Note: after user reconstruction the data is one-row-per-user, so user-level
leakage is already eliminated; a plain StratifiedKFold over users suffices.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import config as C


def run():
    train = pd.read_parquet(C.TRAIN_USER_FEATS)
    test = pd.read_parquet(C.TEST_USER_FEATS)

    y = train[C.LABEL_COL].astype(int).values
    drop = ["user_key", C.LABEL_COL]
    feat_cols = [c for c in train.columns if c not in drop]
    for c in feat_cols:                 # align test columns
        if c not in test.columns:
            test[c] = np.nan
    X = train[feat_cols].values
    X_test = test[feat_cols].values

    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**C.LGBM_PARAMS)
        model.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            eval_metric="binary_error",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / C.N_FOLDS
        acc = accuracy_score(y[va], (oof[va] > 0.5).astype(int))
        print(f"[LGBM] fold{fold} acc={acc:.4f}")

    oof_acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"[LGBM] OOF acc = {oof_acc:.4f}  (report: 87.1%)")

    np.save(C.OOF_LGBM, oof)
    np.save(C.TEST_LGBM, test_pred)
    print("LightGBM probabilities saved.")
    return oof_acc


if __name__ == "__main__":
    run()
