"""
stage3_stack.py - Stage 3: HistGradientBoosting meta-model (stacking).

Inputs: LGBM prob + BERT prob + 1373 structured features + disagreement meta-features.
Output: final user-level prediction -> expand to tweet-level challenge_id -> submission csv.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import config as C


def run():
    train = pd.read_parquet(C.TRAIN_USER_FEATS)
    test = pd.read_parquet(C.TEST_USER_FEATS)
    test_map = pd.read_parquet(C.WORK_DIR / "test_id_map.parquet")

    y = train[C.LABEL_COL].astype(int).values

    # --- probabilities ---
    oof_lgbm = np.load(C.OOF_LGBM)
    test_lgbm = np.load(C.TEST_LGBM)
    bert_oof = pd.read_parquet(C.WORK_DIR / "bert_oof.parquet")
    bert_test = pd.read_parquet(C.WORK_DIR / "bert_test.parquet")

    # align BERT probabilities to train/test order by user_key
    train = train.merge(bert_oof, on="user_key", how="left")
    test = test.merge(bert_test.rename(columns={"bert_test": "bert_oof"}), on="user_key", how="left")
    train["bert_oof"] = train["bert_oof"].fillna(0.5)
    test["bert_oof"] = test["bert_oof"].fillna(0.5)

    # --- assemble meta-features ---
    drop = ["user_key", C.LABEL_COL, "bert_oof"]
    struct_cols = [c for c in train.columns if c not in drop]
    for c in struct_cols:
        if c not in test.columns:
            test[c] = np.nan

    def assemble(df, lgbm_p, bert_p):
        m = pd.DataFrame(index=df.index)
        m["p_lgbm"] = lgbm_p
        m["p_bert"] = bert_p
        m["disagree_abs"] = np.abs(lgbm_p - bert_p)                 # disagreement
        m["disagree_cls"] = ((lgbm_p > 0.5) != (bert_p > 0.5)).astype(int)
        m["p_mean"] = (lgbm_p + bert_p) / 2
        m["p_max"] = np.maximum(lgbm_p, bert_p)
        m["p_std"] = np.stack([lgbm_p, bert_p]).std(axis=0)
        for c in struct_cols:                                       # plus structured features
            m[c] = df[c].values
        return m

    X_meta = assemble(train, oof_lgbm, train["bert_oof"].values)
    X_meta_test = assemble(test, test_lgbm, test["bert_oof"].values)

    # --- 5-fold HGB ---
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold, (tr, va) in enumerate(skf.split(X_meta, y)):
        clf = HistGradientBoostingClassifier(**C.HGB_PARAMS)
        clf.fit(X_meta.iloc[tr], y[tr])
        oof[va] = clf.predict_proba(X_meta.iloc[va])[:, 1]
        test_pred += clf.predict_proba(X_meta_test)[:, 1] / C.N_FOLDS

    oof_acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"[STACK] final OOF acc = {oof_acc:.4f}  (report: 87.7%)")

    # --- user-level prediction -> 0/1 -> expand to tweet-level challenge_id ---
    test_user_pred = pd.DataFrame({
        "user_key": test["user_key"].values,
        "Prediction": (test_pred > 0.5).astype(int),
    })
    sub = test_map.merge(test_user_pred, on="user_key", how="left")
    sub["Prediction"] = sub["Prediction"].fillna(0).astype(int)
    sub = sub.rename(columns={C.ID_COL: "ID"})[["ID", "Prediction"]]
    out = C.SUB_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"Submission written: {out}  ({len(sub)} rows)")
    return oof_acc


if __name__ == "__main__":
    run()
