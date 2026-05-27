"""
run_all.py 
Runs the four stages in order.

Usage:
    python src/run_all.py              # run everything
    python src/run_all.py --skip-bert  # skip Stage 2 (no GPU; validate tree pipeline first)
"""
import argparse
import features
import stage1_lgbm
import stage3_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bert", action="store_true",
                    help="Skip Stage 2 BERT fine-tuning (no GPU); uses 0.5 placeholder probs.")
    args = ap.parse_args()

    print("\n========== Step 1-2: Feature engineering (reconstruct/transductive/aggregate) ==========")
    features.build_features()

    print("\n========== Stage 1: LightGBM ==========")
    stage1_lgbm.run()

    if not args.skip_bert:
        print("\n========== Stage 2: CamemBERTa-v2 fine-tuning (needs GPU) ==========")
        import stage2_bert
        stage2_bert.run()
    else:
        print("\n[Skipping Stage 2] Using 0.5 placeholder BERT probs; validates tree pipeline only.")
        import pandas as pd, config as C
        train = pd.read_parquet(C.TRAIN_USER_FEATS)
        test = pd.read_parquet(C.TEST_USER_FEATS)
        pd.DataFrame({"user_key": train["user_key"], "bert_oof": 0.5}).to_parquet(C.WORK_DIR / "bert_oof.parquet")
        pd.DataFrame({"user_key": test["user_key"], "bert_test": 0.5}).to_parquet(C.WORK_DIR / "bert_test.parquet")

    print("\n========== Stage 3: HGB stacking + submission ==========")
    stage3_stack.run()
    print("\nDone. Submission at ./submissions/submission.csv")


if __name__ == "__main__":
    main()
