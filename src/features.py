"""
features.py - Data loading, user reconstruction, feature engineering, user-level aggregation.

Pipeline:
  1. Read jsonl -> json_normalize (flatten)
  2. Reconstruct user IDs from user.created_at (composite key to avoid collisions)
  3. Build tweet-level features (temporal / text / engagement / profile / entity / freq / geo / quoted / generic)
  4. Transductive: recover deleted followers/friends from quoted_status.user.* across train+test
  5. Aggregate per user (mean/max/min/std + tweet count) -> user-level feature table
"""
import json
import re
import numpy as np
import pandas as pd
from pandas import json_normalize
from datetime import datetime
import config as C


# ---------- Loading ----------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return json_normalize(rows)


# ---------- Helpers ----------
def parse_twitter_time(s):
    """'Wed Mar 17 13:00:59 +0000 2021' -> datetime"""
    if pd.isna(s):
        return pd.NaT
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return pd.NaT


def safe_len(x):
    if isinstance(x, (list, tuple)):
        return len(x)
    return 0


def get_col(df, name, default=np.nan):
    """Safely get a column; return all-default series if missing."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


# ---------- User reconstruction ----------
def build_user_key(df):
    """Composite key: created_at + screen_name (if available) to avoid second-level collisions."""
    key = get_col(df, C.USER_KEY).astype(str)
    if C.USER_KEY_BACKUP in df.columns:
        key = key + "||" + get_col(df, C.USER_KEY_BACKUP).astype(str)
    return key


# ---------- Transductive: recover follower counts from quoted_status.user ----------
def build_followers_lookup(*dfs):
    """
    Scan quoted_status.user.* across all data (train+test) and build
    {quoted user's created_at -> {followers, friends, ...}} lookup table.
    """
    lookup = {}
    qkey = "quoted_status.user.created_at"
    qfields = {
        "followers_count": "quoted_status.user.followers_count",
        "friends_count": "quoted_status.user.friends_count",
        "listed_count": "quoted_status.user.listed_count",
        "statuses_count": "quoted_status.user.statuses_count",
        "favourites_count": "quoted_status.user.favourites_count",
    }
    for df in dfs:
        if qkey not in df.columns:
            continue
        for i in range(len(df)):
            k = df[qkey].iloc[i]
            if pd.isna(k):
                continue
            rec = {}
            for out, col in qfields.items():
                if col in df.columns:
                    v = df[col].iloc[i]
                    if not pd.isna(v):
                        rec[out] = v
            if rec:
                lookup[str(k)] = rec  # last write wins (most recent)
    return lookup


# ---------- Tweet-level features ----------
URL_RE = re.compile(r"http\S+")

def tweet_level_features(df, followers_lookup):
    """Build features per tweet. Returns numeric feature DataFrame + a user_key column."""
    feats = pd.DataFrame(index=df.index)
    feats["user_key"] = build_user_key(df)

    # --- Text (prefer full_text) ---
    text = get_col(df, "extended_tweet.full_text")
    text = text.where(~text.isna(), get_col(df, "text"))
    text = text.fillna("").astype(str)
    feats["txt_len"] = text.str.len()
    feats["txt_n_hashtag"] = text.str.count("#")
    feats["txt_n_mention"] = text.str.count("@")
    feats["txt_n_url"] = text.apply(lambda t: len(URL_RE.findall(t)))
    feats["txt_upper_ratio"] = text.apply(
        lambda t: sum(c.isupper() for c in t) / (len(t) + 1)
    )

    # --- Temporal ---
    user_ct = get_col(df, "user.created_at").apply(parse_twitter_time)
    tweet_ct = get_col(df, "created_at").apply(parse_twitter_time)
    age_days = (tweet_ct - user_ct).dt.total_seconds() / 86400.0  # account age in days
    feats["acct_age_days"] = age_days
    feats["acct_age_log"] = np.log1p(age_days.clip(lower=0))
    feats["tweet_hour"] = tweet_ct.dt.hour
    feats["tweet_weekday"] = tweet_ct.dt.weekday
    feats["tweet_is_weekend"] = (tweet_ct.dt.weekday >= 5).astype(float)

    # --- User engagement (raw + log) ---
    for col in ["user.statuses_count", "user.favourites_count", "user.listed_count"]:
        v = pd.to_numeric(get_col(df, col), errors="coerce")
        short = col.split(".")[-1]
        feats[f"u_{short}"] = v
        feats[f"u_{short}_log"] = np.log1p(v.clip(lower=0))

    # --- Transductive follower/friend counts ---
    uct_str = get_col(df, "user.created_at").astype(str)
    feats["t_followers"] = uct_str.map(lambda k: followers_lookup.get(k, {}).get("followers_count", np.nan))
    feats["t_friends"] = uct_str.map(lambda k: followers_lookup.get(k, {}).get("friends_count", np.nan))
    feats["t_followers_log"] = np.log1p(pd.to_numeric(feats["t_followers"], errors="coerce").clip(lower=0))

    # --- Profile metadata ---
    feats["has_desc"] = (~get_col(df, "user.description").isna()).astype(float)
    feats["desc_len"] = get_col(df, "user.description").fillna("").astype(str).str.len()
    feats["has_url"] = (~get_col(df, "user.url").isna()).astype(float)
    feats["has_location"] = (~get_col(df, "user.location").isna()).astype(float)
    feats["geo_enabled"] = pd.to_numeric(get_col(df, "user.geo_enabled"), errors="coerce").fillna(0)
    feats["default_profile"] = pd.to_numeric(get_col(df, "user.default_profile"), errors="coerce").fillna(0)
    feats["default_profile_image"] = pd.to_numeric(get_col(df, "user.default_profile_image"), errors="coerce").fillna(0)
    feats["use_bg_image"] = pd.to_numeric(get_col(df, "user.profile_use_background_image"), errors="coerce").fillna(0)
    feats["protected"] = pd.to_numeric(get_col(df, "user.protected"), errors="coerce").fillna(0)

    # --- Profile colors (RGB hex -> int) ---
    def hex_to_int(s):
        try:
            return int(str(s), 16)
        except Exception:
            return np.nan
    for col in ["user.profile_link_color", "user.profile_background_color",
                "user.profile_text_color", "user.profile_sidebar_fill_color"]:
        feats[f"col_{col.split('.')[-1]}"] = get_col(df, col).apply(hex_to_int)

    # --- Entity counts ---
    feats["ent_hashtags"] = get_col(df, "entities.hashtags").apply(safe_len)
    feats["ent_urls"] = get_col(df, "entities.urls").apply(safe_len)
    feats["ent_mentions"] = get_col(df, "entities.user_mentions").apply(safe_len)

    # --- Reply behavior (strong signal: influencer 20% vs observer 41%) ---
    feats["is_reply"] = (~get_col(df, "in_reply_to_status_id").isna()).astype(float)
    feats["is_quote"] = pd.to_numeric(get_col(df, "is_quote_status"), errors="coerce").fillna(0)

    # --- Source frequency encoding ---
    src = get_col(df, "source").fillna("unknown").astype(str)
    feats["source_freq"] = src.map(src.value_counts(normalize=True))

    # --- Tweet engagement ---
    feats["retweet_count"] = pd.to_numeric(get_col(df, "retweet_count"), errors="coerce")
    feats["favorite_count"] = pd.to_numeric(get_col(df, "favorite_count"), errors="coerce")

    return feats


# ---------- User-level aggregation ----------
def aggregate_to_user(tweet_feats, labels=None):
    """Aggregate each numeric feature with mean/max/min/std + tweet_count. Optionally attach label."""
    num_cols = [c for c in tweet_feats.columns if c != "user_key"]
    g = tweet_feats.groupby("user_key")[num_cols]
    agg = g.agg(["mean", "max", "min", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg["tweet_count"] = tweet_feats.groupby("user_key").size()

    if labels is not None:
        # label is consistent within a user; take the first
        lab = pd.DataFrame({"user_key": tweet_feats["user_key"], "label": labels.values})
        agg["label"] = lab.groupby("user_key")["label"].first()
    return agg.reset_index()


# ---------- Main ----------
def build_features():
    print("Loading data...")
    train = load_jsonl(C.TRAIN_JSONL)
    test = load_jsonl(C.TEST_JSONL)
    print(f"train tweets {len(train)}, test tweets {len(test)}")

    print("Building transductive follower lookup (scanning quoted_status in train+test)...")
    lookup = build_followers_lookup(train, test)
    print(f"  lookup covers {len(lookup)} quoted users")

    print("Building tweet-level features...")
    train_tf = tweet_level_features(train, lookup)
    test_tf = tweet_level_features(test, lookup)

    print("Aggregating to user level...")
    train_labels = pd.to_numeric(get_col(train, C.LABEL_COL), errors="coerce")
    train_user = aggregate_to_user(train_tf, labels=train_labels)
    test_user = aggregate_to_user(test_tf, labels=None)

    # Keep user_key -> challenge_id map for test, needed to expand predictions back to tweets
    test_map = pd.DataFrame({
        "user_key": build_user_key(test),
        C.ID_COL: get_col(test, C.ID_COL),
    })

    print(f"train users {len(train_user)}, test users {len(test_user)}, "
          f"user-level features {train_user.shape[1]-2}")
    train_user.to_parquet(C.TRAIN_USER_FEATS)
    test_user.to_parquet(C.TEST_USER_FEATS)
    test_map.to_parquet(C.WORK_DIR / "test_id_map.parquet")
    print("Features saved.")
    return train_user, test_user, test_map


if __name__ == "__main__":
    build_features()
