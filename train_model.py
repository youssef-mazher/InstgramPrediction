"""
Clean training script for the Instagram Engagement Predictor.

Run this file from the project root:
    python train_model.py

It creates:
    model.npz
    model_meta.json

The exported model is a compact set of decision-tree arrays, so the
Streamlit app does not need scikit-learn or a pickled sklearn estimator.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model.npz"
META_PATH = ROOT / "model_meta.json"
RANDOM_STATE = 42


def nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year, month, weekday):
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def us_holidays(year):
    days = set()
    fixed = [(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)]

    for month, day in fixed:
        d = date(year, month, day)
        days.add(d)
        if d.weekday() == 5:
            days.add(d - timedelta(days=1))
        elif d.weekday() == 6:
            days.add(d + timedelta(days=1))

    days.update(
        {
            nth_weekday(year, 1, 0, 3),
            nth_weekday(year, 2, 0, 3),
            last_weekday(year, 5, 0),
            nth_weekday(year, 9, 0, 1),
            nth_weekday(year, 10, 0, 2),
            nth_weekday(year, 11, 3, 4),
        }
    )
    return days


def bucket(hour):
    if hour in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    if hour in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    return "normal"


def prepare_grammy(df):
    d = df.copy()
    d["post_time"] = pd.to_datetime(
        d["post_time"], format="%Y-%m-%d_%H-%M-%S_UTC", errors="coerce"
    )
    d["Engagement"] = (
        pd.to_numeric(d["likes"], errors="coerce").fillna(0)
        + pd.to_numeric(d["comments"], errors="coerce").fillna(0)
    )
    d["post_date"] = d["post_time"].dt.date
    years = range(int(d["post_time"].dt.year.min()), int(d["post_time"].dt.year.max()) + 1)
    holidays = set().union(*(us_holidays(y) for y in years))
    d["is_holiday"] = d["post_date"].isin(holidays).astype(int)
    d["hour"] = d["post_time"].dt.hour
    d["reach_time_bucket"] = d["hour"].apply(bucket)

    d = d.sort_values(["user", "post_time"])
    d["user_post_count"] = d.groupby("user").cumcount()
    d["user_median_engagement"] = (
        d.groupby("user")["Engagement"]
        .transform(lambda x: x.expanding().median().shift(1))
        .fillna(0)
    )

    d["hashtag_bucket"] = pd.cut(
        d["number_hashtags"],
        bins=[-1, 0, 5, 15, 30],
        labels=["none", "low", "medium", "high"],
    )
    d["caption_length_bucket"] = pd.cut(
        d["length_caption"],
        bins=[-1, 50, 150, 300, 10000],
        labels=["short", "medium", "long", "very_long"],
    )
    d["is_weekend"] = d["publication_weekday"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    return d[
        [
            "followers", "post_images", "video", "carousel",
            "length_caption", "number_hashtags", "publication_weekday",
            "is_weekend", "is_holiday", "user_median_engagement",
            "user_post_count", "reach_time_bucket",
            "caption_length_bucket", "hashtag_bucket", "Engagement",
        ]
    ]


def prepare_analytics(df):
    d = df.copy()
    d["post_datetime"] = pd.to_datetime(
        d["post_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    d["Engagement"] = sum(
        pd.to_numeric(d[c], errors="coerce").fillna(0)
        for c in ["likes", "comments", "shares", "saves"]
    )
    d["post_date"] = d["post_datetime"].dt.date
    years = range(int(d["post_datetime"].dt.year.min()), int(d["post_datetime"].dt.year.max()) + 1)
    holidays = set().union(*(us_holidays(y) for y in years))
    d["is_holiday"] = d["post_date"].isin(holidays).astype(int)
    d["hour"] = d["post_datetime"].dt.hour
    d["reach_time_bucket"] = d["hour"].apply(bucket)

    d = d.sort_values(["account_id", "post_datetime"])
    d["user_post_count"] = d.groupby("account_id").cumcount()
    d["user_median_engagement"] = (
        d.groupby("account_id")["Engagement"]
        .transform(lambda x: x.expanding().median().shift(1))
        .fillna(0)
    )

    d["hashtag_bucket"] = pd.cut(
        d["hashtags_count"],
        bins=[-1, 0, 5, 15, 30],
        labels=["none", "low", "medium", "high"],
    )
    d["caption_length_bucket"] = pd.cut(
        d["caption_length"],
        bins=[-1, 50, 150, 300, 10000],
        labels=["short", "medium", "long", "very_long"],
    )
    d["is_weekend"] = d["day_of_week"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    d["post_images"] = (d["media_type"] == "image").astype(int)
    d["carousel"] = (d["media_type"] == "carousel").astype(int)
    d["video"] = (d["media_type"] == "reel").astype(int)
    d["publication_weekday"] = d["day_of_week"]
    d["number_hashtags"] = d["hashtags_count"]
    d["followers"] = d["follower_count"]
    d["length_caption"] = d["caption_length"]

    return d[
        [
            "followers", "post_images", "video", "carousel",
            "length_caption", "number_hashtags", "publication_weekday",
            "is_weekend", "is_holiday", "user_median_engagement",
            "user_post_count", "reach_time_bucket",
            "caption_length_bucket", "hashtag_bucket", "Engagement",
        ]
    ]


def encode(train_df, test_df, categorical, numeric):
    train_x = train_df[numeric].copy()
    test_x = test_df[numeric].copy()

    categories = {}
    for col in categorical:
        categories[col] = sorted(train_df[col].astype(str).unique().tolist())
        train_values = train_df[col].astype(str)
        test_values = test_df[col].astype(str)

        for value in categories[col]:
            name = f"{col}__{value}"
            train_x[name] = (train_values == value).astype(np.float32)
            test_x[name] = (test_values == value).astype(np.float32)

    return (
        train_x.astype(np.float32),
        test_x.astype(np.float32),
        categories,
    )


def export_forest(model, path):
    arrays = {}

    for i, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        arrays[f"cl_{i}"] = tree.children_left.astype(np.int32)
        arrays[f"cr_{i}"] = tree.children_right.astype(np.int32)
        arrays[f"feat_{i}"] = tree.feature.astype(np.int16)
        arrays[f"thr_{i}"] = tree.threshold.astype(np.float32)
        arrays[f"val_{i}"] = tree.value[:, 0, 0].astype(np.float32)

    np.savez_compressed(path, **arrays)


def main():
    grammy = pd.read_csv(DATA / "Grammy_IG_posts_v2.csv", sep=";")
    analytics = pd.read_csv(DATA / "Instagram_Analytics.csv")

    grammy = prepare_grammy(grammy)
    analytics = prepare_analytics(analytics)

    common = [
        "followers", "post_images", "video", "carousel",
        "length_caption", "number_hashtags", "publication_weekday",
        "is_weekend", "is_holiday", "user_median_engagement",
        "user_post_count", "reach_time_bucket",
        "caption_length_bucket", "hashtag_bucket",
    ]

    df = pd.concat(
        [grammy[common + ["Engagement"]], analytics[common + ["Engagement"]]],
        ignore_index=True,
    )

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE
    )

    categorical = [
        "publication_weekday",
        "reach_time_bucket",
        "caption_length_bucket",
        "hashtag_bucket",
    ]
    numeric = [
        "followers", "post_images", "video", "carousel",
        "length_caption", "number_hashtags", "is_weekend",
        "is_holiday", "user_median_engagement", "user_post_count",
    ]

    X_train, X_test, categories = encode(
        train_df, test_df, categorical, numeric
    )
    y_train = train_df["Engagement"].astype(np.float64)
    y_test = test_df["Engagement"].astype(np.float64)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
    }

    export_forest(model, MODEL_PATH)

    positive_history = train_df.loc[
        train_df["user_median_engagement"] > 0,
        "user_median_engagement",
    ]
    default_history = (
        float(positive_history.median())
        if len(positive_history)
        else float(train_df["user_median_engagement"].median())
    )

    positive_posts = train_df.loc[
        train_df["user_post_count"] > 0, "user_post_count"
    ]
    default_posts = int(
        round(
            float(positive_posts.median())
            if len(positive_posts)
            else train_df["user_post_count"].median()
        )
    )

    meta = {
        "version": 1,
        "n_trees": len(model.estimators_),
        "feature_names": X_train.columns.tolist(),
        "categorical_values": categories,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": "Engagement",
        "random_state": RANDOM_STATE,
        "metrics": metrics,
        "default_user_median_engagement": default_history,
        "default_user_post_count": default_posts,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Rows: {len(df):,}")
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:,.0f}")
    print(f"Test MAE: {metrics['mae']:,.0f}")
    print(f"Created: {MODEL_PATH}")
    print(f"Created: {META_PATH}")


if __name__ == "__main__":
    main()
