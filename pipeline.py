
import warnings
warnings.filterwarnings("ignore")

import joblib
import holidays
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# Configuration
# ============================================================
RANDOM_STATE = 42

GRAMMY_PATH = "/content/drive/MyDrive/last test/Copy of Grammy_IG_posts_v2.csv"
ANALYTICS_PATH = "/content/drive/MyDrive/last test/Copy of Instagram_Analytics.csv"

ARTIFACT_PATH = "model_artifacts.joblib"


# ============================================================
# Load the same two datasets as the notebook
# ============================================================
df_grammy_raw = pd.read_csv(GRAMMY_PATH, sep=";")
df_analytic_raw = pd.read_csv(ANALYTICS_PATH)

df_grammy_raw["post_time"] = pd.to_datetime(
    df_grammy_raw["post_time"],
    format="%Y-%m-%d_%H-%M-%S_UTC",
    errors="coerce",
)
df_grammy_raw["Engagement"] = (
    pd.to_numeric(df_grammy_raw["likes"], errors="coerce").fillna(0)
    + pd.to_numeric(df_grammy_raw["comments"], errors="coerce").fillna(0)
)

df_analytic_raw["post_datetime"] = pd.to_datetime(
    df_analytic_raw["post_datetime"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)
df_analytic_raw["Engagement"] = (
    pd.to_numeric(df_analytic_raw["likes"], errors="coerce").fillna(0)
    + pd.to_numeric(df_analytic_raw["comments"], errors="coerce").fillna(0)
    + pd.to_numeric(df_analytic_raw["shares"], errors="coerce").fillna(0)
    + pd.to_numeric(df_analytic_raw["saves"], errors="coerce").fillna(0)
)

print(
    f"grammy {df_grammy_raw.shape} "
    f"analytic {df_analytic_raw.shape} "
    f"combined {len(df_grammy_raw) + len(df_analytic_raw)}"
)


# ============================================================
# Feature engineering - same logic as the uploaded notebook
# ============================================================
def bucket(h):
    if h in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    elif h in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    return "normal"


def add_features_grammy(df):
    df = df.copy()

    df["post_date"] = df["post_time"].dt.date

    years = range(
        int(df["post_time"].dt.year.min()),
        int(df["post_time"].dt.year.max()) + 1,
    )
    ch = holidays.US(years=years)

    df["is_holiday"] = df["post_date"].isin(ch)
    df["hour"] = df["post_time"].dt.hour
    df["reach_time_bucket"] = df["hour"].apply(bucket)

    df = df.sort_values(["user", "post_time"])

    df["user_post_count"] = df.groupby("user").cumcount()

    df["user_median_engagement"] = (
        df.groupby("user")["Engagement"]
        .transform(lambda x: x.expanding().median().shift(1))
        .fillna(0)
    )

    df["hashtag_bucket"] = pd.cut(
        df["number_hashtags"],
        bins=[-1, 0, 5, 15, 30],
        labels=["none", "low", "medium", "high"],
    )

    df["caption_length_bucket"] = pd.cut(
        df["length_caption"],
        bins=[-1, 50, 150, 300, 10000],
        labels=["short", "medium", "long", "very_long"],
    )

    df["is_weekend"] = (
        df["publication_weekday"]
        .isin(["Saturday", "Sunday"])
        .astype(int)
    )

    df.drop(
        columns=[
            "post_time", "user", "post_date", "hour",
            "likes", "comments", "Unnamed: 0",
            "awards", "day_difference", "IM_performance",
        ],
        inplace=True,
        errors="ignore",
    )

    return df


def add_features_analytic(df):
    df = df.copy()

    df["post_date"] = df["post_datetime"].dt.date

    years = range(
        int(df["post_datetime"].dt.year.min()),
        int(df["post_datetime"].dt.year.max()) + 1,
    )
    ch = holidays.US(years=years)

    df["is_holiday"] = df["post_date"].isin(ch)
    df["hour"] = df["post_datetime"].dt.hour
    df["reach_time_bucket"] = df["hour"].apply(bucket)

    df = df.sort_values(["account_id", "post_datetime"])

    df["user_post_count"] = df.groupby("account_id").cumcount()

    df["user_median_engagement"] = (
        df.groupby("account_id")["Engagement"]
        .transform(lambda x: x.expanding().median().shift(1))
        .fillna(0)
    )

    df["hashtag_bucket"] = pd.cut(
        df["hashtags_count"],
        bins=[-1, 0, 5, 15, 30],
        labels=["none", "low", "medium", "high"],
    )

    df["caption_length_bucket"] = pd.cut(
        df["caption_length"],
        bins=[-1, 50, 150, 300, 10000],
        labels=["short", "medium", "long", "very_long"],
    )

    df["is_weekend"] = (
        df["day_of_week"]
        .isin(["Saturday", "Sunday"])
        .astype(int)
    )

    df["post_images"] = (df["media_type"] == "image").astype(np.int16)
    df["carousel"] = (df["media_type"] == "carousel").astype(np.int16)
    df["video"] = (df["media_type"] == "reel").astype(np.int16)

    df.drop(columns=["media_type"], inplace=True)

    df["publication_weekday"] = df["day_of_week"]
    df["number_hashtags"] = df["hashtags_count"]
    df["followers"] = df["follower_count"]
    df["length_caption"] = df["caption_length"]

    df.drop(
        columns=[
            "day_of_week", "hashtags_count",
            "follower_count", "caption_length",
        ],
        inplace=True,
    )

    df.drop(
        columns=[
            "post_datetime", "account_id", "post_date", "hour",
            "likes", "comments", "shares", "saves",
            "post_id", "account_type", "content_category",
            "traffic_source", "has_call_to_action",
            "post_hour", "reach", "impressions",
            "engagement_rate", "followers_gained",
            "performance_bucket_label",
        ],
        inplace=True,
        errors="ignore",
    )

    return df


df_grammy_fe = add_features_grammy(df_grammy_raw)
df_analytic_fe = add_features_analytic(df_analytic_raw)

# Same shared feature set as the notebook.
common = [
    "caption_length_bucket",
    "is_weekend",
    "followers",
    "is_holiday",
    "post_images",
    "video",
    "user_median_engagement",
    "publication_weekday",
    "length_caption",
    "hashtag_bucket",
    "Engagement",
    "user_post_count",
    "number_hashtags",
    "carousel",
    "reach_time_bucket",
]

df = pd.concat(
    [
        df_grammy_fe[common],
        df_analytic_fe[common],
    ],
    ignore_index=True,
)


# ============================================================
# Same random 80/20 split as the notebook
# ============================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

print("train/test:", train_df.shape, test_df.shape)

# ============================================================
# Hidden reference values for Streamlit
#
# The original notebook models still require these two features,
# but the marketer will not type them manually.
# Since no account is selected in the app, we use robust
# reference values learned from the training data.
# ============================================================
positive_history = train_df.loc[
    train_df["user_median_engagement"] > 0,
    "user_median_engagement"
]

default_user_median_engagement = float(
    positive_history.median()
    if len(positive_history) > 0
    else train_df["user_median_engagement"].median()
)

positive_post_counts = train_df.loc[
    train_df["user_post_count"] > 0,
    "user_post_count"
]

default_user_post_count = int(
    round(
        float(
            positive_post_counts.median()
            if len(positive_post_counts) > 0
            else train_df["user_post_count"].median()
        )
    )
)

print(
    "Hidden Streamlit reference values:",
    {
        "user_median_engagement": default_user_median_engagement,
        "user_post_count": default_user_post_count,
    }
)


# ============================================================
# Same notebook feature definitions
# ============================================================
catcol = [
    "video",
    "carousel",
    "publication_weekday",
    "caption_length_bucket",
    "hashtag_bucket",
    "reach_time_bucket",
    "is_weekend",
    "is_holiday",
]

numeric_features = [
    "followers",
    "post_images",
    "user_median_engagement",
    "length_caption",
    "user_post_count",
    "number_hashtags",
]


# ============================================================
# Equivalent train-fitted one-hot encoding
# ============================================================
train_processed = train_df.copy()
test_processed = test_df.copy()

for col in numeric_features:
    train_processed["log" + col] = np.log1p(train_processed[col])
    test_processed["log" + col] = np.log1p(test_processed[col])

    train_processed.drop(columns=col, inplace=True)
    test_processed.drop(columns=col, inplace=True)


try:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        drop="first",
    )
except TypeError:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
        drop="first",
    )

ohe.fit(train_processed[catcol])
ohe_cols = list(ohe.get_feature_names_out(catcol))

train_ohe = pd.DataFrame(
    ohe.transform(train_processed[catcol]),
    columns=ohe_cols,
    index=train_processed.index,
).astype(float)

test_ohe = pd.DataFrame(
    ohe.transform(test_processed[catcol]),
    columns=ohe_cols,
    index=test_processed.index,
).astype(float)

train_processed = pd.concat(
    [train_processed.drop(columns=catcol), train_ohe],
    axis=1,
).astype(float)

test_processed = pd.concat(
    [test_processed.drop(columns=catcol), test_ohe],
    axis=1,
).astype(float)

X_train = train_processed.drop("Engagement", axis=1)
y_train_raw = train_processed["Engagement"]

X_test = test_processed.drop("Engagement", axis=1)
y_test_raw = test_processed["Engagement"]

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# ============================================================
# Scaling - same as notebook for global models + KMeans
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 1) Single HGBR - exact notebook settings
# ============================================================
single_hgbr = HistGradientBoostingRegressor(
    max_depth=8,
    learning_rate=0.08,
    max_iter=300,
    random_state=42,
)
single_hgbr.fit(X_train_scaled, y_train_raw)

pred_single_hgbr = single_hgbr.predict(X_test_scaled)


# ============================================================
# 2) Single RF - exact notebook settings
# ============================================================
single_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
single_rf.fit(X_train_scaled, y_train_raw)

pred_single_rf = single_rf.predict(X_test_scaled)


# ============================================================
# KMeans router - exact notebook settings
# ============================================================
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

train_cluster = kmeans.fit_predict(X_train_scaled)
test_cluster = kmeans.predict(X_test_scaled)

silhouette = silhouette_score(
    X_train_scaled,
    train_cluster,
)

# Notebook derives Low / Mid / High from actual raw engagement means.
cluster_means = (
    pd.Series(y_train_raw.values)
    .groupby(train_cluster)
    .mean()
)

ranked = cluster_means.sort_values().index.tolist()

label_map = {
    int(ranked[0]): "Low",
    int(ranked[1]): "Mid",
    int(ranked[2]): "High",
}


# ============================================================
# 3) HGBR experts - exact notebook logic
# ============================================================
experts_hgbr = {}

for c in range(3):
    mask = train_cluster == c

    model = HistGradientBoostingRegressor(
        max_depth=8 if mask.sum() > 1000 else 6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )

    model.fit(
        X_train.loc[mask],
        y_train_raw.loc[mask],
    )

    experts_hgbr[c] = model


pred_moe_hgbr = np.zeros(len(X_test))

for c in range(3):
    mask = test_cluster == c

    if mask.sum() > 0:
        pred_moe_hgbr[mask] = experts_hgbr[c].predict(
            X_test.loc[mask]
        )


# ============================================================
# 4) RF experts - exact notebook logic
# ============================================================
experts_rf = {}

for c in range(3):
    mask = train_cluster == c

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8 if mask.sum() > 1000 else 6,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train.loc[mask],
        y_train_raw.loc[mask],
    )

    experts_rf[c] = model


pred_moe_rf = np.zeros(len(X_test))

for c in range(3):
    mask = test_cluster == c

    if mask.sum() > 0:
        pred_moe_rf[mask] = experts_rf[c].predict(
            X_test.loc[mask]
        )


# ============================================================
# 5) Mixed MoE - exact notebook logic
#
# cluster 0 -> Random Forest
# cluster 1 -> HGBR
# cluster 2 -> HGBR
# ============================================================
experts_mixed = {}

for c in range(3):
    mask = train_cluster == c

    if c == 0:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8 if mask.sum() > 1000 else 6,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_depth=8 if mask.sum() > 1000 else 6,
            learning_rate=0.08,
            max_iter=300,
            random_state=42,
        )

    model.fit(
        X_train.loc[mask],
        y_train_raw.loc[mask],
    )

    experts_mixed[c] = model


pred_moe_mixed = np.zeros(len(X_test))

for c in range(3):
    mask = test_cluster == c

    if mask.sum() > 0:
        pred_moe_mixed[mask] = experts_mixed[c].predict(
            X_test.loc[mask]
        )


# ============================================================
# Metrics
# ============================================================
def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
    }


model_metrics = {
    "Single HGBR": metrics(
        y_test_raw,
        pred_single_hgbr,
    ),
    "Single RF": metrics(
        y_test_raw,
        pred_single_rf,
    ),
    "HGBR MoE": metrics(
        y_test_raw,
        pred_moe_hgbr,
    ),
    "RF MoE": metrics(
        y_test_raw,
        pred_moe_rf,
    ),
    "Mixed MoE": metrics(
        y_test_raw,
        pred_moe_mixed,
    ),
}

best_model_name = max(
    model_metrics,
    key=lambda name: model_metrics[name]["r2"],
)


# ============================================================
# Cluster descriptions
# ============================================================
cluster_profiles = {}

cluster_source = train_df.copy()
cluster_source["cluster"] = train_cluster

for c in range(3):
    g = cluster_source[
        cluster_source["cluster"] == c
    ].copy()

    video_rate = float(
        pd.to_numeric(
            g["video"],
            errors="coerce",
        ).fillna(0).mean()
    )

    carousel_rate = float(
        pd.to_numeric(
            g["carousel"],
            errors="coerce",
        ).fillna(0).mean()
    )

    image_rate = max(
        0.0,
        1.0 - video_rate - carousel_rate,
    )

    dominant = max(
        {
            "video": video_rate,
            "carousel": carousel_rate,
            "image": image_rate,
        },
        key=lambda x: {
            "video": video_rate,
            "carousel": carousel_rate,
            "image": image_rate,
        }[x],
    )

    segment = label_map[c]

    median_engagement = float(
        g["Engagement"].median()
    )

    mean_engagement = float(
        g["Engagement"].mean()
    )

    median_followers = float(
        g["followers"].median()
    )

    median_history = float(
        g["user_median_engagement"].median()
    )

    median_prior_posts = float(
        g["user_post_count"].median()
    )

    median_hashtags = float(
        g["number_hashtags"].median()
    )

    median_caption = float(
        g["length_caption"].median()
    )

    type_name = (
        f"{segment}-Engagement Segment / "
        f"{dominant.title()}-Leaning"
    )

    description = (
        f"This is the {segment.lower()} engagement segment based on the "
        f"cluster's mean raw engagement in the training data. "
        f"Its dominant content tendency is {dominant}. "
        f"Typical values in this cluster are about "
        f"{median_followers:,.0f} followers, "
        f"{median_history:,.0f} prior historical median engagement, "
        f"{median_prior_posts:,.0f} prior posts, "
        f"{median_hashtags:.1f} hashtags, and "
        f"{median_caption:,.0f} caption characters."
    )

    cluster_profiles[c] = {
        "label": segment,
        "type_name": type_name,
        "description": description,
        "mean_engagement": mean_engagement,
        "median_engagement": median_engagement,
        "median_followers": median_followers,
        "median_history": median_history,
        "median_prior_posts": median_prior_posts,
        "video_rate": video_rate,
        "carousel_rate": carousel_rate,
        "image_rate": image_rate,
        "rows": int(len(g)),
    }


# ============================================================
# Save everything Streamlit needs
# ============================================================
artifacts = {
    "scaler": scaler,
    "ohe": ohe,
    "ohe_cols": ohe_cols,
    "catcol": catcol,
    "numeric_features": numeric_features,
    "x_columns": list(X_train.columns),

    "single_hgbr": single_hgbr,
    "single_rf": single_rf,

    "kmeans": kmeans,
    "experts_hgbr": experts_hgbr,
    "experts_rf": experts_rf,
    "experts_mixed": experts_mixed,

    "label_map": label_map,
    "cluster_profiles": cluster_profiles,

    "default_user_median_engagement": default_user_median_engagement,
    "default_user_post_count": default_user_post_count,

    "model_metrics": model_metrics,
    "best_model_name": best_model_name,
    "silhouette": float(silhouette),

    "target": "raw Engagement",
}

joblib.dump(
    artifacts,
    ARTIFACT_PATH,
)

print("\nModel metrics:")
for name, m in model_metrics.items():
    print(
        f"{name:12s} "
        f"R2={m['r2']:.4f} "
        f"RMSE={m['rmse']:.0f} "
        f"MAE={m['mae']:.0f}"
    )

print("\nBest model:", best_model_name)
print("Silhouette:", round(silhouette, 3))

print("\nCluster labels:")
for c in range(3):
    print(
        f"Cluster {c}: "
        f"{cluster_profiles[c]['type_name']}"
    )

print(
    f"\nSaved {ARTIFACT_PATH}"
)
