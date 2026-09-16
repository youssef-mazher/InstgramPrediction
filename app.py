import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Instagram Engagement Predictor",
    page_icon="📊",
    layout="centered",
)

MODEL_PATH = "model.npz"
META_PATH = "model_meta.json"


@st.cache_resource
def load_model():
    meta = json.loads(open(META_PATH, "r", encoding="utf-8").read())
    model = np.load(MODEL_PATH, allow_pickle=False)
    return model, meta


model, meta = load_model()


def bucket(hour):
    if hour in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    if hour in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    return "normal"


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

    # Major US federal holidays used by the original project.
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
            nth_weekday(year, 1, 0, 3),   # MLK Day
            nth_weekday(year, 2, 0, 3),   # Presidents' Day
            last_weekday(year, 5, 0),     # Memorial Day
            nth_weekday(year, 9, 0, 1),   # Labor Day
            nth_weekday(year, 10, 0, 2),  # Columbus/Indigenous Peoples' Day
            nth_weekday(year, 11, 3, 4),  # Thanksgiving
        }
    )
    return days


def make_features(
    followers,
    post_images,
    video,
    carousel,
    length_caption,
    number_hashtags,
    publication_weekday,
    is_weekend,
    is_holiday,
    user_median_engagement,
    user_post_count,
    reach_time_bucket,
    caption_length_bucket,
    hashtag_bucket,
):
    row = {
        "followers": float(followers),
        "post_images": float(post_images),
        "video": float(video),
        "carousel": float(carousel),
        "length_caption": float(length_caption),
        "number_hashtags": float(number_hashtags),
        "is_weekend": float(is_weekend),
        "is_holiday": float(is_holiday),
        "user_median_engagement": float(user_median_engagement),
        "user_post_count": float(user_post_count),
    }

    values = {
        "publication_weekday": str(publication_weekday),
        "reach_time_bucket": str(reach_time_bucket),
        "caption_length_bucket": str(caption_length_bucket),
        "hashtag_bucket": str(hashtag_bucket),
    }

    for col, categories in meta["categorical_values"].items():
        for category in categories:
            row[f"{col}__{category}"] = float(values[col] == category)

    return np.array(
        [row[name] for name in meta["feature_names"]],
        dtype=np.float32,
    )


def predict_one(x):
    total = 0.0
    n_trees = int(meta["n_trees"])

    for i in range(n_trees):
        left = model[f"cl_{i}"]
        right = model[f"cr_{i}"]
        feature = model[f"feat_{i}"]
        threshold = model[f"thr_{i}"]
        value = model[f"val_{i}"]

        node = 0
        while left[node] != -1:
            f = int(feature[node])
            if x[f] <= threshold[node]:
                node = int(left[node])
            else:
                node = int(right[node])

        total += float(value[node])

    return max(0.0, total / n_trees)


# -----------------------------
# UI
# -----------------------------
st.title("📊 Instagram Engagement Predictor")
st.caption(
    "Predict the expected raw engagement of a planned Instagram post "
    "using a compact Random Forest model trained on the supplied datasets."
)

with st.sidebar:
    st.header("Model")
    st.write(f"Trees: **{meta['n_trees']}**")
    st.write(f"Test R²: **{meta['metrics']['r2']:.3f}**")
    st.write(f"Test MAE: **{meta['metrics']['mae']:,.0f}**")
    st.caption("The deployed app does not need scikit-learn to make predictions.")

st.subheader("1. Account history")
c1, c2 = st.columns(2)

with c1:
    user_median_engagement = st.number_input(
        "Historical median engagement",
        min_value=0.0,
        value=float(meta["default_user_median_engagement"]),
        step=1.0,
        help="Median engagement from previous posts of the account.",
    )

with c2:
    user_post_count = st.number_input(
        "Number of prior posts",
        min_value=0,
        value=int(meta["default_user_post_count"]),
        step=1,
    )

st.subheader("2. Planned post")
c1, c2 = st.columns(2)

with c1:
    followers = st.number_input(
        "Current followers",
        min_value=1,
        value=10000,
        step=100,
    )
    number_hashtags = st.number_input(
        "Number of hashtags",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
    )
    length_caption = st.number_input(
        "Caption length (characters)",
        min_value=0,
        max_value=10000,
        value=120,
        step=1,
    )
    post_date = st.date_input("Planned post date", value=date.today())

with c2:
    post_hour = st.slider("Planned post hour", 0, 23, 12)
    media_choice = st.selectbox(
        "Media type",
        ["Image", "Carousel", "Video / Reel"],
    )

    if media_choice == "Carousel":
        post_images = st.number_input(
            "Number of images in carousel",
            min_value=2,
            max_value=30,
            value=5,
            step=1,
        )
    elif media_choice == "Image":
        post_images = 1
        st.caption("Post images = 1")
    else:
        post_images = 0
        st.caption("Post images = 0")

weekday = post_date.strftime("%A")
is_weekend = int(weekday in ["Saturday", "Sunday"])
is_holiday = int(post_date in us_holidays(post_date.year))
reach_time_bucket = bucket(post_hour)

if length_caption <= 50:
    caption_length_bucket = "short"
elif length_caption <= 150:
    caption_length_bucket = "medium"
elif length_caption <= 300:
    caption_length_bucket = "long"
else:
    caption_length_bucket = "very_long"

if number_hashtags == 0:
    hashtag_bucket = "none"
elif number_hashtags <= 5:
    hashtag_bucket = "low"
elif number_hashtags <= 15:
    hashtag_bucket = "medium"
else:
    hashtag_bucket = "high"

st.info(
    f"Detected automatically: **{weekday}** · "
    f"**{reach_time_bucket}** time · "
    f"**{caption_length_bucket}** caption · "
    f"**{hashtag_bucket}** hashtags"
)

if st.button("🚀 Predict engagement", type="primary", use_container_width=True):
    x = make_features(
        followers=followers,
        post_images=post_images,
        video=int(media_choice == "Video / Reel"),
        carousel=int(media_choice == "Carousel"),
        length_caption=length_caption,
        number_hashtags=number_hashtags,
        publication_weekday=weekday,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
        user_median_engagement=user_median_engagement,
        user_post_count=user_post_count,
        reach_time_bucket=reach_time_bucket,
        caption_length_bucket=caption_length_bucket,
        hashtag_bucket=hashtag_bucket,
    )

    prediction = predict_one(x)

    st.subheader("3. Prediction")
    a, b, c = st.columns(3)
    a.metric("Expected engagement", f"{prediction:,.0f}")
    b.metric("Your historical median", f"{user_median_engagement:,.0f}")
    c.metric("Prior posts", f"{user_post_count:,}")

    if user_median_engagement > 0:
        change = (prediction / user_median_engagement - 1) * 100
        st.write(
            f"Compared with the entered historical median: "
            f"**{change:+.1f}%**."
        )

    st.success(
        "Prediction generated successfully. "
        "This is a model estimate, not a guaranteed result."
    )

    with st.expander("Model details"):
        st.write(
            f"Random Forest trees: {meta['n_trees']}  \n"
            f"Test R²: {meta['metrics']['r2']:.3f}  \n"
            f"Test RMSE: {meta['metrics']['rmse']:,.0f}  \n"
            f"Test MAE: {meta['metrics']['mae']:,.0f}"
        )

st.divider()
st.caption("Target: raw Engagement (likes + comments [+ shares + saves where available]).")
