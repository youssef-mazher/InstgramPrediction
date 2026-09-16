
import joblib
import holidays
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Post Performance Predictor",
    layout="centered",
)

# ============================================================
# Load models
# ============================================================
@st.cache_resource
def load_artifacts():
    return joblib.load(
        "model_artifacts.joblib"
    )


art = load_artifacts()


def bucket(h):
    if h in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    elif h in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    return "normal"


# ============================================================
# Header
# ============================================================
st.title(
    "📊 Post Performance Predictor"
)

st.caption(
    "Predict raw engagement for a planned post. "
    "Enter the planned-post details, including the account's historical median "
    "engagement and number of prior posts."
)


# ============================================================
# Account history
# ============================================================
st.subheader(
    "Account history"
)

hist1, hist2 = st.columns(2)

with hist1:
    user_median_engagement = st.number_input(
        "Historical median engagement",
        min_value=0.0,
        value=500.0,
        step=1.0,
        help=(
            "Median engagement from this account's previous posts. "
            "Use the median rather than the mean so unusually viral posts "
            "do not dominate the baseline."
        ),
    )

with hist2:
    user_post_count = st.number_input(
        "Number of prior posts",
        min_value=0,
        value=10,
        step=1,
        help=(
            "How many posts from this account existed before the planned post."
        ),
    )


# ============================================================
# Planned post
# ============================================================
st.subheader(
    "Planned post"
)

col1, col2 = st.columns(2)

with col1:
    followers = st.number_input(
        "Current followers",
        min_value=1,
        value=10000,
    )

    number_hashtags = st.number_input(
        "Number of hashtags",
        min_value=0,
        max_value=30,
        value=5,
    )

    length_caption = st.number_input(
        "Caption length (characters)",
        min_value=0,
        max_value=10000,
        value=120,
    )

    post_date = st.date_input(
        "Planned post date"
    )


with col2:
    post_hour = st.slider(
        "Planned post hour",
        min_value=0,
        max_value=23,
        value=12,
    )

    media_choice = st.selectbox(
        "Media type",
        [
            "Image",
            "Carousel",
            "Video / Reel",
        ],
    )

    if media_choice == "Carousel":
        post_images = st.number_input(
            "Number of images in carousel",
            min_value=2,
            max_value=30,
            value=5,
        )
    elif media_choice == "Image":
        post_images = 1
        st.caption(
            "Post images = 1"
        )
    else:
        post_images = 0
        st.caption(
            "Post images = 0"
        )


weekday = post_date.strftime(
    "%A"
)

st.caption(
    f"Weekday is detected automatically: **{weekday}**"
)


# ============================================================
# Prediction
# ============================================================
predict_btn = st.button(
    "Predict performance",
    type="primary",
)

if predict_btn:

    row = {
        "followers": followers,
        "post_images": post_images,

        # These two are automatic.
        "user_median_engagement":
            user_median_engagement,

        "length_caption":
            length_caption,

        "user_post_count":
            user_post_count,

        "number_hashtags":
            number_hashtags,

        "video":
            1
            if media_choice == "Video / Reel"
            else 0,

        "carousel":
            1
            if media_choice == "Carousel"
            else 0,

        "publication_weekday":
            weekday,

        "caption_length_bucket":
            pd.cut(
                [length_caption],
                bins=[
                    -1,
                    50,
                    150,
                    300,
                    10000,
                ],
                labels=[
                    "short",
                    "medium",
                    "long",
                    "very_long",
                ],
            )[0],

        "hashtag_bucket":
            pd.cut(
                [number_hashtags],
                bins=[
                    -1,
                    0,
                    5,
                    15,
                    30,
                ],
                labels=[
                    "none",
                    "low",
                    "medium",
                    "high",
                ],
            )[0],

        "reach_time_bucket":
            bucket(
                post_hour
            ),

        "is_weekend":
            1
            if weekday
            in [
                "Saturday",
                "Sunday",
            ]
            else 0,

        "is_holiday":
            post_date
            in holidays.US(
                years=[
                    post_date.year
                ]
            ),
    }

    df_in = pd.DataFrame(
        [row]
    )


    # ========================================================
    # Same numeric log1p transforms
    # ========================================================
    for col in art[
        "numeric_features"
    ]:
        df_in[
            "log" + col
        ] = np.log1p(
            pd.to_numeric(
                df_in[col],
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
        )

        df_in.drop(
            columns=col,
            inplace=True,
        )


    # ========================================================
    # Same train-fitted OHE
    # ========================================================
    ohe_out = pd.DataFrame(
        art["ohe"].transform(
            df_in[
                art["catcol"]
            ]
        ),
        columns=art[
            "ohe_cols"
        ],
        index=df_in.index,
    )

    df_in = pd.concat(
        [
            df_in.drop(
                columns=art[
                    "catcol"
                ]
            ),
            ohe_out,
        ],
        axis=1,
    ).astype(float)

    df_in = df_in.reindex(
        columns=art[
            "x_columns"
        ],
        fill_value=0,
    )

    X_scaled = art[
        "scaler"
    ].transform(
        df_in
    )


    # ========================================================
    # Router
    # ========================================================
    cluster = int(
        art[
            "kmeans"
        ].predict(
            X_scaled
        )[0]
    )


    # ========================================================
    # All notebook predictions
    # ========================================================
    pred_single_hgbr = float(
        art[
            "single_hgbr"
        ].predict(
            X_scaled
        )[0]
    )

    pred_single_rf = float(
        art[
            "single_rf"
        ].predict(
            X_scaled
        )[0]
    )

    pred_moe_hgbr = float(
        art[
            "experts_hgbr"
        ][cluster].predict(
            df_in
        )[0]
    )

    pred_moe_rf = float(
        art[
            "experts_rf"
        ][cluster].predict(
            df_in
        )[0]
    )

    pred_moe_mixed = float(
        art[
            "experts_mixed"
        ][cluster].predict(
            df_in
        )[0]
    )

    predictions = {
        "Single HGBR":
            max(
                0.0,
                pred_single_hgbr,
            ),

        "Single RF":
            max(
                0.0,
                pred_single_rf,
            ),

        "HGBR MoE":
            max(
                0.0,
                pred_moe_hgbr,
            ),

        "RF MoE":
            max(
                0.0,
                pred_moe_rf,
            ),

        "Mixed MoE":
            max(
                0.0,
                pred_moe_mixed,
            ),
    }


    # ========================================================
    # Cluster profile
    # ========================================================
    profile = art[
        "cluster_profiles"
    ][cluster]

    cluster_median = float(
        profile["median_engagement"]
    )

    # ========================================================
    # Main three values
    #
    # The visible benchmark is the routed cluster's historical
    # median, not a manually entered account-history value.
    # ========================================================
    st.subheader(
        "Prediction"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mixed MoE",
        f"{predictions['Mixed MoE']:,.0f}",
    )

    c2.metric(
        "Single Random Forest",
        f"{predictions['Single RF']:,.0f}",
    )

    c3.metric(
        "Your historical median",
        f"{user_median_engagement:,.0f}",
    )

    c4.metric(
        "Cluster median",
        f"{cluster_median:,.0f}",
    )


    # ========================================================
    # Cluster description
    # ========================================================
    st.subheader(
        f"Cluster {cluster}: "
        f"{profile['type_name']}"
    )

    st.write(
        profile[
            "description"
        ]
    )

    st.caption(
        f"Training rows in this cluster: "
        f"{profile['rows']:,}. "
        f"Cluster median raw engagement: "
        f"{profile['median_engagement']:,.0f}."
    )


    # ========================================================
    # Final prediction
    # ========================================================
    best_model_name = art[
        "best_model_name"
    ]

    final_pred = predictions[
        best_model_name
    ]

    st.subheader(
        "Final result"
    )

    st.metric(
        f"Expected engagement — {best_model_name}",
        f"{final_pred:,.0f}",
    )


    # First compare the final prediction with the user's own
    # historical median engagement.
    if user_median_engagement > 0:

        ratio = (
            final_pred
            / user_median_engagement
        )

        difference_pct = (
            ratio - 1
        ) * 100

        # Product interpretation rule, not another ML model.
        NEAR_BAND = 20.0

        if abs(
            difference_pct
        ) <= NEAR_BAND:

            st.info(
                f"The prediction is **near this account's usual performance**: "
                f"{difference_pct:+.1f}% versus the entered historical median "
                f"({ratio:.2f}× the median)."
            )

        elif difference_pct > 0:

            st.success(
                f"The prediction is **above this account's usual performance** "
                f"by about {difference_pct:.1f}% "
                f"({ratio:.2f}× the entered historical median)."
            )

        else:

            st.warning(
                f"The prediction is **below this account's usual performance** "
                f"by about {abs(difference_pct):.1f}% "
                f"({ratio:.2f}× the entered historical median)."
            )

    # Keep the cluster median as a second benchmark.
    if cluster_median > 0:
        cluster_ratio = final_pred / cluster_median
        cluster_difference_pct = (cluster_ratio - 1) * 100

        st.caption(
            f"Against the routed cluster median: "
            f"{cluster_difference_pct:+.1f}% "
            f"({cluster_ratio:.2f}× the cluster median)."
        )


    # ========================================================
    # Extra model details - hidden from normal marketer view
    # ========================================================
    with st.expander(
        "Compare all model predictions"
    ):
        comparison = pd.DataFrame(
            {
                "Model":
                    list(
                        predictions.keys()
                    ),
                "Predicted engagement":
                    list(
                        predictions.values()
                    ),
                "Test R²": [
                    art[
                        "model_metrics"
                    ][name]["r2"]
                    for name
                    in predictions.keys()
                ],
            }
        )

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )


    with st.expander(
        "Model evaluation details"
    ):
        st.write(
            "The app automatically uses the model "
            "with the highest saved test R² as the final prediction."
        )

        st.write(
            f"Selected final model: **{best_model_name}**"
        )

        st.write(
            f"K-Means silhouette: "
            f"{art['silhouette']:.3f}"
        )

        st.json(
            art[
                "model_metrics"
            ]
        )


st.divider()

st.caption(
    "Target: raw Engagement. "
    "The marketer manually enters the planned-post details, the account's "
    "historical median engagement, and the number of prior posts. "
    "The routed cluster's historical median is still displayed as a separate "
    "benchmark for comparison."
)
