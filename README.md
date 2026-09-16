# 📊 Instagram Engagement Prediction

A Machine Learning project that predicts the expected **engagement of an Instagram post** based on its planned content, audience size, posting time, caption characteristics, hashtags, and account history.

The project combines **Data Analysis, Feature Engineering, Unsupervised Learning, and Regression** to build a practical prediction system and deploy it as an interactive **Streamlit web application**.

---

## 🚀 Project Overview

Social media performance depends on many factors such as:

* 👥 Number of followers
* 📝 Caption length
* #️⃣ Number of hashtags
* 🖼️ Media type
* 🎠 Carousel posts
* 🎥 Video/Reels
* 📅 Day of the week
* 🕐 Posting time
* 📈 Previous account performance
* 📊 Number of previous posts

This project uses historical Instagram data to learn the relationship between these factors and **raw engagement**.

The final application allows a user to enter the characteristics of a planned Instagram post and receive an estimated engagement value.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Analyze historical Instagram post performance.
2. Clean and combine multiple Instagram datasets.
3. Engineer meaningful features from the raw data.
4. Identify different engagement patterns using clustering.
5. Train regression models for engagement prediction.
6. Compare model performance using standard regression metrics.
7. Build an interactive prediction application using Streamlit.
8. Deploy the model in a lightweight and reliable format.

---

## 🗂️ Project Structure

```text
instagram-engagement-predictor/
│
├── streamlit_app.py
├── train_model.py
├── model.npz
├── model_meta.json
├── requirements.txt
├── requirements-train.txt
├── README.md
│
└── data/
    ├── Grammy_IG_posts_v2.csv
    └── Instagram_Analytics.csv
```

### File Description

| File                     | Description                               |
| ------------------------ | ----------------------------------------- |
| `streamlit_app.py`       | Streamlit web application                 |
| `train_model.py`         | Model training and preprocessing pipeline |
| `model.npz`              | Lightweight serialized model parameters   |
| `model_meta.json`        | Model metadata and configuration          |
| `requirements.txt`       | Dependencies required for deployment      |
| `requirements-train.txt` | Dependencies required for training        |
| `data/`                  | Training datasets                         |
| `README.md`              | Project documentation                     |

---

## 📚 Datasets

The project combines two Instagram datasets:

### 1. Grammy Instagram Posts

Contains historical Instagram post information including engagement and post characteristics.

### 2. Instagram Analytics

Contains additional post-level analytics such as:

* Likes
* Comments
* Shares
* Saves
* Followers
* Hashtags
* Caption information
* Media type
* Posting information

The datasets are processed into a common feature space before model training.

---

# 🔧 Data Processing

The preprocessing pipeline performs several steps.

### Data Cleaning

* Convert numeric columns to numeric types.
* Handle missing values.
* Convert date/time columns into usable datetime features.
* Calculate engagement from available interaction metrics.

For example:

```text
Engagement =
Likes + Comments + Shares + Saves
```

depending on the source dataset.

---

## 🧠 Feature Engineering

Several features are created from the raw data.

### Account Features

```text
followers
user_median_engagement
user_post_count
```

These represent the size and historical performance of an account.

### Content Features

```text
caption_length_bucket
length_caption
number_hashtags
hashtag_bucket
post_images
video
carousel
```

### Time Features

```text
publication_weekday
reach_time_bucket
is_weekend
is_holiday
```

### Log Transformation

Several numeric features are transformed using:

```python
np.log1p(feature)
```

This helps reduce the influence of highly skewed variables such as follower counts and engagement.

---

# 🤖 Machine Learning

The project combines **supervised regression** and **unsupervised clustering**.

## 1. Random Forest Regression

Random Forest is used to model the relationship between post characteristics and engagement.

The model is based on multiple decision trees and combines their predictions.

---

## 2. K-Means Clustering

K-Means is used to divide posts into three engagement segments.

```text
Low
Mid
High
```

The clusters are then interpreted based on their historical engagement characteristics.

This creates an additional layer of segmentation on top of the prediction model.

---

## 3. Cluster-Based Prediction

The system first identifies the cluster associated with the planned post.

The prediction process can therefore be viewed as:

```text
Planned Post
     ↓
Feature Engineering
     ↓
Feature Transformation
     ↓
Engagement Segment
     ↓
Regression Prediction
     ↓
Expected Engagement
```

---

# 📈 Model Evaluation

The regression model is evaluated using:

### R² Score

Measures how much of the variation in engagement is explained by the model.

```text
Higher R² → better fit
```

### RMSE

Measures the typical prediction error while giving larger errors more weight.

```text
RMSE = √Mean Squared Error
```

### MAE

Measures the average absolute difference between actual and predicted engagement.

```text
MAE = Mean Absolute Error
```

The model is evaluated using a train/test split to measure its performance on unseen data.

---

# 🌐 Streamlit Application

The project includes an interactive web application built with **Streamlit**.

The user can enter:

### Account Information

* Current followers
* Historical median engagement
* Number of previous posts

### Post Information

* Caption length
* Number of hashtags
* Post date
* Posting hour
* Media type
* Number of images for carousel posts

The application automatically derives additional features such as:

* Weekday
* Weekend status
* Time bucket
* Caption length category
* Hashtag category
* Holiday status

---

# 📊 Application Output

The application provides:

### Prediction

Estimated engagement for the planned post.

### Model Comparison

Predictions from the available regression models can be compared.

### Historical Benchmark

The prediction can be compared with the account's historical median engagement.

### Cluster Benchmark

The application also displays the historical median engagement of the identified engagement segment.

---

# 🏗️ Deployment Architecture

A major design decision in this version was to avoid loading a large Python/Scikit-learn pickle artifact in the deployed application.

Instead, the trained model information is stored in a lightweight format:

```text
model.npz
```

and metadata is stored separately:

```text
model_meta.json
```

This makes the deployment:

* ⚡ Lightweight
* 📦 Small
* 🔄 Easier to reproduce
* 🌐 Suitable for Streamlit deployment
* 🛡️ Less dependent on Scikit-learn serialization compatibility

The deployment runtime only requires the packages needed by the application.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/youssef-mazher/instagramprediction.git
```

Move into the project directory:

```bash
cd instagramprediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Training the Model

Training dependencies are provided separately in:

```text
requirements-train.txt
```

Install them with:

```bash
pip install -r requirements-train.txt
```

Then run:

```bash
python train_model.py
```

The training process generates the model files required by the application.

---

# 📌 Input → Prediction Example

Example planned post:

```text
Followers:              25,000
Historical Engagement:  1,200
Previous Posts:         50
Caption Length:         180
Hashtags:               8
Posting Hour:            19
Media Type:             Video / Reel
```

The application processes these inputs and returns an estimated engagement value.

---

# 💡 Key Machine Learning Concepts Used

This project demonstrates practical use of:

* Exploratory Data Analysis
* Data Cleaning
* Feature Engineering
* Log Transformation
* One-Hot Encoding
* Feature Scaling
* Train/Test Split
* Random Forest Regression
* K-Means Clustering
* Model Evaluation
* R²
* RMSE
* MAE
* Model Serialization
* Streamlit Deployment

---

# 🛠️ Technologies

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core programming language |
| Pandas           | Data manipulation         |
| NumPy            | Numerical computation     |
| Scikit-learn     | Model training            |
| Streamlit        | Web application           |
| Matplotlib       | Data visualization        |
| Jupyter Notebook | Deve                      |
