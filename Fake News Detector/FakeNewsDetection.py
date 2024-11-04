import pandas as pd
import numpy as np
import streamlit as st
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and label the dataset
data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")
data_fake["class"] = 0
data_true["class"] = 1
data = pd.concat([data_fake, data_true], ignore_index=True)

# Step 2: Data cleaning and preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(preprocess_text)

# Step 3: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['text'])
y = data['class']

# Step 4: Reduce dimensionality using TruncatedSVD
svd = TruncatedSVD(n_components=100)
X = svd.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model training with GridSearchCV
def train_model(model, params, X_train, y_train):
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, params, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search

svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_model = train_model(SVC(kernel='linear', probability=True), svm_params, X_train, y_train)

rf_params = {'n_estimators': [100, 200, 300]}
rf_model = train_model(RandomForestClassifier(random_state=42), rf_params, X_train, y_train)

# Step 7: Model evaluation function
model_metrics = []

def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }
    model_metrics.append(metrics)
    return metrics

# Evaluate SVM and Random Forest models
evaluate_model(svm_model, "SVM")
evaluate_model(rf_model, "Random Forest Classifier")

# Step 8: Plotting metrics
metrics_df = pd.DataFrame(model_metrics).melt(id_vars='Model', var_name='Metric', value_name='Score')

def plot_metrics(metrics_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df)
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(loc='upper right')
    plt.show()

plot_metrics(metrics_df)

# Step 9: Manual testing function
def predict_fake_news(text, model):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    text_reduced = svd.transform(text_vectorized)
    prediction = model.predict(text_reduced)
    return "Fake News" if prediction[0] == 0 else "Not Fake News"

# Example usage
news_article = "According to a new study, eating chocolate every day can improve your memory."
print("\nPrediction for the news article (SVM):", predict_fake_news(news_article, svm_model))
print("\nPrediction for the news article (Random Forest Classifier):", predict_fake_news(news_article, rf_model))

# Step 10: Streamlit dashboard
st.title("Model Evaluation Metrics Comparison")
st.subheader("Model Metrics Comparison")
st.write("Comparison of Accuracy, Precision, Recall, F1 Score, and ROC AUC between SVM and Random Forest models.")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, ax=ax)
ax.set_title('Model Evaluation Metrics')
ax.set_ylim(0, 1)
st.pyplot(fig)

# Precision-Recall curve for SVM
st.subheader("Precision-Recall Curve (SVM)")
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(svm_model, X_test, y_test, ax=ax)
ax.set_title('Precision-Recall Curve (SVM)')
st.pyplot(fig)
