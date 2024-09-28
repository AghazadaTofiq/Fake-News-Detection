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

# Step 1: Load the dataset
data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")

# Step 2: Label the data
data_fake["class"] = 0
data_true['class'] = 1

# Step 3: Data cleaning and preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data_fake['text'] = data_fake['text'].apply(preprocess_text)
data_true['text'] = data_true['text'].apply(preprocess_text)

# Step 4: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_fake = vectorizer.fit_transform(data_fake['text'])
X_true = vectorizer.transform(data_true['text'])

# Step 5: Reduce dimensionality using TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_fake = svd.fit_transform(X_fake)
X_true = svd.transform(X_true)

# Step 6: Concatenate reduced feature matrices
X = np.vstack([X_fake, X_true])
y = pd.concat([data_fake['class'], data_true['class']])

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model training and evaluation (SVM)
svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_model = SVC(kernel='linear', probability=True)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
svm_grid_search = GridSearchCV(svm_model, svm_params, cv=cv)
svm_grid_search.fit(X_train, y_train)

# Step 9: Model training and evaluation (Random Forest Classifier)
rf_params = {'n_estimators': [100, 200, 300]}
rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf_model, rf_params, cv=cv)
rf_grid_search.fit(X_train, y_train)

# Step 10: Model evaluation (SVM and RFS)
# Initialize an empty list to collect metrics for both models
model_metrics = []

# Function to evaluate a model and return metrics
def evaluate_model(model, model_name, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Predict probabilities for ROC-AUC
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Collect metrics in a list for plotting
    model_metrics.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })
    
    # Return metrics
    return accuracy, precision, recall, f1, roc_auc
    
# Example usage for SVM and Random Forest
evaluate_model(svm_grid_search, "SVM", X_test, y_test)
evaluate_model(rf_grid_search, "Random Forest Classifier", X_test, y_test)

# Convert the list of dictionaries into a DataFrame for plotting
metrics_df = pd.DataFrame(model_metrics).melt(id_vars='Model', var_name='Metric', value_name='Score')

# Step 11: Manual testing function
def predict_fake_news(text, model):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    text_reduced = svd.transform(text_vectorized)
    prediction = model.predict(text_reduced)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Not Fake News"

# Example of manual testing for SVM and Random Forest
news_article = "According to a new study, eating chocolate every day can improve your memory."

print("\nPrediction for the news article (SVM):")
print(predict_fake_news(news_article, svm_grid_search))

print("\nPrediction for the news article (Random Forest Classifier):")
print(predict_fake_news(news_article, rf_grid_search))

# Step 12: Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df)
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.xlabel('Metric')
plt.legend(loc='upper right')
plt.show()

# Plot Precision-Recall curve for SVM
plt.figure(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(svm_grid_search, X_test, y_test)
plt.title('Precision-Recall Curve (SVM)')
plt.show()

# Plot Precision

# Step 13: STREAMLIT
# Streamlit Title
st.title("Model Evaluation Metrics Comparison")

# Create the bar plot using Streamlit
st.subheader('Model Metrics Comparison')
st.write("This plot shows the comparison of Accuracy, Precision, Recall, F1 Score, and ROC AUC between SVM and Random Forest models.")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, ax=ax)
ax.set_title('Model Evaluation Metrics')
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_xlabel('Metric')
st.pyplot(fig)  # Render the plot in Streamlit

# Precision-Recall curve for SVM
st.subheader('Precision-Recall Curve (SVM)')
st.write("Below is the Precision-Recall curve for the SVM model.")

fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(svm_grid_search, X_test, y_test, ax=ax)
ax.set_title('Precision-Recall Curve (SVM)')
st.pyplot(fig)