import pandas as pd
import numpy as np
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

# Step 9: Model evaluation (SVM)
y_pred_svm = svm_grid_search.predict(X_test)
y_pred_proba_svm = svm_grid_search.predict_proba(X_test)[:, 1]  # Predict probabilities for ROC-AUC
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_proba_svm)

print("SVM Model Evaluation:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_svm)
print("ROC AUC Score:", roc_auc_svm)

# Step 10: Model training and evaluation (Random Forest Classifier)
rf_params = {'n_estimators': [100, 200, 300]}
rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf_model, rf_params, cv=cv)
rf_grid_search.fit(X_train, y_train)

# Step 11: Model evaluation (Random Forest Classifier)
y_pred_rf = rf_grid_search.predict(X_test)
y_pred_proba_rf = rf_grid_search.predict_proba(X_test)[:, 1]  # Predict probabilities for ROC-AUC
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print("\nRandom Forest Classifier Model Evaluation:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("ROC AUC Score:", roc_auc_rf)

# Step 12: Manual testing function (SVM)
def predict_fake_news_svm(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    text_reduced = svd.transform(text_vectorized)
    prediction = svm_grid_search.predict(text_reduced)
    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Not Fake News"

# Example of manual testing (SVM)
news_article = "According to a new study, eating chocolate every day can improve your memory."
print("\nPrediction for the news article (SVM):")
print(predict_fake_news_svm(news_article))

# Step 13: Manual testing function (Random Forest Classifier)
def predict_fake_news_rf(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    text_reduced = svd.transform(text_vectorized)
    prediction = rf_grid_search.predict(text_reduced)
    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Not Fake News"

# Example of manual testing (Random Forest Classifier)
print("\nPrediction for the news article (Random Forest Classifier):")
print(predict_fake_news_rf(news_article))

# Evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

# SVM Model Metrics
svm_metrics = [accuracy_svm, precision_svm, recall_svm, f1_svm, roc_auc_svm]

# Random Forest Classifier Model Metrics
rf_metrics = [accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf]

# Combine metrics for plotting
data = pd.DataFrame({
    'Metric': metrics * 2,
    'Score': svm_metrics + rf_metrics,
    'Model': ['SVM'] * 5 + ['Random Forest'] * 5
})

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=data)
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
