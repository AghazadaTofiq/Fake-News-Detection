# Fake-News-Detection
Fake News Detection is a Python project aimed at classifying news articles as fake or real using machine learning algorithms. This project utilizes natural language processing techniques and supervised learning to train models capable of distinguishing between genuine news articles and fabricated ones.

## Overview
Fake news has become a significant issue in today's digital age, influencing public opinion and potentially causing harm. This project addresses the problem by developing machine learning models capable of automatically detecting fake news articles. Two classification algorithms, Support Vector Machines (SVM) and Random Forest Classifier, are employed for this purpose.

## Installation
To run the project, follow these steps:

Clone the repository:
git clone https://github.com/AghazadaTofiq/Fake-News-Detection.git

Install the required dependencies:
pip install -r requirements.txt

## Usage
After installing the dependencies, you can execute the main script:
python FakeNewsDetection.py

This script performs data loading, preprocessing, feature extraction, model training, evaluation, and manual testing.

## Dataset
The project utilizes two datasets: Fake.csv and True.csv, taken from https://www.kaggle.com/code/therealsampat/fake-news-detection/input, containing fake and genuine news articles, respectively. These datasets are loaded into Pandas DataFrames for processing.

## Preprocessing
Text data undergoes preprocessing to remove noise and standardize the format. Preprocessing steps include converting text to lowercase, removing URLs, punctuation, special characters, digits, and HTML tags.

## Feature Extraction
Feature extraction is performed using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to convert text data into numerical features. Additionally, dimensionality reduction is applied using Truncated Singular Value Decomposition (SVD).

## Model Training and Evaluation
The project trains two classification models: Support Vector Machines (SVM) and Random Forest Classifier. Grid search and cross-validation are used to optimize model hyperparameters. Evaluation metrics such as accuracy, precision, recall, F1 score, and ROC AUC are computed to assess model performance.

## Manual Testing
Manual testing functions allow users to input their own news articles and receive predictions from the trained models regarding their authenticity.

## Evaluation Metrics
The project evaluates model performance using various metrics, including accuracy, precision, recall, F1 score, and ROC AUC. These metrics provide insights into the models' ability to classify fake and real news articles accurately.

## Results
The project presents evaluation results through visualizations, including bar plots of evaluation metrics and Precision-Recall curves.

## Future Improvements
Future improvements to the project may include experimenting with different machine learning algorithms, incorporating more advanced text preprocessing techniques, and exploring ensemble methods for improved classification performance.

## Contributing
Contributions to the project are welcome. If you encounter any issues or have suggestions for improvement, please feel free to submit a pull request or open an issue on GitHub.
