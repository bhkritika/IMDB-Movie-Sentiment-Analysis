# IMDB Movie Sentiment Analysis

This project aims to perform sentiment analysis on IMDB movie reviews dataset. The sentiment analysis is conducted to classify the reviews as positive or negative. The dataset consists of movie reviews labeled with sentiment, and we will explore various machine learning models to predict the sentiment of a review. Among the models, Logistic Regression has shown the best performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Word Cloud Visualization](#Word-Cloud-Visualization)
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Project Overview
The project consists of the following main components:

- Data Collection: The dataset is collected from IMDB and consists of movie reviews labeled with sentiment.
- Preprocessing: Data preprocessing involves text normalization, which includes converting text to lowercase, removing punctuation, and tokenization.
- Word Cloud Visualization: Visualization of word clouds for positive and negative sentiment reviews.
- Model Training: Training machine learning models including Logistic Regression, Decision Tree, and Random Forest.
- Model Evaluation: Evaluating the trained models using accuracy, precision, recall, F1-score, and ROC AUC.
- Model Comparison: Comparing the performance of different models and selecting the best performing model.

## Dataset
The dataset contains the following columns:

- **review**: Movie review text
- **sentiment**: Sentiment label ('positive' or 'negative')
The dataset is preprocessed to remove any missing values.

## Word Cloud Visualization
Word clouds are generated for positive and negative sentiment reviews to visualize the most frequent words.
![word](https://github.com/bhkritika/IMDB-Movie-Sentiment-Analysis/assets/141895513/6951db4e-df0c-4ae9-8f98-84abc1298b82)

## Model Training
The machine learning models are trained using the TF-IDF vectorization of the movie reviews. The following models are trained:

- Logistic Regression
- Decision Tree
- Random Forest

## Model Performance

The performance of each model is evaluated using the following metrics:

1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. ROC AUC

## Model Comparison

A comparison of the performance metrics of different models is illustrated below:
![compa](https://github.com/bhkritika/IMDB-Movie-Sentiment-Analysis/assets/141895513/f93611c7-8c6d-4d5f-b62e-6d396d459f98)

## Conclusion

After comparing the performance of the trained models, it is observed that the Logistic Regression model performs the best among the models considered.

## Future Work

1. Explore advanced techniques for sentiment analysis such as LSTM and BERT.
2. Incorporate deep learning models for better performance.
3. Enhance the dataset by collecting more reviews and labels.

## References

1. [IMDB](https://www.imdb.com/)
2. [NLTK](https://www.nltk.org/)
3. [Scikit-learn](https://scikit-learn.org/)

