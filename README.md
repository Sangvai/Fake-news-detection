# Fake-news-detection
A machine learning project for classifying news articles as real or fake using text preprocessing, TF-IDF vectorization, and models like Logistic Regression and Passive Aggressive Classifier. Built using Python and scikit-learn in a Jupyter Notebook.
# Overview
This project implements a fake news detection system using DistilBERT, a distilled version of BERT that is faster and lighter while maintaining good performance. The model is trained on a dataset containing both real and fake news articles to classify news content accurately.

# Dataset
Dataset link:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset consists of:
23,481 fake news articles
21,417 real news articles

**Features include:**
Title
Text content
Subject
Date

**Model Architecture**
Base Model: distilbert-base-uncased
Classification Head: Added for binary classification (fake/real)
Parameters: 67 million
**Training**
Epochs: 3
Batch Size: 16
Max Sequence Length: 512 tokens
Learning Rate: 2e-5
Optimizer: AdamW with linear warmup

**Performance**
Test Accuracy: 100%
Precision: 1.00 for both classes
Recall: 1.00 for both classes
F1-Score: 1.00 for both classes

**Requirements**
Python 3.x
PyTorch
Transformers library
CUDA (for GPU acceleration)

**Usage**
Load the dataset
Preprocess the text data
Initialize the DistilBERT model
Train the model
Evaluate on test data

**Results**
The model achieved perfect classification on the test set, demonstrating strong ability to distinguish between real and fake news articles in this dataset.

Note: The 100% accuracy suggests potential data leakage or an overly simple test set - further validation with different datasets is recommended for production use.


