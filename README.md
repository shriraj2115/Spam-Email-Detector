Spam Email Detection Using TensorFlow in Python

Automatically detecting spam emails helps reduce inbox clutter and protects users from phishing or malicious content. This project builds a deep learning model using TensorFlow and Keras to classify emails as Spam or Ham (Not Spam).

Table of Contents

Overview

Dataset

Dataset Columns

Class Imbalance Handling

Features

Installation

Usage

Loading the Dataset

Text Preprocessing

Removing Stopwords

Removing Punctuations

Tokenization and Padding

Model Training

Evaluation

Model Architecture

Training and Evaluation

EarlyStopping and ReduceLROnPlateau

Accuracy Plot

Visualizations

WordCloud for Ham Emails

WordCloud for Spam Emails

Contributing

License

Overview

This project implements a deep learning-based spam email classifier. It automatically classifies emails into Spam or Ham (Not Spam) using natural language processing (NLP) and TensorFlow.

The workflow includes:

Data preprocessing and cleaning

Handling class imbalance

Tokenization and padding for text sequences

Training an LSTM-based model

Evaluating model performance

Visualizing results

Dataset
Dataset Columns

The dataset (Emails.csv) contains emails labeled as spam or ham, with the main columns:

text: the content of the email

label: the classification (spam or ham)

The dataset contains 5171 emails, providing a sufficient sample for training and evaluation.

Class Imbalance Handling

The dataset has significantly more ham emails than spam. To prevent bias during training, the majority class (ham) is downsampled to match the number of spam emails, creating a balanced dataset.

Features

Text preprocessing to remove noise and irrelevant content

Handling imbalanced datasets to improve model accuracy

Tokenization and padding to convert text into sequences suitable for deep learning models

LSTM-based sequential model for classification

EarlyStopping and learning rate reduction during training to optimize performance

WordCloud visualizations to explore the most frequent words in spam and ham emails

Installation

Clone the repository.

Install the required Python libraries, including TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, NLTK, WordCloud, and scikit-learn.

Usage
Loading the Dataset

Load the email dataset and inspect the structure to understand the columns and class distribution.

Text Preprocessing

Text preprocessing is essential to improve model performance:

Removing Stopwords: Removes common words that do not contribute to classification.

Removing Punctuations: Eliminates punctuation symbols to clean the text.

Optional steps include removing headers or unnecessary tokens in email content.

Tokenization and Padding

The text data is converted into numerical sequences using tokenization, and sequences are padded to a fixed length to make them compatible with the deep learning model.

Model Training

The model is an LSTM-based sequential architecture designed to capture patterns in sequences. Training includes using EarlyStopping to avoid overfitting and ReduceLROnPlateau to fine-tune learning rates.

Evaluation

After training, the model is evaluated on a test set to determine accuracy, loss, and overall performance in classifying emails.

Model Architecture

The model consists of:

Embedding Layer: Converts words into dense vector representations

LSTM Layer: Captures sequential patterns in email text

Dense Layer: Extracts relevant features

Output Layer: Sigmoid activation predicts spam or ham

This architecture allows the model to learn complex patterns in text while remaining efficient.

Training and Evaluation
EarlyStopping and ReduceLROnPlateau

EarlyStopping: Stops training when validation performance stops improving to prevent overfitting

ReduceLROnPlateau: Reduces learning rate when the validation loss plateaus, enabling finer adjustments to model weights

Accuracy Plot

Training and validation accuracy are monitored over epochs to ensure convergence and detect overfitting.

Visualizations
WordCloud for Ham Emails

WordClouds display the most frequent words in non-spam emails, helping to identify typical content.

WordCloud for Spam Emails

WordClouds for spam emails highlight common spam keywords and patterns.

Contributing

Contributions are welcome! Steps:

Fork the repository

Create a feature branch

Commit your changes

Push to your branch

Open a Pull Request

License

This project is licensed under the MIT License.
