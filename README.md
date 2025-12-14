# Spam Email Detection Using TensorFlow in Python


Automatically detecting spam emails helps reduce inbox clutter and protects users from phishing or malicious content. This project builds a **deep learning model** using TensorFlow and Keras to classify emails as **Spam** or **Ham (Not Spam)**.

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)

  * [Dataset Columns](#dataset-columns)
  * [Class Imbalance Handling](#class-imbalance-handling)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)

  * [Loading the Dataset](#loading-the-dataset)
  * [Text Preprocessing](#text-preprocessing)

    * [Removing Stopwords](#removing-stopwords)
    * [Removing Punctuations](#removing-punctuations)
  * [Tokenization and Padding](#tokenization-and-padding)
  * [Model Training](#model-training)
  * [Evaluation](#evaluation)
* [Model Architecture](#model-architecture)
* [Training and Evaluation](#training-and-evaluation)

  * [EarlyStopping and ReduceLROnPlateau](#earlystopping-and-reducelronplateau)
  * [Accuracy Plot](#accuracy-plot)
* [Visualizations](#visualizations)

  * [WordCloud for Ham Emails](#wordcloud-for-ham-emails)
  * [WordCloud for Spam Emails](#wordcloud-for-spam-emails)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

This project implements a **deep learning-based spam email classifier**. It automatically classifies emails into **Spam** or **Ham (Not Spam)** using natural language processing (NLP) techniques and TensorFlow.

The workflow includes:

1. Data preprocessing and cleaning
2. Handling class imbalance
3. Tokenization and sequence padding
4. Training an LSTM-based deep learning model
5. Evaluating model performance
6. Visualizing results

---

## Dataset

### Dataset Columns

The dataset (`Emails.csv`) contains emails labeled as spam or ham, with the main columns:

* **text**: the content of the email
* **label**: the classification (`spam` or `ham`)

It contains **5171 emails**, providing a sufficient sample for training and evaluation.

### Class Imbalance Handling

The dataset is imbalanced, with significantly more ham emails than spam. To prevent bias:

* The majority class (ham) is downsampled to match the number of spam emails.
* This ensures balanced training and improved model performance.

---

## Features

* Text preprocessing to remove noise and irrelevant content
* Handling imbalanced datasets to improve accuracy
* Tokenization and padding to convert text into numerical sequences
* LSTM-based sequential model for classification
* EarlyStopping and ReduceLROnPlateau callbacks for optimized training
* WordCloud visualizations to explore common words in spam and ham emails

---

## Installation

1. Clone the repository.
2. Install the required Python libraries, including:

   * TensorFlow
   * Keras
   * Pandas
   * NumPy
   * Matplotlib
   * Seaborn
   * NLTK
   * WordCloud
   * scikit-learn

---

## Usage

### Loading the Dataset

Load the email dataset and inspect the structure to understand the columns and class distribution.

### Text Preprocessing

Text preprocessing is essential to improve model performance.

#### Removing Stopwords

Stopwords (common words like "the", "is") are removed to reduce noise.

#### Removing Punctuations

Punctuation symbols are removed to clean the text.

Optional steps include removing email headers or unnecessary tokens.

### Tokenization and Padding

Text data is converted into numerical sequences, and sequences are padded to a fixed length for compatibility with the deep learning model.

### Model Training

The model is a sequential LSTM architecture designed to capture patterns in email text.

* EarlyStopping is used to prevent overfitting.
* ReduceLROnPlateau is applied to adjust learning rate during training.

### Evaluation

After training, the model is evaluated on a test set to measure accuracy and overall performance.

---

## Model Architecture

The model consists of:

* **Embedding Layer:** Converts words into vector representations
* **LSTM Layer:** Captures sequential patterns in text
* **Dense Layer:** Extracts important features
* **Output Layer:** Predicts spam or ham using sigmoid activation

This architecture allows the model to learn complex text patterns effectively.

---

## Training and Evaluation

### EarlyStopping and ReduceLROnPlateau

* **EarlyStopping:** Stops training when validation accuracy stops improving.
* **ReduceLROnPlateau:** Reduces learning rate when validation loss plateaus.

### Accuracy Plot

Training and validation accuracy are monitored over epochs to ensure convergence and detect overfitting.

---

## Visualizations

### WordCloud for Ham Emails

Visualizes the most frequent words in non-spam emails to understand typical content.

### WordCloud for Spam Emails

Highlights common keywords and patterns found in spam emails.

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to your branch (`git push origin feature-name`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**.


