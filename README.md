Spam Email Detection Using TensorFlow in Python


Automatically detecting spam emails helps reduce clutter and protect users from phishing or malicious content. This project builds a **deep learning model** using TensorFlow to classify emails as **Spam** or **Ham (Not Spam)**.  

---

## Table of Contents
- [Overview](#overview)  
- [Dataset](#dataset)  
  - [Dataset Columns](#dataset-columns)  
  - [Class Imbalance Handling](#class-imbalance-handling)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Loading the Dataset](#loading-the-dataset)  
  - [Text Preprocessing](#text-preprocessing)  
  - [Tokenization and Padding](#tokenization-and-padding)  
  - [Model Training](#model-training)  
  - [Evaluation](#evaluation)  
- [Model Architecture](#model-architecture)  
- [Training and Evaluation](#training-and-evaluation)  
  - [EarlyStopping and ReduceLROnPlateau](#earlystopping-and-reducelronplateau)  
  - [Accuracy Plot](#accuracy-plot)  
- [Visualizations](#visualizations)  
  - [WordCloud for Ham Emails](#wordcloud-for-ham-emails)  
  - [WordCloud for Spam Emails](#wordcloud-for-spam-emails)  
- [Contributing](#contributing)  
- [License](#license)  


---

## Overview
This project implements a **deep learning-based spam email classifier** using TensorFlow and Keras. It includes:
- Data preprocessing (cleaning, stopwords removal, tokenization)
- Handling class imbalance
- WordCloud visualizations
- LSTM-based sequential model for classification
- Model evaluation with accuracy metrics  

---

## Dataset
We use a labeled email dataset with the following columns:

| Column | Description |
|--------|-------------|
| `text` | Email content |
| `label` | Email label: `spam` or `ham` |

Dataset contains **5171 emails**. You can download it from: `Emails.csv`.  

The dataset is **imbalanced**, so we downsample the majority class (`ham`) to match the number of spam emails.  

---

## Features
- Email text cleaning (punctuation removal, stopwords removal)  
- Visualization using WordCloud  
- Tokenization and padding for sequence input  
- LSTM model for sequence classification  
- EarlyStopping and ReduceLROnPlateau callbacks for better training  

---

## Installation
Clone the repository and install the required libraries:

```bash
git clone <repository-url>
cd spam-email-detection
pip install -r requirements.txt

