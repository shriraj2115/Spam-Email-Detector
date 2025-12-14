# =============================
# Spam Email Detection - model.py
# =============================

import numpy as np
import pandas as pd
import string
import pickle
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =============================
# Load Dataset
# =============================
data = pd.read_csv("data/Emails.csv")

# Keep only required columns
data = data[['text', 'label']]

# =============================
# Balance Dataset
# =============================
ham = data[data['label'] == 'ham']
spam = data[data['label'] == 'spam']

ham_balanced = ham.sample(n=len(spam), random_state=42)
balanced_data = pd.concat([ham_balanced, spam]).reset_index(drop=True)

# =============================
# Text Cleaning
# =============================
balanced_data['text'] = balanced_data['text'].str.replace(
    'Subject', '', regex=False
)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

balanced_data['text'] = balanced_data['text'].apply(remove_punctuation)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join(
        word.lower() for word in text.split()
        if word.lower() not in stop_words
    )

balanced_data['text'] = balanced_data['text'].apply(remove_stopwords)

# =============================
# Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    balanced_data['text'],
    balanced_data['label'],
    test_size=0.2,
    random_state=42
)

y_train = (y_train == 'spam').astype(int)
y_test = (y_test == 'spam').astype(int)

# =============================
# Tokenization & Padding
# =============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

MAX_LEN = 100

X_train_pad = pad_sequences(
    X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post'
)
X_test_pad = pad_sequences(
    X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post'
)

# =============================
# Build LSTM Model
# =============================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=32,
        input_length=MAX_LEN
    ),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# Train Model
# =============================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.5
)

model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop, reduce_lr]
)

# =============================
# Evaluate Model
# =============================
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# =============================
# Save Model & Tokenizer
# =============================
model.save("spam_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved successfully!")
