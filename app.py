import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & tokenizer
model = tf.keras.models.load_model("spam_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

st.set_page_config(page_title="Spam Email Detector", layout="centered")

st.title("📧 Spam Email Detector")
st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

user_input = st.text_area("Email Content", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT Spam")
