import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from nltk.corpus import stopwords
import nltk
import pandas as pd
from pathlib import Path

nltk.download('stopwords')

# -----------------------
# Project-relative paths
# -----------------------
project_root = Path(__file__).parent
model_path = project_root / 'Stacked Bidirectional LSTM.h5'
tokenizer_path = project_root / 'tokenizer.pickle'

# -----------------------
# Load model and tokenizer
# -----------------------
model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# -----------------------
# Label map
# -----------------------
label_map = {
    0: 'sport',
    1: 'business',
    2: 'politics',
    3: 'tech',
    4: 'entertainment'
}

# -----------------------
# Preprocessing
# -----------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="BBC News Classifier", layout="centered")

# App Title
st.markdown("<h1 style='text-align: center;'>📰 BBC News Category Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the category of a BBC news article using a Stacked Bidirectional LSTM model.</p>", unsafe_allow_html=True)
st.markdown("---")

# User Input
user_input = st.text_area(
    "✏️ Paste your BBC news article text below:",
    placeholder="E.g. The football team secured a last-minute victory in the championship...",
    height=300
)

# Classification
if st.button("🔍 Classify Article"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        cleaned = preprocess(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=250)

        prediction = model.predict(padded)
        pred_class = np.argmax(prediction)
        pred_label = label_map[pred_class]

        st.markdown("---")
        st.subheader("✅ Prediction Result")
        st.metric(label="Predicted Category", value=pred_label.upper())

        # Show prediction probabilities
        st.markdown("### 🔢 Category Probabilities")
        probs_df = pd.DataFrame({
            'Category': list(label_map.values()),
            'Probability': prediction[0]
        }).sort_values(by='Probability', ascending=False)

        st.bar_chart(probs_df.set_index('Category'))

        # Optional: show numerical values below chart
        with st.expander("Show raw probabilities"):
            st.dataframe(probs_df.reset_index(drop=True).style.format({'Probability': '{:.4f}'}))

# Footer
st.markdown("---")
st.markdown("<small style='color:gray;'>Developed with 💻 using TensorFlow & Streamlit</small>", unsafe_allow_html=True)
