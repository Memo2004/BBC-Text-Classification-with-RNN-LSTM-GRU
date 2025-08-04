# 📰 BBC News Text Classification

Classify BBC news articles into one of five categories — **sport**, **business**, **politics**, **tech**, or **entertainment** — using deep learning models including RNN, LSTM, GRU, and a final **Stacked Bidirectional LSTM** deployed in a **Streamlit web app**.

---

## 🔍 Overview

This project involves:
- Preprocessing and vectorizing BBC news text data
- Building and comparing models: RNN, LSTM, GRU
- Selecting the best model: **Stacked Bidirectional LSTM**
- Deploying the best model in a clean and interactive **Streamlit** app

---


## 📂 Dataset

The BBC News dataset contains categorized news articles:
- ✅ 2225 total articles
- ✅ 5 categories:
  - `sport`
  - `business`
  - `politics`
  - `tech`
  - `entertainment`

---

## 🧠 Models Trained

| Model                 | Accuracy |
|----------------------|----------|
| RNN                  | ~80%     |
| LSTM                 | ~85%     |
| GRU                  | ~84%     |
| **BiLSTM (Stacked)** | **87–89%** |

> The **Stacked Bidirectional LSTM** gave the best results and was used for deployment.

---
📦bbc-news-text-classification
├── app.py                          # Streamlit web app
├── tokenizer.pickle                # Trained tokenizer
├── Stacked Bidirectional LSTM.h5   # Best trained model
├── requirements.txt
└── README.md
