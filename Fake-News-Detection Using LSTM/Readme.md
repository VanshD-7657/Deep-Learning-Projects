# 📰 Fake News Detection using LSTM (NLP Project)

This project aims to detect whether a given news article is **real or fake** using Natural Language Processing (NLP) techniques and a deep learning model built with **LSTM (Long Short-Term Memory)** neural networks.

---

## 🔍 Problem Statement

Fake news spreads rapidly on social media, creating misinformation. This project tackles the challenge of automatically classifying news articles as **real** or **fake** using a machine learning pipeline built in Python.

---

## 🚀 Tech Stack & Tools

- 🐍 Python 3
- 🧠 TensorFlow / Keras
- 📊 Pandas, NumPy
- ✍️ NLTK (Natural Language Toolkit)
- 🧼 Sklearn (for preprocessing & metrics)
- 📚 Matplotlib, Seaborn
- 🧠 Deep Learning (LSTM with Embedding Layer)
- 🛠️ VS Code & GitHub



## 🧪 Dataset

- Source: [Training Dataset]
- Columns:
 - `title`: News title or headline
  - `text`: News article or headline
  - `label`: 0 = Real, 1 = Fake

---

## 📊 Preprocessing Steps

- Lowercasing
- Removing special characters and punctuation
- Removing stopwords
- Tokenization
- Padding sequences to uniform length
- Stemming using `PorterStemmer`

---

## 🧠 Model Architecture

- `Embedding Layer`
- `BatchNormalization`
- `LSTM Layer (64 units, dropout)`
- `Dense Layer` with sigmoid activation
- `EarlyStopping` and `L2 Regularization` to prevent overfitting

---

## 📈 Model Performance

| Metric        | Train Accuracy | Validation Accuracy |
|---------------|----------------|---------------------|
| Accuracy      | 94%            | 90%                 |
| Loss          | ~0.15          | ~0.28               |

> Some overfitting observed. Mitigated using dropout, batch normalization, and L2 regularization.


## 📌 Future Improvements

Use pre-trained GloVe embeddings

Add Bidirectional LSTM

Build a Streamlit web app for live demo

Explain predictions using LIME/SHAP

🤝 Contributing:

Pull requests are welcome. For major changes, please open an issue first.

📜 License
This project is open-source and free to use for educational purposes.