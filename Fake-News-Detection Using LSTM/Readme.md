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

---

## 📁 Project Structure

├── FakeNewsDetector/
│ ├── model.h5 # Trained LSTM model
│ ├── tokenizer.pkl # Saved tokenizer
│ ├── fake_news.ipynb # Jupyter Notebook for model building
│ ├── README.md # Project README
│ ├── requirements.txt # Project dependencies

yaml
Copy
Edit

---

## 🧪 Dataset

- Source: [Your Dataset Source – Kaggle or custom]
- Columns:
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
| Accuracy      | 98%            | 89%                 |
| Loss          | ~0.04          | ~0.41               |

> Some overfitting observed. Mitigated using dropout, batch normalization, and L2 regularization.

---

## 🧪 Test the Model

```python
def predict_news(news_text):
    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    return "FAKE" if pred > 0.5 else "REAL"

predict_news("NASA confirms Earth will be dark for six days")
🛠️ How to Run
Clone this repo:

bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/Fake-News-Detection-LSTM.git
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook or script:

bash
Copy
Edit
jupyter notebook fake_news.ipynb
📌 Future Improvements
Use pre-trained GloVe embeddings

Add Bidirectional LSTM

Build a Streamlit web app for live demo

Explain predictions using LIME/SHAP

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.

📜 License
This project is open-source and free to use for educational purposes.