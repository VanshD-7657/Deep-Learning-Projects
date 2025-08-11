# 🩺 Breast Cancer Detection using Deep Learning

A deep learning project that predicts whether a tumor is **malignant** or **benign** using the **Breast Cancer Wisconsin Diagnostic Dataset** from `scikit-learn`.

## 📌 Overview
Breast cancer is one of the most common cancers in women worldwide. Early detection can significantly improve the chances of successful treatment.  
In this project, we build a deep learning model using **TensorFlow/Keras** to classify tumors based on various diagnostic measurements.

---

## 📂 Dataset
- **Source:** [Scikit-learn Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
- **Samples:** 569  
- **Features:** 30 numerical features (tumor measurements)  
- **Target:**  
  - `0` → Malignant  
  - `1` → Benign  

---

## 🛠️ Technologies Used
- **Python**
- **Pandas, NumPy** → Data manipulation
- **Matplotlib, Seaborn** → Data visualization
- **Scikit-learn** → Dataset loading & preprocessing
- **TensorFlow / Keras** → Deep learning model creation

---

## 📊 Exploratory Data Analysis (EDA)
Performed:
- Data distribution analysis
- Count plot of target classes

---

## 🤖 Model Architecture
- **Input Layer:** 30 neurons (one per feature)
- **Hidden Layers:** Dense layers with ReLU activation
- **Output Layer:** 2 neuron with sigmoid activation
- **Loss Function:** sparse_categorical_crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

## 🚀 How to Run the Project

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
