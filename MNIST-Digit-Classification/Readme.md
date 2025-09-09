# 🔢 MNIST Handwritten Digit Classification

### A Deep Learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.  
### The model is trained on the MNIST dataset and also supports predicting digits from custom handwritten images by providing the image path.


## 📌 Project Overview

Built a CNN model for recognizing handwritten digits.  

Utilized the MNIST dataset (60,000 training and 10,000 testing images).  

Achieved high accuracy(96.7%) through effective training and evaluation.  

Added functionality to predict on user-provided handwritten images.  

## 📂 Dataset

Source: MNIST Handwritten Digit Dataset  

Training Samples: 60,000  

Test Samples: 10,000  

Image Size: 28x28 grayscale  

## 🛠️ Technologies Used 

Python  

TensorFlow / Keras → Model building & training  

NumPy, Pandas → Data handling  

Matplotlib → Visualization  

## 🤖 Model Architecture

Input Layer: 28×28 grayscale image  

Convolution Layers: For feature extraction  

Pooling Layers: For dimensionality reduction  

Dense Layers: For classification  

Output Layer: 10 neurons (digits 0–9) with softmax activation  

## 📊 Results

Training Accuracy: ~95%

Test Accuracy: ~96.71%

Robust model performance on unseen handwritten digit images.

## 🚀 Features

✔️ End-to-End CNN model for digit classification  
✔️ Prediction on custom handwritten images (by providing image path)  
✔️ Clean modular code for easy understanding and extension