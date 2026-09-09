# 🎓 Exam Score Predictor

A machine learning project that predicts a student's **exam score** based on academic and lifestyle-related factors using a **tuned XGBoost Regression model**.

The project includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, evaluation, model serialization, and an interactive **Streamlit web application** for making predictions.

---

## 🚀 Live Application

👉 **Streamlit App:**  
_Add your deployed Streamlit URL here_

---

## 📌 Project Overview

The goal of this project is to predict a student's exam score using factors such as:

- Study Hours
- Class Attendance
- Sleep Hours
- Sleep Quality
- Study Method
- Facility Rating

An **XGBoost Regressor** was trained and optimized using hyperparameter tuning to obtain the best-performing model.

The trained model and Label Encoders are saved and used by the Streamlit application to generate predictions for new student data.

---

## 🧠 Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Label Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
XGBoost Regression
   ↓
Hyperparameter Tuning
   ↓
Best XGBoost Model
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Web Application
   ↓
Exam Score Prediction
