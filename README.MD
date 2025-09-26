# ❤️ Heart Disease Prediction App

This project predicts the likelihood of **heart disease** based on patient clinical data using **XGBoost**.  
It features a **Streamlit web app** for a user-friendly interface with real-time predictions and visualizations.

---

## **Features**
- Predicts heart disease using patient data such as Age, Sex, Cholesterol, Max Heart Rate, Chest Pain type, etc.
- Provides prediction probability and model accuracy.
- Shows **confusion matrix** and **feature importance** graphs.
- Handles imbalanced data using **SMOTE** for better predictions.

---

## **Installation**
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
3.Run the Streamlit app:
```bash
streamlit run heart_disease_xgb_streamlit.py
```
Dataset
```bash
heart.csv (UCI Heart Disease dataset)
```
Contains features such as:
```bash
Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease (target)
```
Model & Evaluation
```bash
Algorithm: XGBoost Classifier

Accuracy: ~85-90% (depending on train-test split)

Confusion matrix and feature importance plots are displayed in the app.
```
License
```bash
MIT License
```
---



