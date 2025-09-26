# heart_disease_xgb_streamlit.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("❤️ Heart Disease Prediction App")
st.markdown(
    "This app predicts the likelihood of **heart disease** based on patient clinical data. "
    "Adjust the parameters in the sidebar and see the prediction in real-time."
)

# ===============================
# Load Dataset & Prepare Model
# ===============================
@st.cache_data
def load_model():
    df = pd.read_csv("heart.csv")  # make sure your CSV matches this name
    # Columns in the dataset
    # ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    #  'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']

    target_col = "HeartDisease"
    X = pd.get_dummies(df.drop(target_col, axis=1), drop_first=True)
    y = df[target_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, X, y, X_test, y_test

model, scaler, X_columns, y, X_test, y_test = load_model()

# ===============================
# Sidebar: User Input
# ===============================
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Adjust the features below to predict heart disease.")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', ('typical', 'atypical', 'non-anginal', 'asymptomatic'))
    resting_bp = st.sidebar.slider('Resting BP', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('Yes', 'No'))
    restecg = st.sidebar.selectbox('Resting ECG', ('normal', 'ST-T abnormality', 'left ventricular hypertrophy'))
    max_hr = st.sidebar.slider('Max Heart Rate', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.0, 1.0, 0.1)
    st_slope = st.sidebar.selectbox('ST Slope', ('upsloping', 'flat', 'downsloping'))

    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBP': resting_bp,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'RestingECG': restecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exang,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode categorical features
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_columns.columns, fill_value=0)
input_scaled = scaler.transform(input_encoded)

# ===============================
# Prediction
# ===============================
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ Heart Disease Detected!")
else:
    st.success("✅ No Heart Disease Detected!")

st.subheader("Prediction Probability")
st.write(f"Probability of No Disease: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Probability of Disease: {prediction_proba[0][1]*100:.2f}%")

# ===============================
# Model Evaluation
# ===============================
st.subheader("Model Performance on Test Data")
y_pred = model.predict(X_test)
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred)*100:.2f}%")

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=X_columns.columns).sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(x=feat_importance.values, y=feat_importance.index, palette="viridis", ax=ax2)
ax2.set_title("XGBoost Feature Importance")
st.pyplot(fig2)
