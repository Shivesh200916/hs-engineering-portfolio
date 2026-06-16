import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(page_title="AI ECG Arrhythmia Detector", page_icon="❤️", layout="wide")

st.title("❤️ AI Irregular Heartbeat Detector")
st.write("This application uses a Machine Learning model (Random Forest) to analyze ECG signals and detect abnormal rhythms.")

@st.cache_resource
def load_and_train_model():
    # Load public ECG benchmark dataset
    url = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'
    df = pd.read_csv(url, header=None)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test, y_test

model, scaler, X_test, y_test = load_and_train_model()

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
sample_index = st.sidebar.slider("Select a Test Heartbeat Sample:", 0, len(X_test) - 1, 10)

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("ECG Waveform Analysis")
    current_signal = X_test[sample_index]
    actual_label = "Normal" if y_test[sample_index] == 1 else "Abnormal/Arrhythmia"
    
    scaled_signal = scaler.transform(current_signal.reshape(1, -1))
    prediction = model.predict(scaled_signal)[0]
    prediction_prob = model.predict_proba(scaled_signal)[0]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    color = "green" if prediction == 1 else "red"
    ax.plot(current_signal, color=color, linewidth=2.5, label="ECG Lead Signal")
    ax.set_title(f"Patient Sample #{sample_index} Signal Waveform")
    ax.set_xlabel("Time-steps (ms)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("AI Diagnostic Results")
    if prediction == 1:
        st.success("Verdict: **NORMAL HEART RHYTHM**")
    else:
        st.error("Verdict: **IRREGULAR HEARTBEAT DETECTED**")
        
    st.write("---")
    st.write(f"**True Medical Label:** {actual_label}")
    st.write("**Model Confidence Metrics:**")
    st.progress(float(prediction_prob[prediction]))
    st.write(f"The AI is **{prediction_prob[prediction]*100:.1f}%** confident.")

st.write("---")
st.caption("Disclaimer: This is an educational AI project framework and should not be used as a real-world medical diagnostic tool.")
