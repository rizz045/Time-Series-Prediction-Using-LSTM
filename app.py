import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# Load model and scaler
model = load_model("power_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Feature names used during training
FEATURES = ['hour', 'day', 'weekday', 'month']

st.set_page_config(page_title="⚡ Power Forecast", layout="centered")
st.title("🔋 LSTM Power Supply Forecast")
st.markdown("Enter a date and time to predict the future power supply.")

# --- User Input ---
date_input = st.date_input("Select date:", value=datetime(2002, 4, 8).date())
time_input = st.time_input("Select time:", value=datetime(2002, 4, 8, 1, 0).time())
input_datetime = datetime.combine(date_input, time_input)

if st.button("📈 Predict Power"):
    try:
        dt = pd.to_datetime(input_datetime)

        # Extract features
        features = pd.DataFrame([{
            'hour': dt.hour,
            'day': dt.day,
            'weekday': dt.weekday(),
            'month': dt.month
        }])

        # Pad with dummy target for scaling
        dummy_target = [0]
        input_scaled = scaler.transform(np.hstack([features.values, [dummy_target]]))[:, :-1]

        # Reshape for LSTM (1, time steps, features)
        input_reshaped = input_scaled.reshape((1, 1, len(FEATURES)))

        # Predict
        scaled_output = model.predict(input_reshaped)

        # Pad zeros for inverse scaling
        padded_output = np.hstack([np.zeros((1, scaler.n_features_in_ - 1)), scaled_output])
        predicted_power = scaler.inverse_transform(padded_output)[0, -1]

        st.success(f"🔮 Predicted Power Supply: **{predicted_power:.2f} units**")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
