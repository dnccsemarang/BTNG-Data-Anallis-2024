import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# Load trained LSTM model and scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('lstm_weather_model.keras')

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.pkl')
    except FileNotFoundError:
        return fit_scaler()

def fit_scaler():
    df = pd.read_csv('seattle-weather.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_sunny'] = df['weather'].map(lambda x: 1 if x == 'sun' else 0)
    df['day_of_week'] = df['date'].dt.dayofweek

    scaler = MinMaxScaler()
    scaler.fit(df[['precipitation', 'temp_max', 'temp_min', 'wind', 'day', 'month', 'year', 'is_sunny', 'day_of_week']])
    joblib.dump(scaler, 'scaler.pkl')
    return scaler

# UI Configuration
st.set_page_config(page_title="Weather Prediction App", layout="centered")

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTitle {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stSubheader {
        color: #0D47A1;
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .input-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .prediction-container {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #757575;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown("<h1 class='stTitle'>🌦️ Weather Prediction App</h1>", unsafe_allow_html=True)

# Create two columns for input form and prediction results
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h2 class='stSubheader'>Enter Weather Features</h2>", unsafe_allow_html=True)
    
    with st.form(key='weather_form'):

        precipitation = st.number_input('Precipitation (mm)', min_value=0.0, step=0.01, format="%.2f")
        temp_max = st.number_input('Max Temperature (°C)', step=0.1, format="%.1f")
        temp_min = st.number_input('Min Temperature (°C)', step=0.1, format="%.1f")
        wind = st.number_input('Wind Speed (m/s)', step=0.1, format="%.1f")

        # Date input section
        st.markdown("<h3 class='stSubheader'>Date Information</h3>", unsafe_allow_html=True)
        min_date = date(2012, 1, 1)
        max_date = date(2015, 12, 31)
        
        selected_date = st.date_input(
            "Select Date",
            value=date(2012, 1, 1),
            min_value=min_date,
            max_value=max_date
        )

        is_sunny_input = st.selectbox('Is it Sunny?', ['Yes', 'No'])

        st.markdown("</div>", unsafe_allow_html=True)
        
        submit_button = st.form_submit_button(label='Predict Weather')

with col2:

    if submit_button:
        lstm_model = load_model()
        scaler = load_scaler()

        is_sunny = 1 if is_sunny_input == 'Yes' else 0
        day_of_week = selected_date.weekday()

        input_data = np.array([[precipitation, temp_max, temp_min, wind, 
                                selected_date.day, selected_date.month, selected_date.year, 
                                is_sunny, day_of_week]])
        input_data_scaled = scaler.transform(input_data)
        input_data_lstm = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

        # Make prediction
        prediction = lstm_model.predict(input_data_lstm)
        predicted_category = np.argmax(prediction, axis=-1)

        # Map the prediction to weather categories
        label_mapping = {0: 'drizzle', 1: 'fog', 2: 'rain', 3: 'snow', 4: 'sun'}
        predicted_weather = label_mapping[predicted_category[0]]

        st.markdown(f"""
            <h3 style="color: #0D47A1; margin-bottom: 10px;">Predicted Weather</h3>
            <p style="font-size: 24px; font-weight: bold; color: #1E88E5;">{predicted_weather.capitalize()}</p>
        """, unsafe_allow_html=True)
    else:
        st.info("Enter the weather features and click 'Predict Weather' to see the prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Ⓒ Copyright 2024, DNCC Data Analyst</div>", unsafe_allow_html=True)