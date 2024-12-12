import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Set up page configuration with a background image
st.set_page_config(page_title="Health Impact Prediction", page_icon="🌍", layout="centered")

# CSS for background image and styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://cdn.pixabay.com/photo/2016/11/18/16/20/storm-1838341_960_720.jpg');
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("air_quality_health_impact_data.csv")  # Replace 'dataset.csv' with your dataset file path in the repository

# Train a Random Forest model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Main content
st.title("🌍 Health Impact Score Prediction")
st.markdown(
    """
    Enter air quality parameters below to predict the **Health Impact Score**.
    This tool helps in analyzing urban air quality and its impact on public health.
    """
)

# Load dataset
dataset = load_data()

# Check required columns
required_columns = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'HealthImpactScore']
if all(col in dataset.columns for col in required_columns):
    # Prepare data
    X = dataset[['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']]
    y = dataset['HealthImpactScore']
    model = train_model(X, y)

    # Interactive input sliders
    st.write("### Input Air Quality Parameters")
    col1, col2 = st.columns(2)

    with col1:
        aqi = st.slider("AQI (Air Quality Index)", int(X['AQI'].min()), int(X['AQI'].max()), int(X['AQI'].mean()))
        pm10 = st.slider("PM10 (Particulate Matter ≤ 10μm)", int(X['PM10'].min()), int(X['PM10'].max()), int(X['PM10'].mean()))
        pm2_5 = st.slider("PM2.5 (Particulate Matter ≤ 2.5μm)", int(X['PM2_5'].min()), int(X['PM2_5'].max()), int(X['PM2_5'].mean()))
        no2 = st.slider("NO2 (Nitrogen Dioxide)", int(X['NO2'].min()), int(X['NO2'].max()), int(X['NO2'].mean()))

    with col2:
        so2 = st.slider("SO2 (Sulfur Dioxide)", int(X['SO2'].min()), int(X['SO2'].max()), int(X['SO2'].mean()))
        o3 = st.slider("O3 (Ozone)", int(X['O3'].min()), int(X['O3'].max()), int(X['O3'].mean()))
        temperature = st.slider("Temperature (°C)", float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
        humidity = st.slider("Humidity (%)", float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
        wind_speed = st.slider("Wind Speed (m/s)", float(X['WindSpeed'].min()), float(X['WindSpeed'].max()), float(X['WindSpeed'].mean()))

    # Predict Health Impact Score
    if st.button("💡 Predict Health Impact Score"):
        input_data = pd.DataFrame({
            'AQI': [aqi],
            'PM10': [pm10],
            'PM2_5': [pm2_5],
            'NO2': [no2],
            'SO2': [so2],
            'O3': [o3],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'WindSpeed': [wind_speed],
        })
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Health Impact Score: {prediction:.2f}")
else:
    st.error(f"The dataset must include the following columns: {', '.join(required_columns)}")
