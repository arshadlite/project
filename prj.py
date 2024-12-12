import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset from the repository
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

# Streamlit App
st.set_page_config(page_title="Health Impact Prediction", page_icon="🌍", layout="wide")
st.title("🌍 Health Impact Score Prediction")
st.markdown(
    """
    Enter air quality parameters below to predict the **Health Impact Score**.
    This tool is designed to assist in analyzing urban air quality and its effects on public health.
    """
)

# Add a GIF
st.markdown("![Urban Pollution](https://media.giphy.com/media/3o6gDWzmAzrpi5DQU8/giphy.gif)")

# Load dataset
dataset = load_data()

# Ensure required columns are in the dataset
required_columns = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'HealthImpactScore']
if all(col in dataset.columns for col in required_columns):
    # Split data into features and target
    X = dataset[['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']]
    y = dataset['HealthImpactScore']

    # Train the model
    model = train_model(X, y)

    # Interactive Inputs for Prediction
    st.sidebar.header("🔧 Input Air Quality Parameters")
    aqi = st.sidebar.slider("AQI (Air Quality Index)", int(X['AQI'].min()), int(X['AQI'].max()), int(X['AQI'].mean()))
    pm10 = st.sidebar.slider("PM10 (Particulate Matter ≤ 10μm)", int(X['PM10'].min()), int(X['PM10'].max()), int(X['PM10'].mean()))
    pm2_5 = st.sidebar.slider("PM2.5 (Particulate Matter ≤ 2.5μm)", int(X['PM2_5'].min()), int(X['PM2_5'].max()), int(X['PM2_5'].mean()))
    no2 = st.sidebar.slider("NO2 (Nitrogen Dioxide)", int(X['NO2'].min()), int(X['NO2'].max()), int(X['NO2'].mean()))
    so2 = st.sidebar.slider("SO2 (Sulfur Dioxide)", int(X['SO2'].min()), int(X['SO2'].max()), int(X['SO2'].mean()))
    o3 = st.sidebar.slider("O3 (Ozone)", int(X['O3'].min()), int(X['O3'].max()), int(X['O3'].mean()))
    temperature = st.sidebar.slider("Temperature (°C)", float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
    humidity = st.sidebar.slider("Humidity (%)", float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", float(X['WindSpeed'].min()), float(X['WindSpeed'].max()), float(X['WindSpeed'].mean()))

    # Prediction Button
    if st.button("💡 Predict Health Impact Score"):
        # Create input DataFrame
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

        # Predict the Health Impact Score
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Health Impact Score: {prediction:.2f}")

        # Display an illustrative image
        st.image("https://cdn.pixabay.com/photo/2016/11/21/17/55/pollution-1848827_960_720.jpg", caption="Urban Pollution")
else:
    st.error(f"The dataset must include the following columns: {', '.join(required_columns)}")
