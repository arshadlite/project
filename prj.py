import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset from the repository
@st.cache
def load_data():
    return pd.read_csv("air_quality_health_impact_data.csv")  # Replace 'dataset.csv' with your dataset file path in the repository

# Streamlit App
st.title("Health Impact Score Prediction")
st.write("Enter air quality parameters to predict the Health Impact Score based on preloaded models.")

# Load dataset
dataset = load_data()

# Ensure required columns are in the dataset
required_columns = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'HealthImpactScore']
if all(col in dataset.columns for col in required_columns):
    # Split data into features and target
    X = dataset[['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']]
    y = dataset['HealthImpactScore']

    # Train a Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Interactive Input for Prediction
    st.write("### Enter Air Quality Parameters")
    aqi = st.number_input("AQI (Air Quality Index)", min_value=int(X['AQI'].min()), max_value=int(X['AQI'].max()), value=int(X['AQI'].mean()))
    pm10 = st.number_input("PM10 (Particulate Matter ≤ 10μm)", min_value=int(X['PM10'].min()), max_value=int(X['PM10'].max()), value=int(X['PM10'].mean()))
    pm2_5 = st.number_input("PM2.5 (Particulate Matter ≤ 2.5μm)", min_value=int(X['PM2_5'].min()), max_value=int(X['PM2_5'].max()), value=int(X['PM2_5'].mean()))
    no2 = st.number_input("NO2 (Nitrogen Dioxide)", min_value=int(X['NO2'].min()), max_value=int(X['NO2'].max()), value=int(X['NO2'].mean()))
    so2 = st.number_input("SO2 (Sulfur Dioxide)", min_value=int(X['SO2'].min()), max_value=int(X['SO2'].max()), value=int(X['SO2'].mean()))
    o3 = st.number_input("O3 (Ozone)", min_value=int(X['O3'].min()), max_value=int(X['O3'].max()), value=int(X['O3'].mean()))
    temperature = st.number_input("Temperature (°C)", min_value=float(X['Temperature'].min()), max_value=float(X['Temperature'].max()), value=float(X['Temperature'].mean()))
    humidity = st.number_input("Humidity (%)", min_value=float(X['Humidity'].min()), max_value=float(X['Humidity'].max()), value=float(X['Humidity'].mean()))
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=float(X['WindSpeed'].min()), max_value=float(X['WindSpeed'].max()), value=float(X['WindSpeed'].mean()))

    # Predict Health Impact Score
    if st.button("Predict Health Impact Score"):
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
        st.write(f"### Predicted Health Impact Score: {prediction:.2f}")
else:
    st.error(f"The dataset must include the following columns: {', '.join(required_columns)}")
