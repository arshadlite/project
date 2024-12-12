import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App
st.title("Health Impact Score Prediction")
st.write("Upload your dataset and enter air quality parameters to predict the Health Impact Score.")

# Sidebar for uploading dataset
uploaded_file = st.sidebar.file_uploader("air_quality_health_impact_data.csv", type=["csv"])

if uploaded_file:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(dataset.head())

    # Ensure required columns are in the dataset
    required_columns = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'HealthImpactScore']
    if all(col in dataset.columns for col in required_columns):
        # Split data into features and target
        X = dataset[['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']]
        y = dataset['HealthImpactScore']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        st.write("### Model Performance")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

        # Interactive Input for Prediction
        st.write("### Predict Health Impact Score")
        aqi = st.slider("AQI (Air Quality Index)", int(X['AQI'].min()), int(X['AQI'].max()), int(X['AQI'].mean()))
        pm10 = st.slider("PM10 (Particulate Matter ≤ 10μm)", int(X['PM10'].min()), int(X['PM10'].max()), int(X['PM10'].mean()))
        pm2_5 = st.slider("PM2.5 (Particulate Matter ≤ 2.5μm)", int(X['PM2_5'].min()), int(X['PM2_5'].max()), int(X['PM2_5'].mean()))
        no2 = st.slider("NO2 (Nitrogen Dioxide)", int(X['NO2'].min()), int(X['NO2'].max()), int(X['NO2'].mean()))
        so2 = st.slider("SO2 (Sulfur Dioxide)", int(X['SO2'].min()), int(X['SO2'].max()), int(X['SO2'].mean()))
        o3 = st.slider("O3 (Ozone)", int(X['O3'].min()), int(X['O3'].max()), int(X['O3'].mean()))
        temperature = st.slider("Temperature (°C)", float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
        humidity = st.slider("Humidity (%)", float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
        wind_speed = st.slider("Wind Speed (m/s)", float(X['WindSpeed'].min()), float(X['WindSpeed'].max()), float(X['WindSpeed'].mean()))

        # Make a prediction
        if st.button("Predict"):
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
else:
    st.info("Please upload a CSV file to proceed.")
