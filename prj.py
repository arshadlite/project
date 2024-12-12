import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

# Streamlit App
st.title("Urban Air Quality and Health Impact Analysis")

# Sidebar for uploading dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("air_quality_health_impact_data.csv", type=["csv"])

if uploaded_file:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(dataset.head())
    
    # Feature and Target Selection
    st.sidebar.header("Feature Selection")
    features = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
    target = 'HealthImpactScore'

    # Ensure columns are present
    if all(col in dataset.columns for col in features + [target]):
        X = dataset[features]
        y = dataset[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model Training
        st.sidebar.header("Model Parameters")
        n_estimators = st.sidebar.slider("Number of Trees in Random Forest", 10, 200, 100)
        max_depth = st.sidebar.slider("Maximum Depth of Trees", 5, 50, 20)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Model Performance
        st.write("### Model Performance")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

        # SHAP Feature Importance
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        st.write("### Feature Importance (SHAP)")
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot()

        # Predictions vs Actuals
        st.write("### Predictions vs Actuals")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        
        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        corr = dataset[features + [target]].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    else:
        st.error("The uploaded dataset does not contain all required columns.")
else:
    st.info("Please upload a CSV file to proceed.")
