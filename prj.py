import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap

# Streamlit UI
st.title("Urban Air Quality and Health Impact Analysis")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("air_quality_health_impact_data.csv", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(data.head())
    
    # Select columns for features and target
    st.sidebar.write("### Model Configuration")
    features = st.sidebar.multiselect("Select feature columns", options=data.columns)
    target = st.sidebar.selectbox("Select target column", options=data.columns)

    if features and target:
        # Splitting data
        X = data[features]
        y = data[target]

        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Random Forest Regressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Display model performance
        st.write("### Model Performance")
        from sklearn.metrics import mean_squared_error, r2_score
        st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")

        # Feature importance using SHAP
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        st.write("### Feature Importance (SHAP)")
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot()

        # Predictions vs Actuals
        st.write("### Predictions vs Actual")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=predictions, ax=ax)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
