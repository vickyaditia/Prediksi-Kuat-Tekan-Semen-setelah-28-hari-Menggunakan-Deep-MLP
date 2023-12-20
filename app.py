import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib

# Load the trained model
model = load_model("kel10.h5")

# Load the StandardScaler
scaler = joblib.load("scaler.joblib")

def preprocess_input(input_data, scaler):
    # Standardize input features using the loaded scaler
    input_data_std = scaler.transform(input_data)
    return input_data_std

def predict(input_data):
    # Make predictions using the loaded model
    predictions = model.predict(input_data)
    return predictions

def main():
    st.title("Model Deployment with Streamlit")

    # Create input form for user input
    st.header("Input Features:")
    mgo = st.number_input("MgO", min_value=0.0, max_value=100.0, value=50.0)
    cao = st.number_input("CaO", min_value=0.0, max_value=100.0, value=50.0)
    so3 = st.number_input("SO3", min_value=0.0, max_value=100.0, value=50.0)
    loi = st.number_input("LOI", min_value=0.0, max_value=100.0, value=50.0)
    fl = st.number_input("FL", min_value=0.0, max_value=100.0, value=50.0)
    insol = st.number_input("Insol", min_value=0.0, max_value=100.0, value=50.0)

    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'MgO': [mgo],
        'CaO': [cao],
        'SO3': [so3],
        'LOI': [loi],
        'FL': [fl],
        'Insol': [insol]
    })

    # Preprocess input features using the loaded scaler
    input_data_std = preprocess_input(input_data, scaler)

    # Make predictions
    predictions = predict(input_data_std)

    # Display the predictions
    st.header("Predictions:")
    st.write(f"Predicted Value: {predictions[0][0]}")

if __name__ == "__main__":
    main()
