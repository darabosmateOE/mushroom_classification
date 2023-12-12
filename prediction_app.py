import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from time import sleep


feature_names = ['bruises', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'ring-type']
df = pd.read_csv('mushrooms.csv')

df = df[feature_names]
print(df.head())

# Assuming you have already trained and saved your models
# Replace these with the actual paths to your saved models
log_reg_model_path = 'logistic_regression_model.pkl'
svm_model_path = 'svm_model.pkl'
nn_model_path = 'neural_network_model.pkl'
rf_model_path = 'random_forest_model.pkl'

# Load the models
log_reg_model = joblib.load(log_reg_model_path)
svm_model = joblib.load(svm_model_path)
nn_model = joblib.load(nn_model_path)
rf_model = joblib.load(rf_model_path)

# Create a dictionary to map input feature names to their default values
input_features = {
    'bruises': 't',
    'gill-spacing': 'c',
    'gill-size': 'b',
    'gill-color': 'k',
    'stalk-root': 'c',
    'stalk-surface-above-ring': 's',
    'stalk-surface-below-ring': 's',
    'ring-type': 'p'
}

# Streamlit App
st.title("Mushroom Prediction App")

# Sidebar for User Inputs
st.sidebar.header("Input Mushroom Parameters")

# Allow the user to input values for each feature
for feature, default_value in input_features.items():
    input_features[feature] = st.sidebar.selectbox(f"{feature.capitalize()}: ", df[feature].unique()) #index=df[feature].unique().tolist().index(default_value)

# Display a button to trigger predictions
if st.sidebar.button("Get Predictions"):

    # Create a DataFrame with user-input values
    user_input = pd.DataFrame([input_features])
    

    # Display user-input values
    st.subheader("User Input:")
    st.table(user_input)

    #add user input to dataset for encoding
    df.iloc[-1] = user_input.iloc[0]

    # encode dataframe
    mappings = list()
    encoder = LabelEncoder()
    for column in range(len(df.columns)):
        df[df.columns[column]] = encoder.fit_transform(df[df.columns[column]]) #transform every column to numerical values
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)} #create dictionary for encoded and original values
        mappings.append(mappings_dict) #append dictionary to mappings list

    user_input = df.iloc[[-1]]
    #st.table(user_input)

    
    # Make predictions with each model
    log_reg_prediction = log_reg_model.predict(user_input)[0]
    svm_prediction = svm_model.predict(user_input)[0]
    nn_prediction = nn_model.predict(user_input)[0]
    rf_prediction = rf_model.predict(user_input)[0]


    # Display predictions
    st.subheader("Model Predictions:")
    st.write(f"Logistic Regression Prediction: {log_reg_prediction}")
    st.write(f"SVM Prediction: {svm_prediction}")
    st.write(f"Neural Network Prediction: {nn_prediction}")
    st.write(f"Random Forest Prediction: {rf_prediction}")