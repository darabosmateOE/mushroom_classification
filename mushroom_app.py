import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# Assuming you have already trained and saved your models
# Replace these with the actual paths to your saved models
log_reg_model_path = 'path_to_log_reg_model.pkl'
svm_model_path = 'path_to_svm_model.pkl'
nn_model_path = 'path_to_nn_model.pkl'
rf_model_path = 'path_to_rf_model.pkl'

# Load the models
log_reg_model = joblib.load(log_reg_model_path)
svm_model = joblib.load(svm_model_path)
nn_model = joblib.load(nn_model_path)
rf_model = joblib.load(rf_model_path)

# Load your dataset
df = pd.read_csv('your_dataset.csv')  # Replace with the path to your dataset

# Assuming X is your feature matrix and y is your target variable
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your models (you might not need this part if your models are pre-trained)

# Streamlit App
st.title("Model Comparison App")

# Sidebar for User Inputs
model_selector = st.sidebar.selectbox("Select Model", ("Logistic Regression", "SVM", "Neural Network", "Random Forest"))

# Display Confusion Matrix and Metrics
if model_selector == "Logistic Regression":
    st.header("Logistic Regression Model")
    y_pred = log_reg_model.predict(X_test)
elif model_selector == "SVM":
    st.header("SVM Model")
    y_pred = svm_model.predict(X_test)
elif model_selector == "Neural Network":
    st.header("Neural Network Model")
    y_pred = nn_model.predict(X_test)
elif model_selector == "Random Forest":
    st.header("Random Forest Model")
    y_pred = rf_model.predict(X_test)

# Display Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.table(cm)

# Display Metrics
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Metrics")
st.write(f"Accuracy: {accuracy}")

# Display ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
st.pyplot(fig)
