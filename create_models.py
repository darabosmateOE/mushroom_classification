#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler #could be onehot encoder?
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier #could use tensorflow?

import joblib


def prepare_data(noise=0):
    data = pd.read_csv('mushrooms.csv')
    # select features
    selected_columns = ['class', 'bruises', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-root',
                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'ring-type']

    # Select the desired columns from the DataFrame
    data = data.loc[:, selected_columns]
    # encode dataframe
    mappings = list()
    encoder = LabelEncoder()
    for column in range(len(data.columns)):
        data[data.columns[column]] = encoder.fit_transform(data[data.columns[column]]) #transform every column to numerical values
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)} #create dictionary for encoded and original values
        mappings.append(mappings_dict) #append dictionary to mappings list


    # Split data into features and target
    y = data['class']
    X = data.drop(['class'], axis=1)

    

    # set noise percentage
    noise_percentage = noise

    # Calculate the number of noisy data points
    num_noise_points = int(len(X) * (noise_percentage / 100))

    # Choose random indices to add noise
    noise_indices = np.random.choice(len(X), num_noise_points, replace=False)

    # Add noise to selected indices
    for column in X.columns:
        # Assuming your data is categorical with numerical values
        unique_values = X[column].unique()
        X.loc[noise_indices, column] = np.random.choice(unique_values, size=num_noise_points)

    return X, y



data = pd.read_csv('mushrooms.csv')

X, y = prepare_data(noise=13)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
rf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42, max_depth=20, min_samples_leaf=1, min_samples_split=5)
nn = MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu', solver='adam')#set max iter etc.

log_reg.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
nn.fit(X_train, y_train)

joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(nn, 'neural_network_model.pkl')

print('Logistic Regression Score: ', log_reg.score(X_test, y_test))
print('----------------SVM Score: ', svm.score(X_test, y_test))
print('------Random Forest Score: ', rf.score(X_test, y_test))
print('-----Neural Network Score: ', nn.score(X_test, y_test))

print(log_reg.predict([X_train.iloc[0]]))
print(X_train)