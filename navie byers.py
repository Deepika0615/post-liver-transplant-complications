import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


import sklearn.metrics as skmet
import joblib
import pickle

liverT = pd.read_csv(r"E:\project\project 1\LiverT_dataset1.csv")

# Convert empty strings to NaN
liverT['RBloodTransfusion'] = liverT['RBloodTransfusion'].replace('', np.nan)
# Drop rows with NaN values in R_Blood_Transfusion column
liverT = liverT.dropna(subset=['RBloodTransfusion'])

# Convert the column to int64
liverT['RBloodTransfusion'] = liverT['RBloodTransfusion'].astype('int64')

X = liverT.drop('Complications', axis=1)
Y = liverT['Complications']

# numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

# categorical columns
categorical_features = X.select_dtypes(include=['object']).columns
categorical_features

numeric_features1 = X.select_dtypes(include = ['int64']).columns
numeric_features2 = X.select_dtypes(include = ['float64']).columns

# Mode imputation for Integer (categorical) data
num_pipeline1 = Pipeline(steps=[('impute1', SimpleImputer(strategy = 'most_frequent'))])

# Mean imputation for Continuous (Float) data
num_pipeline2 = Pipeline(steps=[('impute2', SimpleImputer(strategy = 'mean'))])


#Imputation Transformer
preprocessor = ColumnTransformer([
        ('mode', num_pipeline1, numeric_features1),
        ('mean', num_pipeline2, numeric_features2)])

# Fit the data to train imputation pipeline model
impute_data = preprocessor.fit(X)


# Save the pipeline
# joblib.dump(impute_data, 'E:/project/project 108/impute')

# Transform the original dat
num_data = pd.DataFrame(impute_data.transform(X), columns = numeric_features) 

num_data.isna().sum()

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['RNa', 'RMg'])

# outlier pipeline
outlier_pipeline = Pipeline(steps = [('winsor', winsor)])

# Outlier Transformer
preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, numeric_features)], remainder = 'passthrough')

# Fit the data to train outlier pipeline model
winz_data = preprocessor1.fit(num_data)

# Save the pipeline
# joblib.dump(winz_data, 'E:/project/project 108/winsor')

# Transform the original data
outlier_data = pd.DataFrame(winz_data.transform(num_data), columns = numeric_features) 
outlier_data

import seaborn as sns
print(sns.boxplot(outlier_data.RNa))
print(sns.boxplot(outlier_data.RMg))

# Scale pipeline
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

# Scale Transformer
preprocessor2 = ColumnTransformer([('scale', scale_pipeline, numeric_features)])

# Fit the data to train Scale pipeline model
scale = preprocessor2.fit(num_data)

# Save the MinMaxScaler Model
# joblib.dump(scale, 'E:/project/project 108/minmax')

# Transform the original data
scaled_data = pd.DataFrame(scale.transform(num_data), columns = numeric_features)
scaled_data

# OneHotEncoder pipeline
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Encoding Transformer
preprocessor3 = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])

# Fit the data to train Scale pipeline model
onehot = preprocessor3.fit(X)

# Save the Encoding model
# joblib.dump(onehot,'E:/project/project 108/encoding')

# Transform the original data
encode_data = pd.DataFrame(onehot.transform(X),columns = onehot.get_feature_names_out())
encode_data


clean_data= pd.concat([scaled_data, encode_data], axis = 1, ignore_index = True)
clean_data

X_train, X_test, Y_train, Y_test = train_test_split(clean_data, Y, test_size = 0.2, stratify = Y, random_state = 0) 


from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# SMOTE 
smote = SMOTE(random_state=0)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Create a Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train_resampled, Y_train_resampled)

# Make predictions on the train and test data
y_train_pred = nb_classifier.predict(X_train_resampled)
y_test_pred = nb_classifier.predict(X_test)

# Calculate train and test accuracy
train_accuracy = accuracy_score(Y_train_resampled, y_train_pred)
test_accuracy = accuracy_score(Y_test, y_test_pred)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

from sklearn.metrics import f1_score


# Create a Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train_resampled, Y_train_resampled)

# Make predictions on the train and test data
y_train_pred = nb_classifier.predict(X_train_resampled)
y_test_pred = nb_classifier.predict(X_test)

# Calculate the F1 score for the train and test sets
train_f1_score = f1_score(Y_train_resampled, y_train_pred, average='weighted')
test_f1_score = f1_score(Y_test, y_test_pred, average='weighted')

print("Train F1 Score:", train_f1_score)
print("Test F1 Score:", test_f1_score)

# grid serach cv
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, Y_train_resampled)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)

# Calculate the accuracy of the best model on the test data
accuracy = accuracy_score(Y_test, y_pred)
print("Test Accuracy:", accuracy)

from sklearn.metrics import precision_score, recall_score

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate precision and recall scores
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
