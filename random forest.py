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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train_resampled, Y_train_resampled)  

# Make predictions on the train set
y_train_pred = rf_classifier.predict(X_train_resampled)  

# Calculate the train accuracy
train_accuracy = accuracy_score(Y_train_resampled, y_train_pred)  
print("Train Accuracy:", train_accuracy)

# Make predictions on the test set
y_test_pred = rf_classifier.predict(X_test)  

# Calculate the test accuracy
test_accuracy = accuracy_score(Y_test, y_test_pred)  
print("Test Accuracy:", test_accuracy)

# grid serach cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of each tree
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, Y_train_resampled)  

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Print the best mean cross-validated score
print("Best Mean Cross-Validated Score:", grid_search.best_score_)

# f1 score

from sklearn.metrics import f1_score

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the classifier to the training data
rf_classifier.fit(X_train_resampled, Y_train_resampled)  

# Calculate the F1 score for the train data
train_f1_score = f1_score(Y_train_resampled, y_train_pred, average='macro')  
print("Train F1 Score:", train_f1_score)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)  
# Calculate the F1 score
f1 = f1_score(Y_test, y_pred, average='macro')  
print("F1 Score:", f1)

from sklearn.metrics import precision_score, recall_score

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)  
# Calculate precision and recall scores
precision = precision_score(Y_test, y_pred, average='macro')  
recall = recall_score(Y_test, y_pred, average='macro')  

print("Precision:", precision)
print("Recall:", recall)
