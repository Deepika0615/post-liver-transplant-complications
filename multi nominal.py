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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE



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
joblib.dump(impute_data, 'E:\project\project 108\final model\impute')

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
joblib.dump(winz_data, 'E:\project\project 108\final model\winsor')

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
joblib.dump(scale, 'E:\project\project 108\final model\minmax')

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
joblib.dump(onehot,'E:\project\project 108\final model\encoding')

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


logmodel = LogisticRegression(multi_class="multinomial", solver="newton-cg")

# basic model 

# Train the classifier
model = logmodel.fit(X_train_resampled, Y_train_resampled)

train_pred = model.predict(X_train_resampled)
test_pred = model.predict(X_test)

# Calculate the accuracy of the classifier on the training set
train_accuracy = accuracy_score(Y_train_resampled, train_pred)

# Calculate the accuracy of the classifier on the test set
test_accuracy = accuracy_score(Y_test, test_pred)

precision = precision_score(Y_test, test_pred, average='weighted')
recall = recall_score(Y_test, test_pred, average='weighted')
f1 = f1_score(Y_test, test_pred, average='weighted')

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

param_grid = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"]
}

grid_search = GridSearchCV(logmodel, param_grid, cv=5)
best_clf = grid_search.fit(X_train_resampled, Y_train_resampled)
best_model = best_clf.best_estimator_
best_score = best_clf.best_score_

print("Best Parameters:", best_clf.best_params_)
print("Best Score:", best_score)

# Train the best model on the entire training set
best_model.fit(X_train_resampled, Y_train_resampled)
best_model_test_pred = best_model.predict(X_test)

best_model_test_accuracy = accuracy_score(Y_test, best_model_test_pred)
best_model_precision = precision_score(Y_test, best_model_test_pred, average='weighted')
best_model_recall = recall_score(Y_test, best_model_test_pred, average='weighted')
best_model_f1 = f1_score(Y_test, best_model_test_pred, average='weighted')
best_model_train_pred = best_model.predict(X_train_resampled)
best_model_train_accuracy = accuracy_score(Y_train_resampled, best_model_train_pred)


print("Best Model Training Accuracy:", best_model_train_accuracy)
print("Best Model Test Accuracy:", best_model_test_accuracy)
print("Best Model Precision:", best_model_precision)
print("Best Model Recall:", best_model_recall)
print("Best Model F1-score:", best_model_f1)

# 2 using rfe 

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


# Apply Recursive Feature Elimination
logmodel = LogisticRegression(multi_class="multinomial", solver="newton-cg")
rfe = RFE(logmodel, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train_resampled, Y_train_resampled)
X_test_rfe = rfe.transform(X_test)

# Train the classifier
model = logmodel.fit(X_train_rfe, Y_train_resampled)

train_pred = model.predict(X_train_rfe)
test_pred = model.predict(X_test_rfe)

# Calculate the accuracy of the classifier on the training set
train_accuracy = accuracy_score(Y_train_resampled, train_pred)

# Calculate the accuracy of the classifier on the test set
test_accuracy = accuracy_score(Y_test, test_pred)

precision = precision_score(Y_test, test_pred, average='weighted')
recall = recall_score(Y_test, test_pred, average='weighted')
f1 = f1_score(Y_test, test_pred, average='weighted')

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Save the model
import pickle
pickle.dump(rfe, open('E:\project\project 108\final model\model.pkl', 'wb'))

# Define the parameter grid for grid search
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10.0]
}

# Perform grid search
grid_search = GridSearchCV(logmodel, param_grid, cv=5)
grid_search.fit(X_train_rfe, Y_train_resampled)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on train and test sets
train_pred = best_model.predict(X_train_rfe)
test_pred = best_model.predict(X_test_rfe)

# Calculate metrics
train_accuracy = accuracy_score(Y_train_resampled, train_pred)
test_accuracy = accuracy_score(Y_test, test_pred)
precision = precision_score(Y_test, test_pred, average='weighted')
recall = recall_score(Y_test, test_pred, average='weighted')
f1 = f1_score(Y_test, test_pred, average='weighted')

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Predictions on New Data

model = pickle.load(open('E:\project\project 108\final model\model.pkl', 'rb'))
impute = joblib.load('E:\project\project 108\final model\impute')
winzor = joblib.load('E:\project\project 108\final model\winsor')
minmax = joblib.load('E:\project\project 108\final model\minmax')
onehot = joblib.load('E:\project\project 108\final model\encoding')

data = pd.read_csv(r"C:/Users/Lenovo/test.csv")
data.shape
numeric_features = data.select_dtypes(exclude = ['object']).columns

clean = pd.DataFrame(impute.transform(data), columns = numeric_features)
clean1 = pd.DataFrame(winzor.transform(clean), columns = numeric_features)
clean2 = pd.DataFrame(minmax.transform(clean1), columns = numeric_features)
clean3 = pd.DataFrame(onehot.transform(data),columns = onehot.get_feature_names_out())


clean_data = pd.concat([clean2, clean3], axis = 1, ignore_index = True)
clean_data

prediction = pd.DataFrame(model.predict(clean_data), columns = ['Complications'])
prediction
