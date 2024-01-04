import autoclean
import pandas as pd
import tweet_preprocessor

from autoclean import autoclean
from tweet_preprocessor import clean
import preprocessor as tweet_preprocessor


# Load the dataset
df = pd.read_csv('E:/project/project 1/LiverT_dataset1.csv')

# Preprocess the dataset
df = autoclean.clean(df)

# Save the preprocessed dataset
df.to_csv('preprocessed_data.csv', index=False)


import h2o

# Initialize H2O
h2o.init()

# Load the dataset
df = h2o.import_file('E:/project/project 1/LiverT_dataset1.csv')

# Preprocess the dataset
df = h2o.clean(df)

# Save the preprocessed dataset
df.write('preprocessed_data.csv')

import pycaret
from pycaret.preprocessing import setup

# Initialize PyCaret
setup()

# Load the dataset
df = pd.read_csv('E:/project/project 1/LiverT_dataset1.csv')

# Preprocess the dataset
df = setup(df, target='target')

# Save the preprocessed dataset
df.to_csv('preprocessed_data.csv', index=False)

import pandas as pd
import dabl

# Load the dataset
df = pd.read_csv('E:/project/project 1/LiverT_dataset1.csv')

# Perform automated data cleaning
cleaned_df = dabl.clean(df)

# Save the cleaned dataset
cleaned_df.to_csv('cleaned_dataset.csv', index=False)




import pandas as pd
from autoimpute.imputations import MultipleImputer

# Load your dataset with missing values
data = pd.read_csv("E:/project/project 1/LiverT_dataset1.csv")

# Initialize the MultipleImputer with desired imputation strategy
imputer = MultipleImputer(strategy="random")

# Fit and transform the data to impute missing values
imputed_data = imputer.fit_transform(data)

# Convert generator object to DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=data.columns)

# View the imputed dataset
print(imputed_df.head())

