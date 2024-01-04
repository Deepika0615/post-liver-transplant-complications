import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the trained model and other necessary transformers
model = pickle.load(open('model.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winsor')
minmax = joblib.load('minmax')
onehot = joblib.load('encoding')

# Define the Streamlit app
def main():
    st.title("Prediction Model Deployment")
    st.write("Upload a CSV file for predictions.")

    # Create a file uploader
    file = st.file_uploader("Upload a CSV file", type="csv")

    if file is not None:
        # Read the uploaded file
        data = pd.read_csv(file)
     
        numeric_features = data.select_dtypes(exclude = ['object']).columns
  
       # Apply data preprocessing steps to numeric columns
        clean_numeric = pd.DataFrame(impute.transform(data), columns=numeric_features)
        clean_numeric = pd.DataFrame(winzor.transform(clean_numeric), columns=numeric_features)
        clean_numeric = pd.DataFrame(minmax.transform(clean_numeric), columns=numeric_features)

        # Apply one-hot encoding to categorical columns
        clean_categorical = pd.DataFrame(onehot.transform(data), columns=onehot.get_feature_names_out())

        # Combine preprocessed numeric and categorical data
        clean_data = pd.concat([clean_numeric, clean_categorical], axis=1)

        # Perform predictions using the model
        prediction = pd.DataFrame(model.predict(clean_data), columns=['Complications'])

        # Concatenate the prediction with the original data
        final = pd.concat([prediction, data], axis=1)

        # Display the final prediction
        st.write("Final Prediction:")
        st.write(final)

if __name__ == '__main__':
    main()
