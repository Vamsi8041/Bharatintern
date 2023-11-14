import streamlit as st
import pandas as pd
import joblib
from PIL import Image

def load_model():
    model = joblib.load('new_model.sav')
    return model

model = load_model()
def main():
    st.title('Iris Flower Classification')
    st.write('This app predicts the species of Iris flowers.')
    st.write('B.R.M.Vamsi\n'
             '\n GMR Institutes of Technology\n'
             '\n Computer Science')

    
    st.header('Enter Input Values')
    sepal_length = st.number_input('Sepal Length', value=5.4)
    sepal_width = st.number_input('Sepal Width', value=3.4)
    petal_length = st.number_input('Petal Length', value=1.3)
    petal_width = st.number_input('Petal Width', value=0.2)
    if st.button('Predict'):
        input_data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }

        input_df = pd.DataFrame([input_data])

        error_message = None
        if any(value <= 0 for value in input_data.values()):
            error_message = "Please enter positive values for all features."
        elif any(value > 20 for value in input_data.values()):
            error_message = "Please enter reasonable values for all features."

        if error_message:
            st.error(error_message)
        else:
            prediction = model.predict(input_df)
            species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}  
            predicted_species = species[prediction[0]]

        
            st.write(f"Predicted Species: {predicted_species}")

main()
