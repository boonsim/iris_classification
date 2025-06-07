import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('iris_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Iris class names for display
iris_species = ['Setosa', 'Versicolor', 'Virginica']

st.title('Iris Flower Species Classification')
st.write('Enter the flower measurements below:')

# Input features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width  = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width  = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Predict button
if st.button('Predict'):
    # Process input and predict
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f'The predicted species is **{iris_species[prediction]}**.')

st.write('Model file: iris_classifier.pkl (Random Forest)')