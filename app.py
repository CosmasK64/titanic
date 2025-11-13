import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler= joblib.load('scaler.pkl')
model=joblib.load('titanic_model.pkl')

st.title("Titanic Survival Prediction")

st.write("Please enter the passenger details below:")
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex=st.selectbox('Sex', ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
sibsp= st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch= st.slider("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 600.0, 40.0)
embarked=st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
adult_male= st.selectbox('Is Adult Male?', [0, 1])

if st.button("Predict Survival"):

    sex_encoded = 1 if sex == 'male' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    input_data = pd.DataFrame({
        'pclass': [pclass],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'adult_male': [adult_male],
        'sex_encoded': [sex_encoded],
        'embarked_Q': [embarked_Q],
        'embarked_S': [embarked_S]
    })

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][prediction]

    if prediction == 1:
        st.success(f"The model predicts that the passenger would survive with a probability of {probability:.2f}.")
    else:
        st.error(f"The model predicts that the passenger would not survive with a probability of {probability:.2f}.")