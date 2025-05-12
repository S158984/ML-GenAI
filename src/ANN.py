import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title('Churn Prediction')
st.write('This is a simple web app to predict customer churn using a neural network model.')
print('stramlit initial page')
#st.set_option('deprecation.showPyplotGlobalUse', False)

model=load_model('F:\MLProject\ML-GenAI\Data\model.h5')
# Load the model from the file
         #Lable encoder unpickling
with open('F:\MLProject\ML-GenAI\Data\lencoder.pkl', 'rb') as f:
    labelcoder = pickle.load(f)     

#onehot encoder unpickling

with open('F:\MLProject\ML-GenAI\Data\onehot.pkl', 'rb') as f:
    onehot = pickle.load(f)    
##Standard scaler unpickling

with open('F:\MLProject\ML-GenAI\Data\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)   

print('models are loaded')

creditscore=st.slider('Credit Score', 0, 850, 600)
geography=st.selectbox('Geography', onehot.categories_[0])
gender=st.radio('gender',labelcoder.classes_)
age=st.slider('Age', 18, 100, 10)
tenure=st.slider('Tenure', 0, 10)
balance=st.text_input('Balance') 
numofproducts=st.slider('Num of Products', 1, 4, 1)
hascrcard=st.radio('Has Credit Card',[1,0])
active=st.radio('Active',[1,0])
estimatedsalary=st.slider('Estimated Salary', 0, 200000, 10000)

print('data is entered')



input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender': [labelcoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numofproducts],
    'HasCrCard': [hascrcard],
    'IsActiveMember': [active],
    'EstimatedSalary': [estimatedsalary]
})

print('geography is ', geography)
onehot_geo=onehot.transform([[geography]])
print('onehot_geo ----->>>>', onehot_geo)
#print('------->>>>>',onehot.get_feature_names(['Geography']))
onehot_geo_df=pd.DataFrame(onehot_geo, columns=onehot.get_feature_names_out(['Geography']))

input_data_df=pd.concat([input_data,onehot_geo_df],axis=1)

''' input_data = {     'CreditScore': 600,     'Geography': 'France',     'Gender': 'Male',    'Age': 40,     'Tenure': 3,
    'Balance': 60000,     'NumOfProducts': 2,     'HasCrCard': 1,     'IsActiveMember': 1,     'EstimatedSalary': 50000
}'''

input_data_scaled = scaler.transform(input_data_df)
print('before predictions')

predction = model.predict(input_data_scaled)
if predction[0][0] > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')