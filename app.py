import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

columns= [['age/num', 'swirl/cat', 'airbleb/cat', 'contusion/cat', 'edh/cat', 'v1/num', 'location/cat']]



model = joblib.load('hematoma.pkl')


st.title('Prediction of EDH expansion based on patient data :brain:')

age = st.number_input("Enter the patient age")
airbleb = st.selectbox('Does the patient have Intra-hematoma airbleb? 0: No     1: yes', [0,1])
swirl_sign = st.selectbox('Does the patient have Swirl Sign? 0: No     1: yes', [0,1])
edh_volume = st.number_input("Enter initial EDH volume")
contusion = st.selectbox('Does the patient have contusion? 0: No     1: yes', [0,1])
other_side_extra_axial_hematoma = st.selectbox('Does the patient have other side extra-axial hematoma? 0: No     1: yes', [0,1])
location= st.selectbox('Where is the location of the hematoma? 1:Parietal, 2:Occipital, 3:Posterior fossa, 4:Frontotemporal, 5:Frontotemporoparietal, 6:Others',
                       [1,2,3,4,5,6])


def predict(): 
    row = np.array([age,swirl_sign, airbleb,contusion,other_side_extra_axial_hematoma,edh_volume,location])
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] >= 0.20: 
        print('The EDH volume is more likely to be increased :heavy_exclamation_mark:')
    elif prediction[0] < 0.20:
        print('The EDH volume is more likely not to be increased :white_check_mark:')
        

trigger = st.button('Predict', on_click=predict)