# -*- coding: utf-8 -*-

import streamlit as st
import requests
import joblib
import numpy as np
from streamlit_lottie import st_lottie
st.set_page_config(page_title="stroke_detection",page_icon='::star::');

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200 :
     return None
 
    return r.json()
     
def prepare_input_data_for_model(Age,Hypertension,Heart_disease,Bmi,Avg_glucose):
    if Hypertension=='High':
       h=1
    else:
        h=0
    if Heart_disease =='yes,he/she suffers from heart_disease':
       heart=1
    else:
        heart=0
        
    S=[Age,h,heart,Bmi,Avg_glucose]
    sample = np.array(S).reshape(-1,len(S))
    
    return sample
    
 
loaded_model=joblib.load(open("Drug_Model",'rb'))


st.write('# Stroke Detection')
lottie_link="https://assets6.lottiefiles.com/packages/lf20_ggwvc2su.json"
animation = load_lottie(lottie_link)
st.write("----------")
st.subheader("enter the person details to predict the stroke addiction")
with st.container():
    right_column,left_column=st.columns(2)
    
    with right_column:
        Age=st.number_input('Age : ')
        Hypertension=st.radio('Hypertension :',['Low','High'])
        Heart_disease=st.radio('Heart disease :',['yes,he/she suffers from heart_disease','NO,he/she does not'])
        Bmi=st.number_input('Bmi : ', min_value=0.0)
        Avg_glucose=st.number_input('Average of glucose :')
        
        sample = prepare_input_data_for_model(Age,Hypertension,Heart_disease,Bmi,Avg_glucose)
    with left_column:
      st_lottie(animation,speed=1,height=400,key="initial")
    if st.button('predict'):  
            y_predict1 =loaded_model.predict(sample)
            if y_predict1==0 :
                st.write('not addicted')
            else:
                st.write('addicted')