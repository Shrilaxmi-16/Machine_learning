import streamlit as st
import pandas as pd
st.title('🤖 Machine Learning App')

st.info('This App is for machine learning analysis')
df= pd.read_csv("C:\Users\Admin\Downloads\dataset-final.csv")
df
