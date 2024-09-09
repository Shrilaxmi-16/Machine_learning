import streamlit as st
import pandas as pd
st.title('🤖 Machine Learning App')

st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  df= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  df
with st.expander('Data Visualization'):
  st.write('## Crop Production per Year')
  source = df
  st.bar_chart(source, x=State_x, y=Production(in_Tonnes), color=Crop, horizontal=True)
  
