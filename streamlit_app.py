import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
st.title('🤖 Machine Learning App')

st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  df= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  df
with st.expander('Data Visualization'):
  st.subheader("Dataset Preview")
  st.write(df.head())

# Sidebar for selecting options
st.sidebar.title("Visualization Options")
plot_type = st.sidebar.selectbox("Choose Plot Type", ["Line Plot", "Bar Plot", "Histogram"])

# Select columns for x and y axis
x_column = st.sidebar.selectbox("Select X-axis Column", data.columns)
y_column = st.sidebar.selectbox("Select Y-axis Column", data.columns)

# Plot based on selection
if plot_type == "Line Plot":
    st.subheader(f"Line Plot of {y_column} vs {x_column}")
    line_chart = alt.Chart(data).mark_line().encode(
        x=x_column,
        y=y_column
    )
    st.altair_chart(line_chart, use_container_width=True)

elif plot_type == "Bar Plot":
    st.subheader(f"Bar Plot of {y_column} vs {x_column}")
    bar_chart = alt.Chart(data).mark_bar().encode(
        x=x_column,
        y=y_column
    )
    st.altair_chart(bar_chart, use_container_width=True)

elif plot_type == "Histogram":
    st.subheader(f"Histogram of {x_column}")
    hist_chart = alt.Chart(data).mark_bar().encode(
        alt.X(x_column, bin=True),
        y='count()'
    )
    st.altair_chart(hist_chart, use_container_width=True)


  
