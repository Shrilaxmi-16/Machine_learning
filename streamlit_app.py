import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

st.title("Agricultural and Employment Data Analysis")

st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  data= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  data

# Sidebar filters
st.sidebar.header("Filter Options")
selected_state = st.sidebar.selectbox('Select State', data['State_x'].unique())
selected_crop = st.sidebar.selectbox('Select Crop', data['Crop'].unique())

# Filter data based on sidebar selections
filtered_data = data[(data['State_x'] == selected_state) & (data['Crop'] == selected_crop)]

# Data Analysis Section
st.header(f"Data Analysis for {selected_state} and Crop: {selected_crop}")

# Show filtered data
st.write(filtered_data)

# Visualization Section
st.subheader("Visualizations")

# Line plot for employment data
st.write("### Employment Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Employment_demanded'], label="Employment Demanded")
ax.plot(filtered_data['year'], filtered_data['Employment_offered'], label="Employment Offered")
ax.plot(filtered_data['year'], filtered_data['Employment_Availed'], label="Employment Availed")
ax.set_xlabel('Year')
ax.set_ylabel('Number of People')
ax.legend()
st.pyplot(fig)

# Bar chart for crop production and area
st.write("### Crop Production and Area")
fig, ax = plt.subplots()
ax.bar(filtered_data['year'], filtered_data['Production_(in_Tonnes)'], label='Production (Tonnes)', alpha=0.6)
ax.bar(filtered_data['year'], filtered_data['Area_(in_Ha)'], label='Area (Ha)', alpha=0.6)
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

# Correlation heatmap
st.write("### Correlation Heatmap")
correlation_cols = ['Rural_Population', 'No_of_Registered', 'Employment_demanded', 'Employment_offered',
                    'Employment_Availed', 'Area_(in_Ha)', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)',
                    'MSP', 'Annual_rainfall', 'WPI']

correlation_data = filtered_data[correlation_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Statistical Summary Section
st.subheader("Statistical Analysis")

st.write("### Summary Statistics")
st.write(filtered_data.describe())

st.write("### Yield vs Rainfall Scatter Plot")
fig, ax = plt.subplots()
ax.scatter(filtered_data['Annual_rainfall'], filtered_data['Yield_(kg/Ha)'])
ax.set_xlabel('Annual Rainfall')
ax.set_ylabel('Yield (kg/Ha)')
st.pyplot(fig)

# Conclusion Section
st.write("### Conclusion")
st.write("This analysis provides insights into employment and agricultural data trends over time for the selected state and crop.")
