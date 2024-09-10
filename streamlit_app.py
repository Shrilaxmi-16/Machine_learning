import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


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

# Adding year filter as dropdown list
selected_year = st.sidebar.multiselect('Select Year(s)', data['year'].unique(), default=data['year'].unique())

# Filter data based on sidebar selections and selected years
filtered_data = data[(data['State_x'] == selected_state) & 
                     (data['Crop'] == selected_crop) &
                     (data['year'].isin(selected_year))]

# Data Analysis Section
st.header(f"Data Analysis for {selected_state} and Crop: {selected_crop} for Selected Years")

# Show filtered data
st.write(filtered_data)

# 1. Trend Analysis Section
st.subheader("Trend Analysis")

st.write("### Employment Trends Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Employment_demanded'], label="Employment Demanded")
ax.plot(filtered_data['year'], filtered_data['Employment_offered'], label="Employment Offered")
ax.plot(filtered_data['year'], filtered_data['Employment_Availed'], label="Employment Availed")
ax.set_xlabel('Year')
ax.set_ylabel('Number of People')
ax.legend()
st.pyplot(fig)

st.write("### Crop Production Trends Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Production_(in_Tonnes)'], label="Production (Tonnes)")
ax.set_xlabel('Year')
ax.set_ylabel('Production (Tonnes)')
ax.legend()
st.pyplot(fig)

st.write("### Rainfall Trends Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Annual_rainfall'], label="Annual Rainfall")
ax.set_xlabel('Year')
ax.set_ylabel('Rainfall (mm)')
ax.legend()
st.pyplot(fig)

# 2. Comparative Analysis Section
st.subheader("Comparative Analysis Across States")

metric = st.selectbox("Select a metric to compare across states", ['Production_(in_Tonnes)', 'Yield_(kg/Ha)', 'MSP', 'Employment_demanded'])
comp_data = data.groupby('State_x')[[metric]].mean().reset_index()

st.write(f"### Comparison of {metric} across states")
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='State_x', y=metric, data=comp_data, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# 3. Top N Analysis Section
st.subheader("Top N States or Crops")

top_n = st.slider("Select Top N", 1, 10, 5)
top_metric = st.selectbox("Select a metric for ranking", ['Production_(in_Tonnes)', 'Yield_(kg/Ha)', 'Employment_demanded'])

st.write(f"### Top {top_n} States Based on {top_metric}")
top_states = data.groupby('State_x')[[top_metric]].mean().nlargest(top_n, top_metric).reset_index()

fig, ax = plt.subplots()
sns.barplot(x='State_x', y=top_metric, data=top_states, ax=ax)
ax.set_xlabel('State')
ax.set_ylabel(top_metric)
st.pyplot(fig)

# 4. Advanced Correlation Analysis Section
st.subheader("Advanced Correlation Analysis")

# Pair Plot for selected metrics
pairplot_cols = ['Rural_Population', 'Employment_demanded', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)', 'Annual_rainfall']
st.write("### Pairplot for Key Metrics")
sns.pairplot(data[pairplot_cols])
st.pyplot()

# Correlation heatmap
st.write("### Correlation Heatmap of the Full Dataset")
correlation_cols = ['Rural_Population', 'No_of_Registered', 'Employment_demanded', 'Employment_offered',
                    'Employment_Availed', 'Area_(in_Ha)', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)',
                    'MSP', 'Annual_rainfall', 'WPI']
correlation_data = data[correlation_cols].corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 5. Regression Model to Predict Crop Yield
st.subheader("Prediction Model for Crop Yield")

# Features for prediction
features = ['Area_(in_Ha)', 'Production_(in_Tonnes)', 'Annual_rainfall', 'WPI']
X = data[features]
y = data['Yield_(kg/Ha)']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Prediction Section
st.write("### Predict Yield Based on Input Features")

input_area = st.number_input("Enter Area (in Ha)", min_value=0.0, value=1000.0)
input_production = st.number_input("Enter Production (in Tonnes)", min_value=0.0, value=500.0)
input_rainfall = st.number_input("Enter Annual Rainfall (in mm)", min_value=0.0, value=10.0)
input_wpi = st.number_input("Enter WPI", min_value=0.0, value=100.0)

predicted_yield = model.predict([[input_area, input_production, input_rainfall, input_wpi]])[0]
st.write(f"Predicted Yield (kg/Ha): {predicted_yield:.2f}")

# Conclusion Section
st.write("### Conclusion")
st.write("This extended analysis provides more insights into the trends, comparisons, and relationships in the agricultural and employment data.")
