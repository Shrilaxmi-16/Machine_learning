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
selected_year = st.sidebar.multiselect('Select Year(s)', data['year'].unique(), default=data['year'].unique())

# Filter data based on sidebar selections
filtered_data = data[(data['State_x'] == selected_state) & 
                     (data['Crop'] == selected_crop) &
                     (data['year'].isin(selected_year))]

# Data Analysis Section
st.header(f"Detailed Data Analysis for {selected_state} and Crop: {selected_crop} for Selected Years")

# Show filtered data
st.write(filtered_data)

# 1. Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write("### Summary Statistics of the Filtered Data")
st.write(filtered_data.describe())

# 2. Distribution Analysis
st.subheader("Distribution Analysis")
st.write("### Distribution of Key Metrics")

# Boxplot for Employment Demanded, Offered, and Availed
st.write("#### Employment Demanded, Offered, and Availed")
fig, ax = plt.subplots()
sns.boxplot(data=filtered_data[['Employment_demanded', 'Employment_offered', 'Employment_Availed']], ax=ax)
st.pyplot(fig)

# 3. High Employment Demand Analysis
st.subheader("High Employment Demand Analysis")

# Determine high employment demand: Get the top N states for each year
top_n_employment = st.slider("Select the Top N States for Employment Demand", 1, 10, 5)

# Group by year and state, then get the top N for each year based on Employment Demanded
high_demand_data = data.groupby(['year', 'State_x'])[['Employment_demanded', 'Employment_offered', 'Yield_(kg/Ha)', 'Annual_rainfall']].mean().reset_index()

# Filter for top N states based on employment demand for each year
top_states_each_year = high_demand_data.groupby('year').apply(lambda x: x.nlargest(top_n_employment, 'Employment_demanded')).reset_index(drop=True)

# Display the top states with high employment demand
st.write(f"### Top {top_n_employment} States with Highest Employment Demand Each Year")
st.write(top_states_each_year)

# Plot Employment Demanded vs Offered for Top States
st.write("### Employment Demanded vs Offered for Top States with High Employment Demand")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Employment_demanded', y='Employment_offered', hue='State_x', data=top_states_each_year, ax=ax, s=100, palette='coolwarm')
ax.set_xlabel('Employment Demanded')
ax.set_ylabel('Employment Offered')
ax.set_title('Employment Demanded vs Employment Offered for Top States')
st.pyplot(fig)

# 4. Analysis of Agricultural Factors for High Employment Demand Years
st.subheader("Analysis of Agricultural Factors in High Employment Demand Years")

# Create scatter plots for Yield, Rainfall, and Employment Demanded for top states
st.write("### Crop Yield vs Employment Demanded in High Employment Demand Years")
fig, ax = plt.subplots()
sns.scatterplot(x='Yield_(kg/Ha)', y='Employment_demanded', hue='State_x', data=top_states_each_year, ax=ax, s=100, palette='viridis')
ax.set_xlabel('Crop Yield (kg/Ha)')
ax.set_ylabel('Employment Demanded')
st.pyplot(fig)

st.write("### Rainfall vs Employment Demanded in High Employment Demand Years")
fig, ax = plt.subplots()
sns.scatterplot(x='Annual_rainfall', y='Employment_demanded', hue='State_x', data=top_states_each_year, ax=ax, s=100, palette='coolwarm')
ax.set_xlabel('Annual Rainfall (mm)')
ax.set_ylabel('Employment Demanded')
st.pyplot(fig)

# Correlation between Employment Demanded and Agricultural Factors for top states in high demand years
st.write("### Correlation Between Employment Demanded and Agricultural Factors for High Employment Demand Years")
correlation_top_states = top_states_each_year[['Employment_demanded', 'Yield_(kg/Ha)', 'Annual_rainfall']].corr()
st.write(correlation_top_states)

# 5. Prediction Model for Crop Yield
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
st.write("This detailed analysis examines high employment demand years and evaluates the related agricultural and employment metrics.")
