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

# Histogram of Crop Production
st.write("#### Distribution of Crop Production (in Tonnes)")
fig, ax = plt.subplots()
ax.hist(filtered_data['Production_(in_Tonnes)'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Production (Tonnes)')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Density Plot of Annual Rainfall
st.write("#### Distribution of Annual Rainfall")
fig, ax = plt.subplots()
sns.kdeplot(filtered_data['Annual_rainfall'], shade=True, color="green", ax=ax)
ax.set_xlabel('Annual Rainfall (mm)')
st.pyplot(fig)

# 3. Time Series Analysis
st.subheader("Time Series Analysis")

# Plot with Moving Averages for Production
st.write("### Time Series: Crop Production with Moving Average")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Production_(in_Tonnes)'], label='Production (Tonnes)', color='blue')
ax.plot(filtered_data['year'], filtered_data['Production_(in_Tonnes)'].rolling(window=3).mean(), label='3-Year Moving Average', linestyle='--', color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Production (Tonnes)')
ax.legend()
st.pyplot(fig)

# Rolling Sum for Employment Demanded
st.write("### Time Series: Employment Demanded with Rolling Sum")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Employment_demanded'], label='Employment Demanded', color='purple')
ax.plot(filtered_data['year'], filtered_data['Employment_demanded'].rolling(window=3).sum(), label='3-Year Rolling Sum', linestyle='--', color='orange')
ax.set_xlabel('Year')
ax.set_ylabel('Employment Demanded')
ax.legend()
st.pyplot(fig)

# 4. Correlation and Pair Plot Analysis
st.subheader("Correlation and Relationship Analysis")

# Correlation Heatmap
st.write("### Correlation Heatmap")
correlation_cols = ['Rural_Population', 'No_of_Registered', 'Employment_demanded', 'Employment_offered',
                    'Employment_Availed', 'Area_(in_Ha)', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)',
                    'MSP', 'Annual_rainfall', 'WPI']
correlation_data = filtered_data[correlation_cols].corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Pair Plot for Key Metrics
st.write("### Pairplot for Key Metrics")
pairplot_cols = ['Employment_demanded', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)', 'Annual_rainfall']
sns.pairplot(filtered_data[pairplot_cols])
st.pyplot()

# 5. Relationship Between Agricultural Productivity and Rural Employment Demand
st.subheader("Relationship Between Agricultural Productivity and Rural Employment Demand")

# Scatter plot with regression line for Crop Production vs Employment Demanded
st.write("### Crop Production vs Employment Demanded")
fig, ax = plt.subplots()
sns.regplot(x='Production_(in_Tonnes)', y='Employment_demanded', data=filtered_data, ax=ax, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
ax.set_xlabel('Production (in Tonnes)')
ax.set_ylabel('Employment Demanded')
st.pyplot(fig)

# Scatter plot with regression line for Rainfall vs Employment Demanded
st.write("### Annual Rainfall vs Employment Demanded")
fig, ax = plt.subplots()
sns.regplot(x='Annual_rainfall', y='Employment_demanded', data=filtered_data, ax=ax, scatter_kws={'color': 'green'}, line_kws={'color': 'red'})
ax.set_xlabel('Annual Rainfall (mm)')
ax.set_ylabel('Employment Demanded')
st.pyplot(fig)

# Scatter plot with regression line for Yield vs Employment Demanded
st.write("### Crop Yield vs Employment Demanded")
fig, ax = plt.subplots()
sns.regplot(x='Yield_(kg/Ha)', y='Employment_demanded', data=filtered_data, ax=ax, scatter_kws={'color': 'purple'}, line_kws={'color': 'red'})
ax.set_xlabel('Yield (kg/Ha)')
ax.set_ylabel('Employment Demanded')
st.pyplot(fig)

# Display correlation between Employment Demanded and key agricultural metrics
st.write("### Correlation Between Employment Demanded and Agricultural Metrics")
corr_employment_production = filtered_data[['Employment_demanded', 'Production_(in_Tonnes)', 'Annual_rainfall', 'Yield_(kg/Ha)']].corr()
st.write(corr_employment_production)

# 6. Top N Analysis
st.subheader("Top N Analysis")
top_n = st.slider("Select Top N", 1, 10, 5)
top_metric = st.selectbox("Select a metric for ranking", ['Production_(in_Tonnes)', 'Yield_(kg/Ha)', 'Employment_demanded'])

st.write(f"### Top {top_n} States Based on {top_metric}")
top_states = data.groupby('State_x')[[top_metric]].mean().nlargest(top_n, top_metric).reset_index()

fig, ax = plt.subplots()
sns.barplot(x='State_x', y=top_metric, data=top_states, ax=ax)
ax.set_xlabel('State')
ax.set_ylabel(top_metric)
st.pyplot(fig)

# 7. Prediction Model for Crop Yield
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
st.write("This detailed analysis explores the relationship between agricultural productivity and rural employment demand under MGNREGA.")
