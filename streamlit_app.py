import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title('🤖 Machine Learning App')

st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  data= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  data
st.write('## Data Visualization')
st.title("Agricultural and Employment Data Analysis")

# Display the raw data
if st.checkbox("Show Raw Data"):
    st.write(data)

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

# Sidebar for options
st.sidebar.title("Visualization Options")

# 1. Descriptive Statistics
st.sidebar.subheader("Descriptive Statistics")
if st.sidebar.checkbox("Show Descriptive Statistics"):
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

# 2. Pandas Profiling Report
st.sidebar.subheader("Pandas Profiling Report")
if st.sidebar.checkbox("Generate Pandas Profiling Report"):
    st.subheader("Exploratory Data Analysis Report")
    profile = ProfileReport(data, minimal=True)
    st_profile_report(profile)

# 3. Histogram with Matplotlib
st.sidebar.subheader("Matplotlib Histogram")
hist_column = st.sidebar.selectbox("Select Column for Histogram", data.select_dtypes(include=['int64', 'float64']).columns)
if st.sidebar.checkbox(f"Show Histogram for {hist_column}"):
    st.subheader(f"Histogram of {hist_column}")
    fig, ax = plt.subplots()
    ax.hist(data[hist_column], bins=30, edgecolor='black')
    plt.title(f'Histogram of {hist_column}')
    st.pyplot(fig)

# 4. Scatter Plot with Matplotlib
st.sidebar.subheader("Matplotlib Scatter Plot")
scatter_x_column = st.sidebar.selectbox("Select X-axis for Scatter Plot", data.select_dtypes(include=['int64', 'float64']).columns)
scatter_y_column = st.sidebar.selectbox("Select Y-axis for Scatter Plot", data.select_dtypes(include=['int64', 'float64']).columns)
if st.sidebar.checkbox(f"Show Scatter Plot of {scatter_x_column} vs {scatter_y_column}"):
    st.subheader(f"Scatter Plot of {scatter_y_column} vs {scatter_x_column}")
    fig, ax = plt.subplots()
    ax.scatter(data[scatter_x_column], data[scatter_y_column], c='blue', alpha=0.5)
    ax.set_xlabel(scatter_x_column)
    ax.set_ylabel(scatter_y_column)
    plt.title(f'Scatter Plot of {scatter_x_column} vs {scatter_y_column}')
    st.pyplot(fig)

# 5. Bar Plot using Altair
st.sidebar.subheader("Altair Bar Plot")
bar_column = st.sidebar.selectbox("Select Categorical Column for Bar Plot", data.select_dtypes(include=['object']).columns)
if st.sidebar.checkbox(f"Show Bar Plot for {bar_column}"):
    st.subheader(f"Bar Plot for {bar_column}")
    bar_data = data[bar_column].value_counts().reset_index()
    bar_chart = alt.Chart(bar_data).mark_bar().encode(
        x='index',
        y=bar_column
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(bar_chart)

# 6. Line Plot using Altair
st.sidebar.subheader("Altair Line Plot")
line_x_column = st.sidebar.selectbox("Select X-axis for Line Plot", data.select_dtypes(include=['int64', 'float64']).columns)
line_y_column = st.sidebar.selectbox("Select Y-axis for Line Plot", data.select_dtypes(include=['int64', 'float64']).columns)
if st.sidebar.checkbox(f"Show Line Plot of {line_x_column} vs {line_y_column}"):
    st.subheader(f"Line Plot of {line_x_column} vs {line_y_column}")
    line_chart = alt.Chart(data).mark_line().encode(
        x=line_x_column,
        y=line_y_column
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(line_chart)

# 7. Box Plot using Altair
st.sidebar.subheader("Altair Box Plot")
box_x_column = st.sidebar.selectbox("Select X-axis for Box Plot", data.columns)
box_y_column = st.sidebar.selectbox("Select Y-axis for Box Plot", data.select_dtypes(include=['int64', 'float64']).columns)
if st.sidebar.checkbox(f"Show Box Plot of {box_y_column} grouped by {box_x_column}"):
    st.subheader(f"Box Plot of {box_y_column} grouped by {box_x_column}")
    box_plot = alt.Chart(data).mark_boxplot().encode(
        x=box_x_column,
        y=box_y_column
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(box_plot)
  
 
