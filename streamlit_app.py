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

# Visualizations Section
st.subheader("Visualizations")

# 1. Employment Efficiency
st.write("### Employment Efficiency (Availed vs Offered)")
filtered_data['Efficiency'] = (filtered_data['Employment_Availed'] / filtered_data['Employment_offered']) * 100
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Efficiency'], label='Employment Efficiency (%)')
ax.set_xlabel('Year')
ax.set_ylabel('Efficiency (%)')
ax.legend()
st.pyplot(fig)

# 2. Yield vs MSP
st.write("### Yield vs Minimum Support Price (MSP)")
fig, ax = plt.subplots()
ax.scatter(filtered_data['Yield_(kg/Ha)'], filtered_data['MSP'])
ax.set_xlabel('Yield (kg/Ha)')
ax.set_ylabel('MSP (₹)')
st.pyplot(fig)

# 3. Rainfall vs Crop Production
st.write("### Rainfall vs Crop Production")
fig, ax = plt.subplots()
sns.regplot(x='Annual_rainfall', y='Production_(in_Tonnes)', data=filtered_data, ax=ax)
ax.set_xlabel('Annual Rainfall (mm)')
ax.set_ylabel('Production (Tonnes)')
st.pyplot(fig)

# 4. Time Series of Key Variables
st.write("### Time Series Analysis of Key Variables")
fig, ax = plt.subplots()
ax.plot(filtered_data['year'], filtered_data['Rural_Population'], label='Rural Population', color='blue')
ax.plot(filtered_data['year'], filtered_data['Employment_demanded'], label='Employment Demanded', color='green')
ax.plot(filtered_data['year'], filtered_data['WPI'], label='Wholesale Price Index (WPI)', color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Values')
ax.legend()
st.pyplot(fig)

# 5. Top States by Crop Production
st.write("### Top States by Crop Production")
top_producing_states = data.groupby('State_x')['Production_(in_Tonnes)'].sum().nlargest(10).reset_index()
fig, ax = plt.subplots()
sns.barplot(x='Production_(in_Tonnes)', y='State_x', data=top_producing_states, ax=ax)
ax.set_xlabel('Production (Tonnes)')
ax.set_ylabel('State')
st.pyplot(fig)

# Statistical Summary Section
st.subheader("Statistical Analysis")

# Summary statistics
st.write("### Summary Statistics")
st.write(filtered_data.describe())

# Correlation heatmap
st.write("### Correlation Heatmap")
correlation_cols = ['Rural_Population', 'No_of_Registered', 'Employment_demanded', 'Employment_offered',
                    'Employment_Availed', 'Area_(in_Ha)', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)',
                    'MSP', 'Annual_rainfall', 'WPI']

correlation_data = filtered_data[correlation_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Conclusion Section
st.write("### Conclusion")
st.write("This enhanced analysis provides deeper insights into employment, crop production, and agricultural trends over time, focusing on factors like efficiency, rainfall impact, and more.")

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
  
 
