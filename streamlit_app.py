import streamlit as st
import pandas as pd
import altair as alt
import numpy as np


st.title('🤖 Machine Learning App for employement demand')

st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  data= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  data
st.write('## Data Visualization')
st.sidebar.title('Visualization Options')
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
  
 
