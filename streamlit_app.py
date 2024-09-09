import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
st.title('🤖 Machine Learning App')

st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  data= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  data
with st.expander('Data Visualization'):
  st.subheader("Dataset Preview")
  st.write(data.head())
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
# Sidebar for analysis options
st.sidebar.title("Statistical Analysis Options")

# 1. Descriptive Statistics
st.sidebar.subheader("Descriptive Statistics")
if st.sidebar.checkbox("Show Descriptive Statistics"):
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

# 2. Correlation Matrix
st.sidebar.subheader("Correlation Matrix")
if st.sidebar.checkbox("Show Correlation Matrix"):
    st.subheader("Correlation Matrix")
    corr_matrix = data.corr()
    st.write(corr_matrix)
    
    # Heatmap visualization using Altair
    corr_chart = alt.Chart(corr_matrix.reset_index().melt('index')).mark_rect().encode(
        alt.X('variable:N', title='Columns'),
        alt.Y('index:N', title='Columns'),
        alt.Color('value:Q', scale=alt.Scale(scheme='blueorange'))
    )
    st.altair_chart(corr_chart, use_container_width=True)

# 3. Distribution of individual columns
st.sidebar.subheader("Column Distribution")
column_to_analyze = st.sidebar.selectbox("Select Column for Distribution", data.select_dtypes(include=['int64', 'float64']).columns)
if st.sidebar.checkbox(f"Show Distribution for {column_to_analyze}"):
    st.subheader(f"Distribution of {column_to_analyze}")
    hist_chart = alt.Chart(data).mark_bar().encode(
        alt.X(column_to_analyze, bin=True),
        y='count()'
    )
    st.altair_chart(hist_chart, use_container_width=True)

# 4. Grouping and Aggregation
st.sidebar.subheader("Grouping and Aggregation")
group_column = st.sidebar.selectbox("Select Grouping Column", data.columns)
agg_column = st.sidebar.selectbox("Select Column to Aggregate", data.select_dtypes(include=['int64', 'float64']).columns)
agg_function = st.sidebar.selectbox("Select Aggregation Function", ['mean', 'sum', 'min', 'max', 'count'])

if st.sidebar.checkbox("Show Grouped Data"):
    st.subheader(f"Grouped Data by {group_column}")
    if agg_function == 'mean':
        grouped_data = data.groupby(group_column)[agg_column].mean()
    elif agg_function == 'sum':
        grouped_data = data.groupby(group_column)[agg_column].sum()
    elif agg_function == 'min':
        grouped_data = data.groupby(group_column)[agg_column].min()
    elif agg_function == 'max':
        grouped_data = data.groupby(group_column)[agg_column].max()
    elif agg_function == 'count':
        grouped_data = data.groupby(group_column)[agg_column].count()

    st.write(grouped_data)
