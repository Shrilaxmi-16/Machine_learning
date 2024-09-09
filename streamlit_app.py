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

# Sidebar for visualization options
st.sidebar.title("Visualization Options")

# 1. Descriptive Statistics
st.sidebar.subheader("Descriptive Statistics")
if st.sidebar.checkbox("Show Descriptive Statistics"):
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

# 2. Correlation Matrix (using Seaborn heatmap)
st.sidebar.subheader("Correlation Matrix")
if st.sidebar.checkbox("Show Correlation Matrix"):
    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# 3. Pairplot (Scatter Plot Matrix using Plotly)
st.sidebar.subheader("Pairplot")
if st.sidebar.checkbox("Show Pairplot"):
    st.subheader("Pairplot of Numeric Variables")
    pairplot_fig = px.scatter_matrix(data, dimensions=numeric_data.columns)
    st.plotly_chart(pairplot_fig)

# 4. Box Plot (using Plotly)
st.sidebar.subheader("Box Plot")
box_x_column = st.sidebar.selectbox("Select X-axis Column for Box Plot", data.columns)
box_y_column = st.sidebar.selectbox("Select Y-axis Column for Box Plot", numeric_data.columns)
if st.sidebar.checkbox("Show Box Plot"):
    st.subheader(f"Box Plot of {box_y_column} grouped by {box_x_column}")
    box_plot = px.box(data, x=box_x_column, y=box_y_column, points="all")
    st.plotly_chart(box_plot)

# 5. Violin Plot (using Plotly)
st.sidebar.subheader("Violin Plot")
if st.sidebar.checkbox("Show Violin Plot"):
    st.subheader(f"Violin Plot of {box_y_column} grouped by {box_x_column}")
    violin_plot = px.violin(data, x=box_x_column, y=box_y_column, box=True, points="all")
    st.plotly_chart(violin_plot)

# 6. Scatter Plot (using Plotly for interactive scatter plots)
st.sidebar.subheader("Scatter Plot")
scatter_x_column = st.sidebar.selectbox("Select X-axis Column for Scatter Plot", numeric_data.columns)
scatter_y_column = st.sidebar.selectbox("Select Y-axis Column for Scatter Plot", numeric_data.columns)
if st.sidebar.checkbox("Show Scatter Plot"):
    st.subheader(f"Scatter Plot of {scatter_y_column} vs {scatter_x_column}")
    scatter_plot = px.scatter(data, x=scatter_x_column, y=scatter_y_column, color=box_x_column)
    st.plotly_chart(scatter_plot)

# 7. Pie Chart for categorical data (using Plotly)
st.sidebar.subheader("Pie Chart")
pie_column = st.sidebar.selectbox("Select Column for Pie Chart", data.select_dtypes(include=['object']).columns)
if st.sidebar.checkbox(f"Show Pie Chart for {pie_column}"):
    st.subheader(f"Pie Chart for {pie_column}")
    pie_data = data[pie_column].value_counts().reset_index()
    pie_chart = px.pie(pie_data, values=pie_column, names='index')
    st.plotly_chart(pie_chart)
 
