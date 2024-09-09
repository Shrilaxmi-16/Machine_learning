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

st.sidebar.title("Visualization Options")
# 1. Descriptive Statistics
st.sidebar.subheader("Descriptive Statistics")
if st.sidebar.checkbox("Show Descriptive Statistics"):
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

# 2. Correlation Matrix (handling only numerical columns)
st.sidebar.subheader("Correlation Matrix")
if st.sidebar.checkbox("Show Correlation Matrix"):
    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_data.corr()
    st.write(corr_matrix)
    corr_chart = alt.Chart(corr_matrix.reset_index().melt('index')).mark_rect().encode(
        alt.X('variable:N', title='Columns'),
        alt.Y('index:N', title='Columns'),
        alt.Color('value:Q', scale=alt.Scale(scheme='blueorange'))
    )
    st.altair_chart(corr_chart, use_container_width=True)

# 3. Distribution of individual columns
st.sidebar.subheader("Column Distribution")
column_to_analyze = st.sidebar.selectbox("Select Column for Distribution", numeric_data.columns)
if st.sidebar.checkbox(f"Show Distribution for {column_to_analyze}"):
    st.subheader(f"Distribution of {column_to_analyze}")
    hist_chart = alt.Chart(data).mark_bar().encode(
        alt.X(column_to_analyze, bin=True),
        y='count()'
    )
    st.altair_chart(hist_chart, use_container_width=True)


# 5. Pairplot (Scatter Plot Matrix)
st.sidebar.subheader("Pairplot")
if st.sidebar.checkbox("Show Pairplot"):
    st.subheader("Pairplot of Numeric Variables")
    pairplot_fig = sns.pairplot(data[numeric_data.columns])
    st.pyplot(pairplot_fig)

# 6. Box Plot
st.sidebar.subheader("Box Plot")
box_x_column = st.sidebar.selectbox("Select X-axis Column for Box Plot", data.columns)
box_y_column = st.sidebar.selectbox("Select Y-axis Column for Box Plot", numeric_data.columns)
if st.sidebar.checkbox("Show Box Plot"):
    st.subheader(f"Box Plot of {box_y_column} grouped by {box_x_column}")
    box_plot = sns.boxplot(x=box_x_column, y=box_y_column, data=data)
    st.pyplot(plt.gcf())  # Render current figure

# 7. Violin Plot
st.sidebar.subheader("Violin Plot")
if st.sidebar.checkbox("Show Violin Plot"):
    st.subheader(f"Violin Plot of {box_y_column} grouped by {box_x_column}")
    violin_plot = sns.violinplot(x=box_x_column, y=box_y_column, data=data)
    st.pyplot(plt.gcf())  # Render current figure

# 8. Pie Chart for categorical data
st.sidebar.subheader("Pie Chart")
pie_column = st.sidebar.selectbox("Select Column for Pie Chart", data.select_dtypes(include=['object']).columns)
if st.sidebar.checkbox(f"Show Pie Chart for {pie_column}"):
    st.subheader(f"Pie Chart for {pie_column}")
    pie_data = data[pie_column].value_counts().reset_index()
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field=pie_column, type="quantitative"),
        color=alt.Color(field='index', type="nominal"),
    )
    st.altair_chart(pie_chart, use_container_width=True)
 
