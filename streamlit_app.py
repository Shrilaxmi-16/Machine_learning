import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')

# Title for the app
st.title("Data Visualization Dashboard")

# Display the dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Sidebar for selecting options
st.sidebar.title("Visualization Options")
plot_type = st.sidebar.selectbox("Choose Plot Type", ["Line Plot", "Bar Plot", "Scatter Plot"])

# Select columns for x and y axis
x_column = st.sidebar.selectbox("Select X-axis Column", data.columns)
y_column = st.sidebar.selectbox("Select Y-axis Column", data.columns)

# Plot based on selection
if plot_type == "Line Plot":
    st.subheader(f"Line Plot of {y_column} vs {x_column}")
    st.line_chart(data[[x_column, y_column]].set_index(x_column))
elif plot_type == "Bar Plot":
    st.subheader(f"Bar Plot of {y_column} vs {x_column}")
    st.bar_chart(data[[x_column, y_column]].set_index(x_column))
elif plot_type == "Scatter Plot":
    st.subheader(f"Scatter Plot of {y_column} vs {x_column}")
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)
