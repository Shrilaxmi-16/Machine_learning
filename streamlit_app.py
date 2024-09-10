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
  data= pd.read_csv('https://raw.githubusercontent.com/Shrilaxmi-16/MLOps-shri/main/unique_states_crops.csv')
  data


# State selection from user input
st.sidebar.header("Select State")
selected_state = st.sidebar.selectbox("Select the state", data["State"].unique())

# Filter data for the selected state
state_data = data[data["State"] == selected_state]

# 1. Display state's all information
st.header(f"State Data for {selected_state}")
st.write(state_data)

# 2. Summary statistics
st.header(f"Summary Statistics for {selected_state}")
st.write(state_data.describe())

# 3. Normality Test (QQ Plot)
st.subheader("Normality Test (QQ Plot)")

def qq_plot(column):
    fig, ax = plt.subplots()
    stats.probplot(state_data[column], dist="norm", plot=ax)
    plt.title(f"QQ Plot for {column}")
    st.pyplot(fig)

# Select a column for normality testing
numeric_columns = state_data.select_dtypes(include=['int64', 'float64']).columns
selected_column = st.selectbox("Select a column for QQ plot", numeric_columns)
qq_plot(selected_column)

# 4. Spearman Correlation Test (Handle Non-Numeric Data and NaN)
st.subheader("Spearman Correlation Matrix")

# Select only numeric columns for correlation
numeric_data = state_data.select_dtypes(include=['float64', 'int64'])

# Fill missing values with 0 or any appropriate value
numeric_data = numeric_data.fillna(0)

# Compute Spearman correlation
if not numeric_data.empty:
    spearman_corr = numeric_data.corr(method='spearman')
    st.write(spearman_corr)
else:
    st.write("No numeric data available for correlation.")

# 5. Visualizing trends over the years for various metrics
st.header(f"Yearly Analysis for {selected_state}")

# MGNREGA Demand over the years
st.subheader(f"MGNREGA Demand Over the Years for {selected_state}")
mgnrega_demand = state_data.groupby('year')['Employment_demanded'].sum().reset_index()

line_chart_mgnrega = alt.Chart(mgnrega_demand).mark_line().encode(
    x='year',
    y='Employment_demanded'
).properties(
    width=600,
    height=400
)
st.altair_chart(line_chart_mgnrega)

# Production of the state each year
st.subheader(f"Production Over the Years for {selected_state}")
production_data = state_data.groupby('year')['Production_(in_Tonnes)'].sum().reset_index()

line_chart_production = alt.Chart(production_data).mark_line().encode(
    x='year',
    y='Production_(in_Tonnes)'
).properties(
    width=600,
    height=400
)
st.altair_chart(line_chart_production)

# Rainfall of the state each year
st.subheader(f"Rainfall Over the Years for {selected_state}")
rainfall_data = state_data.groupby('year')['Annual_rainfall'].sum().reset_index()

line_chart_rainfall = alt.Chart(rainfall_data).mark_line().encode(
    x='year',
    y='Annual_rainfall'
).properties(
    width=600,
    height=400
)
st.altair_chart(line_chart_rainfall)

# Adjusted MSP over the years
st.subheader(f"Adjusted MSP Over the Years for {selected_state}")
msp_data = state_data.groupby('year')['MSP'].sum().reset_index()

line_chart_msp = alt.Chart(msp_data).mark_line().encode(
    x='year',
    y='MSP'
).properties(
    width=600,
    height=400
)
st.altair_chart(line_chart_msp)

# 6. Multi-line chart: comparing all metrics over the years
st.subheader(f"Comparison of Various Metrics Over the Years for {selected_state}")

# Combine data for all metrics
combined_data = state_data.groupby('year').agg({
    'Employment_demanded': 'sum',
    'Production_(in_Tonnes)': 'sum',
    'Annual_rainfall': 'sum',
    'MSP': 'sum'
}).reset_index()

# Create a multi-line chart
multi_line_chart = alt.Chart(combined_data).transform_fold(
    ['Employment_demanded', 'Production_(in_Tonnes)', 'Annual_rainfall', 'MSP'],
    as_=['Metric', 'Value']
).mark_line().encode(
    x='Year:O',
    y='Value:Q',
    color='Metric:N'
).properties(
    width=800,
    height=400
)

st.altair_chart(multi_line_chart)
