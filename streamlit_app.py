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
  df= pd.read_csv('https://raw.githubusercontent.com/Shrilaxmi-16/MLOps-shri/main/unique_states_crops.csv')
  df

# Function to plot QQ plot
def qq_plot(data, column):
    stats.probplot(df[column], dist="norm", plot=plt)
    st.pyplot()

# Function to calculate Spearman correlation
def spearman_correlation(df, columns):
    correlation_matrix = df[columns].corr(method='spearman')
    st.write(correlation_matrix)

# Function to plot MGNREGA Demand
def plot_mgnrega_demand(df, state):
    state_data = df[df['State'] == state]
    st.line_chart(state_data[['year', 'Employment_demanded']].set_index('year'))

# Main Streamlit app
def main():
    st.title('Crop Data Analysis by State')


    # Select state from the dropdown
    state = st.selectbox('Select State', df['State'].unique())

    # Display all information for the selected state
    state_data = df[df['State'] == state]
    st.subheader(f'Data for {state}')
    st.write(state_data)

    # Summary statistics
    st.subheader('Summary Statistics')
    st.write(state_data.describe())

    # Normality test using QQ plot
    st.subheader('Normality Test (QQ Plot)')
    column_to_test = st.selectbox('Select column for normality test', state_data.columns)
    qq_plot(state_data, column_to_test)

    # Spearman Correlation
    st.subheader('Spearman Correlation Test')
    columns_to_correlate = st.multiselect('Select columns for correlation', state_data.columns)
    if len(columns_to_correlate) > 1:
        spearman_correlation(state_data, columns_to_correlate)

    # Plot MGNREGA demand for the state
    st.subheader('MGNREGA Demand Over the Years')
    plot_mgnrega_demand(df, state)

    # Production data over the years
    st.subheader('Production Over the Years')
    production_column = 'Production_(in_Tonnes)'
    if production_column in state_data.columns:
        st.line_chart(state_data[['year', production_column]].set_index('year'))

    # Rainfall data over the years
    st.subheader('Rainfall Over the Years')
    rainfall_column = 'Annual_rainfall'
    if rainfall_column in state_data.columns:
        st.line_chart(state_data[['year', rainfall_column]].set_index('year'))

    # Adjusted MSP (simple example of calculating it)
    st.subheader('Adjusted Minimum Support Price (MSP)')
    # Assuming there's a column in your data related to MSP
    if 'MSP' in state_data.columns:
        adjusted_msp = state_data['MSP'] * 1.05  # Example adjustment of 5%
        st.write(f'Adjusted MSP for {state}:')
        st.write(adjusted_msp)

if __name__ == '__main__':
    main()
