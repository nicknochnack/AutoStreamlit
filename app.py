import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report


def load_data(file): 
    global df 
    df = pd.read_csv(file, index_col=None)
    return df 

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoNickML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = load_data(file)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    pass

if choice == "Download": 
    pass 