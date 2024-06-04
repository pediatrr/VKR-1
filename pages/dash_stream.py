import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
from pycaret.datasets import get_data
from streamlit_plotly_events import plotly_events
import streamlit as st   
from menu import menu_with_redirect
# Перенаправить на app.py если вы не вошли в систему, в противном случае отобразить навигационное меню
menu_with_redirect()

         
# Инициализировать связь с py g walker
#init_streamlit_comm()
@st.cache_resource
def get_pyg_html(df: pd.DataFrame) -> str:
    html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
    return html
# Кэширование
@st.cache_data
def get_df() -> pd.DataFrame:
    return pd.read_csv("diabetes_prediction_dataset.csv")
df = get_df()

# Загрузить датасет
df = pd.read_csv('diabetes_prediction_dataset.csv')




        


        


    


        


    



