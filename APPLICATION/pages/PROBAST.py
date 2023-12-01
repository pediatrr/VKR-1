import pandas as pd
import numpy as np
import streamlit as st
import matplotlib as plt
from st_aggrid import AgGrid
import pandas as pd
st.set_page_config(page_title="PROBAST", page_icon="📈")
st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.title('PROBAST')
st.header('PROBAST CHECKER')
def probast():
    df_2 = pd.read_excel("12874_2023_1849_MOESM2_ESM.xlsx",sheet_name='PROBAST summary')
    #np.random.seed(19680801)
    #data_0 = np.random.randn(5, 4)
    #data_0 = pd.DataFrame(data_0).style.background_gradient(cmap='plasma')
    #st.dataframe(data_0)
    df_2
    AgGrid(df_2)
    st.button("Reset", type="primary")
probast()
#https://stackoverflow.com/questions/74355967/customize-the-sidebar-pages-in-streamlit-to-have-sections
#https://github.com/PablocFonseca/streamlit-aggrid
