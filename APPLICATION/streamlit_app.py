# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np

def create_st_button(link_text, link_url, st_col):
    st_col.markdown(f'<a href="{link_url}" target="_blank">{link_text}</a>', unsafe_allow_html=True)

def main():
    st.set_page_config(
    page_title="Welcome",
    page_icon="👋", layout='wide')
    st.header('Представления результатов медицинских моделей', divider='rainbow')
    st.image('phone.png')
    st.markdown("#### Эффективность моделей машинного обучения зависит не только от их технических характеристик, но и от того, насколько конечные пользователи могут правильно и верно интерпретировать результаты работы модели.")
    st.markdown ("- ### Контекст проблемы") 
    st.markdown ("- ### Цели и ожидания пользователей")
    st.markdown ("- ### Возможные риски и ограничения модели")  
    st.button("Reset", type="primary")

    database_link_dict = {
        "Interpretable Machine Learning": "https://christophm.github.io/interpretable-ml-book/",
        "Interpretable Machine Learning with Python": "https://www.google.com/books/edition/Interpretable_Machine_Learning_with_Pyth/iIwmEAAAQBAJ?hl=en",
        }

    st.sidebar.markdown("## Литература для понимания")
    for link_text, link_url in database_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    community_link_dict = {
        "SHAP": "https://github.com/shap/shap",
        "LIME": "https://github.com/marcotcr/lime",
        "PDPbox": "https://github.com/SauceCat/PDPbox",
        "InterpretML":"https://interpret.ml"
    }

    st.sidebar.markdown("## Подробно о методах")
    for link_text, link_url in community_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    software_link_dict = {
        "Pandas": "https://pandas.pydata.org",
        "NumPy": "https://numpy.org",
        "SciPy": "https://scipy.org",
        "Sklearn": "https://scikit-learn.org/stable/",
        "Plotly":"https://plotly.com/python/plotly-express",
        "Streamlit": "https://streamlit.io",
    }

    st.sidebar.markdown("## Ссылки на используемый софт")
    link_1_col, link_2_col, link_3_col = st.sidebar.columns(3)

    i = 0
    link_col_dict = {0: link_1_col, 1: link_2_col, 2: link_3_col}
    for link_text, link_url in software_link_dict.items():

        st_col = link_col_dict[i]
        i += 1
        if i == len(link_col_dict.keys()):
            i = 0

        create_st_button(link_text, link_url, st_col=st_col)

if __name__ == "__main__":
    main()

