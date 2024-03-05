import streamlit as st
import pandas as pd
import numpy as np
def run():
    st.set_page_config(
    page_title="Welcome",
    page_icon="👋",)
    st.header('Представления результатов медицинских моделей', divider='rainbow')
    st.image('phone.png')
    st.markdown("#### Эффективность моделей машинного обучения зависит не только от их технических характеристик, но и от того, насколько конечные пользователи могут правильно и верно интерпретировать результаты работы модели.")
    st.divider()
    st.markdown ("- ### Контекст проблемы") 
    st.markdown ("- ### Цели и ожидания пользователей")
    st.markdown ("- ### Возможные риски и ограничения модели")  
    st.divider()
    st.button("Reset", type="primary")
if __name__ == "__main__":
    run()
# Github_deskop_test
# next_test
# retest
#hello 29/02/24
# best practice:https://github.com/Jumitti/TFinder/blob/main/TFinder-v1.py