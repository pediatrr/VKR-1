import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
def dashboard():
    # Пример данных
    data = pd.DataFrame({
        'Имя': ['Пациент1', 'Пациент2', 'Пациент3', 'Пациент4'],
        'Номер': [1, 2, 3, 4],
        'Прогноз вероятности': [0.8, 0.6, 0.2, 0.9],
        'Пол': ['Мужской', 'Женский', 'Мужской', 'Женский'],
        'Возраст': [35, 45, 28, 50],
        'ИМТ': [22, 30, 18, 25],
    })

    # Заголовок дашборда
    st.title('Интерактивный прогноз')

    # Фильтры для данных
    age_range = st.slider('Выберите диапазон возраста', min_value=int(data['Возраст'].min()), max_value=int(data['Возраст'].max()), value=(int(data['Возраст'].min()), int(data['Возраст'].max())))
    bmi_range = st.slider('Выберите диапазон ИМТ', min_value=int(data['ИМТ'].min()), max_value=int(data['ИМТ'].max()), value=(int(data['ИМТ'].min()), int(data['ИМТ'].max())))

    # Фильтрация данных
    filtered_data = data[(data['Возраст'] >= age_range[0]) & (data['Возраст'] <= age_range[1]) & (data['ИМТ'] >= bmi_range[0]) & (data['ИМТ'] <= bmi_range[1])]

    # Вывод таблицы данных
    st.subheader('Данные о пациентах и результатах моделей')
    st.table(filtered_data)

    st.subheader('График распределения прогнозов')
    hist_values = np.histogram(filtered_data['Прогноз вероятности'], bins=10, range=(0, 1))[0]
    st.bar_chart(hist_values)

    # Взаимосвязь с возрастом и ИМТ
    st.subheader('Взаимосвязь с возрастом и ИМТ:')
    fig = px.scatter(filtered_data, x='Возраст', y='ИМТ', color='Прогноз вероятности', labels={'Прогноз вероятности': 'Вероятность заболевания'})
    st.plotly_chart(fig)

    # Таблица сопоставления
    st.subheader('Таблица сопоставления результатов прогнозов')
    selected_patients = st.multiselect('Выберите пациентов для сравнения', filtered_data['Имя'].unique())

    if selected_patients:
        comparison_table = filtered_data[filtered_data['Имя'].isin(selected_patients)][['Имя', 'Прогноз вероятности']]
        st.table(comparison_table)
    else:
        st.info('Выберите пациентов для сравнения в таблице сопоставления.')
if __name__ == "__main__":
    dashboard()
