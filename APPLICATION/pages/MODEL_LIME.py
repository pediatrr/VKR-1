import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from statsmodels.formula.api import ols
st.set_page_config(page_title="MODEL", page_icon="📊")
st.markdown("# MODEL Demo LIME")
st.sidebar.header("MODEl Demo LIME library")
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
st.header('Выберите переменные для модели', divider='rainbow')
multiselect = st.multiselect('Multiselect', co)

def lime ():
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    import lime
    import lime.lime_tabular
    X = df[multiselect]
    y = y = df['Outcome']  
    # добавим (единица) константа
    X = sm.add_constant(X)

    # разбивка
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Делаем модель
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # предсказываем
    y_pred = model.predict(X_test)

    selected_patient_data = X.loc[5]
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=X_train.columns, mode='classification')

    # Получение интерпретации для выбранного пациента
    explanation = explainer.explain_instance(selected_patient_data.values, model.predict_proba)
    fig = explanation.as_pyplot_figure()
    fig.savefig('explanation.png')
    st.image('explanation.png')
    # Визуализация важности признаков
    explanation.show_in_notebook()
if __name__ == "__main__":
    lime()
# Show and update progress bar
bar = st.progress(50)
bar.progress(100)

