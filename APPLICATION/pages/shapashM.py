import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shapash import SmartExplainer
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
st.set_page_config(page_title="ShapashM", page_icon="🚩")
st.header('Выберите переменные для модели', divider='rainbow')

df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')
df['Pregnancies'] = df['Pregnancies'].astype('category')
y = df['Outcome']
X = df.drop(columns='Outcome')

col1, col2, col3 = st.columns(3)

with col1:
    multiselect = st.selectbox('Переменная', X.columns.tolist())

with col2:
    multiselect_out = st.multiselect('Класс', y.unique().tolist(), default=[1])

with col3:
    patient = st.slider('Выберите пациента', min_value=min(X.index), max_value=max(X.index))

# разбивка
model = GradientBoostingClassifier()
model.fit(X, y)

# предсказываем
y_pred = pd.DataFrame(model.predict(X),columns=['Outcome'],index=X.index)
y= y.astype(int)

# фильтр классов
y_pred_filtered = y_pred[y_pred['Outcome'].isin(multiselect_out)].astype(int)
X_filtered = X.loc[y_pred_filtered.index]

xpl = SmartExplainer(model=model)
xpl.compile(x=X.loc[y_pred_filtered.index],
            y_pred=y_pred_filtered,
            y_target=y.loc[y_pred_filtered.index])

col1.write(xpl.plot.features_importance())
col2.write(xpl.plot.scatter_plot_prediction())
index = patient
col3.write(xpl.plot.local_plot(index=index))

col1.write(xpl.plot.top_interactions_plot(nb_top_interactions=5))
col3.write(xpl.plot.contribution_plot(multiselect))
summary_df = xpl.to_pandas(
    max_contrib=8, 
    threshold=5000,
)
xpl_shap = SmartExplainer(
    model=model,
    backend='shap',
)
xpl_shap.compile(x=X_filtered)

xpl_lime_2 = SmartExplainer(
    model=model,
    backend='lime',
    data=X_filtered,
)
xpl_lime_2.compile(x=X_filtered)

col1.subheader('SHAP доверительный уровень')
col1.write(xpl_shap.plot.stability_plot())
col3.subheader('LIME доверительный уровень')
col3.write(xpl_lime_2.plot.stability_plot())
