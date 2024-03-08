import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from shapash import SmartExplainer
import matplotlib.pyplot as plt

st.set_page_config(page_title="ShapashM", page_icon="🚩")
st.header('Выберите переменные для модели', divider='rainbow')

df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')
#мультиселект
co = X.columns.tolist()
yo =  y.values.tolist()
yo = set(yo)
multiselect = st.selectbox('Переменная', co)


yo =  y.unique().tolist()
default_values_out = [1]
multiselect_out = st.multiselect('Класс', yo, default=default_values_out)

#def shapash():
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  
# разбивка
model = XGBClassifier()
model.fit(X, y)
# предсказываем
y_pred = pd.DataFrame(model.predict(X),columns=['Outcome'],index=X.index)
y= y.astype(int)

# фильтр классов
y_pred_filtered = y_pred[y_pred['Outcome'].isin(multiselect_out)].astype(int)
X_filtered = X.loc[y_pred_filtered.index]
min_index = min(X_filtered.index)
max_index = max(X_filtered.index)
xpl = SmartExplainer(model=model)
xpl.compile(x=X.loc[y_pred_filtered.index],
            y_pred=y_pred_filtered,
            y_target=y.loc[y_pred_filtered.index])

st.write(xpl.plot.features_importance())
st.write(xpl.plot.scatter_plot_prediction())

fig, ax = plt.subplots(figsize=(6, 4))

st.write(xpl.plot.contribution_plot(multiselect))
st.write(xpl.plot.top_interactions_plot(nb_top_interactions=5))

patient = st.slider('Выберите пациента', min_value=min_index, max_value=max_index)
index = patient
st.write(xpl.plot.local_plot(index=index))
summary_df = xpl.to_pandas(
    max_contrib=8, 
    threshold=5000,
)
#st.dataframe(summary_df)
xpl_shap = SmartExplainer(
    model=model,
    backend='shap',
)
xpl_shap.compile(x=X_filtered[0:])

xpl_lime_2 = SmartExplainer(
    model=model,
    backend='lime',
    data=X_filtered[0:],
)
xpl_lime_2.compile(x=X_filtered[0:])

st.subheader('SHAP доверительный уровень')
st.write(xpl_shap.plot.stability_plot())
st.subheader('LIME доверительный уровень')
st.write(xpl_lime_2.plot.stability_plot())
#xpl.save('./xpl.pkl')