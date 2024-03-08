import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
from shapash import SmartExplainer
import io
import matplotlib.pyplot as plt
import plotly.express as px

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

# Filter y_pred based on selected classes
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

#st.write(xpl.plot.interactions_plot('Sex', 'Pclass'))
#st.write(xpl.filter(max_contrib=8,threshold=100))
patient = st.slider('Выберите пациента', min_value=min_index, max_value=max_index)
index = patient
st.write(xpl.plot.local_plot(index=index))
summary_df = xpl.to_pandas(
    max_contrib=8, # Number Max of features to show in summary
    threshold=5000,
)
st.write(summary_df)
#xpl.save('./xpl.pkl')