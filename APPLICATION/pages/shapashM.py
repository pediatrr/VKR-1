import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib
from shapash import SmartExplainer
import io
import matplotlib.pyplot as plt
from PIL import Image
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
#def shapash():
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  
# разбивка
model = XGBClassifier()
model.fit(X_train, y_train)
# предсказываем
y_pred = pd.DataFrame(model.predict(X_test),columns=['Outcome'],index=X_test.index)
y_test= y_test.astype(int)
px.scatter(y_train)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

image = Image.open(buf)
st.dataframe(X_train) # для оценки
st.dataframe(y_pred)
st.dataframe(y_test)


xpl = SmartExplainer(model=model)
xpl.compile(x=X_test,
 y_pred=y_pred,
 y_target=y_test, # Optional: allows to display True Values vs Predicted Values
 )
xpl.plot.features_importance()

st.image(image, caption='ALE plot for decision function', use_column_width=True)
xpl.plot.contribution_plot("BMI")
xpl.plot.contribution_plot("Age")
xpl.filter(max_contrib=8,threshold=100)
xpl.plot.local_plot(index=23)
summary_df = xpl.to_pandas(
    max_contrib=3, # Number Max of features to show in summary
    threshold=5000,
)
st.dataframe(summary_df)
#xpl.save('./xpl.pkl')