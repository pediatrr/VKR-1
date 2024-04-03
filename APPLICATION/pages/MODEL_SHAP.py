import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from statsmodels.formula.api import ols
st.set_page_config(page_title="MODEL", page_icon="📊",layout='wide')
st.markdown("# MODEL Demo SHAP")
st.sidebar.header("MODEl Demo")
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
st.header('Выберите переменные для модели', divider='rainbow')
multiselect = st.multiselect('Multiselect', co, default=co)

def shap():
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from streamlit_shap import st_shap
    import shap
    X = df[multiselect]
    y = y = df['Outcome']  
    # добавим (единица) константа
    X = sm.add_constant(X)

    # разбивка
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Делаем модель
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # предсказываем
    y_pred = model.predict(X_test)
    # используем shap для трактования
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    expected_value = explainer.expected_value
    
    # Суммируем эффекты фич
    #shap.summary_plot(shap_values, X_test)
    X_display=sm.add_constant(df[multiselect])
    # визуализация
    #shap.plots.waterfall(shap_values[0])shap.plots.waterfall(shap_values[0, 0])
    st_shap(shap.plots.waterfall(shap_values[0,:]), height=400,width=1250)
    st_shap(shap.plots.beeswarm(shap_values), height=400,width=1250)
    st_shap(shap.plots.bar(shap_values,max_display=3),height=400, width=1250) # new
    st_shap(shap.plots.scatter(shap_values),height=400, width=1250)
    st_shap(shap.plots.heatmap(shap_values),height=800, width=1450)
    st_shap(shap.plots.partial_dependence("Age",model.predict,X_test,feature_expected_value=True,ice = False, model_expected_value = True),height=400, width=1250) # Сюда воткни преключатель
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1250)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:]), height=400, width=1250)
        #return  shap_values
if __name__ == "__main__":
    shap()
    #def ss (shap_values):
        #st.write(shap.plots.scatter(shap_values[:, shap_values[0,:]], color=shap_values))
    #ss()
@st.cache_data
@st.cache_resource
def model(multiselect):
    X_selected = sm.add_constant(df[multiselect])
    model= sm.OLS(y, X_selected).fit()
    fig, ax = plt.subplots(figsize=(4,4))
    fig = sm.graphics.qqplot(model.resid, fit=True, line="45", ax=ax)
    st.pyplot(fig)
    #fig, ax = plt.subplots(figsize=(16,10))
    #fig = sm.graphics.plot_regress_exog(model, ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'], fig=fig)
    #st.pyplot(fig)
if st.button('Старт',type="primary"):
    model(multiselect)
# Show and update progress bar
#bar = st.progress(50)
#bar.progress(100)

