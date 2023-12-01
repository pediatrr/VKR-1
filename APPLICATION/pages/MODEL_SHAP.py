import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from statsmodels.formula.api import ols
st.set_page_config(page_title="MODEL", page_icon="📊")
st.markdown("# MODEL Demo SHAP")
st.sidebar.header("MODEl Demo")
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
st.header('Выберите переменные для модели', divider='rainbow')
multiselect = st.multiselect('Multiselect', co)

def shap():
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from xgboost import XGBClassifier
    from streamlit_shap import st_shap
    import shap
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
    # используем shap для трактования
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Суммируем эффекты фич
    #shap.summary_plot(shap_values, X_test)
    X_display=sm.add_constant(df[multiselect])
    # визуализация
    #shap.plots.waterfall(shap_values[0])
    st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    st_shap(shap.plots.beeswarm(shap_values), height=300)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1000)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:]), height=400, width=1000)
if __name__ == "__main__":
    shap()
@st.cache_data
@st.cache_resource
def model(multiselect):
    X_selected = sm.add_constant(df[multiselect])
    model= sm.OLS(y, X_selected).fit()
    #fig = sm.graphics.influence_plot(model, criterion="cooks")
    #fig.tight_layout(pad=1.0)
    #st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(4,4))
    fig = sm.graphics.qqplot(model.resid, fit=True, line="45", ax=ax)
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(16,10))
    fig = sm.graphics.plot_regress_exog(model, 'BMI', fig=fig)
    st.pyplot(fig)
if st.button('Старт',type="primary"):
    model(multiselect)
# Show and update progress bar
bar = st.progress(50)
bar.progress(100)

