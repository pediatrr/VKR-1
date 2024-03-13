import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import streamlit
import streamlit.components.v1 as components

st.set_page_config(page_title="MODEL", page_icon="📊")
st.markdown("# MODEL Demo LIME")
st.sidebar.header("MODEl Demo LIME library")
df = pd.read_csv('diabetes.csv')
#df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
st.header('Выберите переменные для модели', divider='rainbow')
multiselect = st.multiselect('Multiselect', co)


def lime ():
    from sklearn.model_selection import train_test_split
    X = df[multiselect]
    y = df['Outcome']  
    # добавим (единица) константа
    X = sm.add_constant(X)

    # разбивка
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    from interpret import show
    from interpret import set_visualize_provider
    from interpret.provider import InlineProvider
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from interpret.blackbox import LimeTabular
    from interpret.blackbox import MorrisSensitivity

    pca = PCA()
    rf = RandomForestClassifier()

    blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
    blackbox_model.fit(X_train, y_train)

    lime = LimeTabular(blackbox_model, X_train)
    set_visualize_provider(InlineProvider(detected_envs=['streamlit'])) #Очень Важно
    streamlit.write(show(lime.explain_local(X_test, y_test)))
    streamlit.success('Success message')
    msa = MorrisSensitivity(blackbox_model, X_train)
    streamlit.write(show(msa.explain_global()))
    streamlit.success('Success message_2')
if __name__ == "__main__":
    lime()

