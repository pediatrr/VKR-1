import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import streamlit
import streamlit.components.v1 as components
import  lime
st.set_page_config(page_title="LIME", page_icon="🧊",layout='wide')
st.markdown("# MODEL Demo LIME")
st.sidebar.header("MODEl Demo LIME library")
df = pd.read_csv('diabetes.csv')
#df['Outcome'] = df['Outcome'].astype('category')
col1, col2 = st.columns(2)
y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
st.header('Выберите переменные для модели', divider='rainbow')
multiselect = st.multiselect('Переменные', co, default=['BMI','Age'])
with st.expander('Объяснение метода'):
    st.write('- Поддержка объясненимости отдельных предсказаний для  классификаторов lime (сокращение от local interpretable model-agnostic explanations')
    st.write('- Lime основан на работе, представленной в этой статье https://arxiv.org/abs/1602.04938')
    st.write('- Вот ссылка на промо-ролик:https://www.youtube.com/watch?v=hUnRCxnydCc')

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
    from interpret.perf import ROC
    from interpret.perf import PR
    from interpret.greybox import ShapTree
    pca = PCA()
    rf = RandomForestClassifier()

    blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
    blackbox_model.fit(X_train, y_train)

    lime = LimeTabular(blackbox_model, X_train)
    set_visualize_provider(InlineProvider(detected_envs=['streamlit'])) #Очень Важно
    with col1:
        streamlit.write(show(lime.explain_local(X_test, y_test, name='Локальная LIME модель')))
    rfc_roc = ROC(blackbox_model.predict_proba).explain_perf(X_test,y_test,name='ROC')
    rfc_PR = PR(blackbox_model.predict_proba).explain_perf(X_test,y_test,name='График Precision')
    with col1:
        streamlit.write(show(rfc_roc))
    
    with col2:
        streamlit.write(show(rfc_PR))


    #rfc_shap = ShapTree(blackbox_model, X_test)
    # SHAPTREE still unresolved
    #streamlit.write(show(rfc_shap.explain_local(X_test, y_test, name='Локальная ShapTree модель')))    
    #streamlit.success('Success message')

    msa = MorrisSensitivity(blackbox_model, X_train)
    with col2:
        streamlit.write(show(msa.explain_global(name='Тест Чувствительности Морриса')))
    streamlit.success('Success message_2')
if __name__ == "__main__":
    lime()

