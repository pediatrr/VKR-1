import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE, plot_ale
import matplotlib.pyplot as plt
import io
from PIL import Image
st.set_page_config(page_title="ALE", page_icon="🚩",layout='wide')
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')
co = X.columns.tolist()
yo = y.to_frame().columns.tolist()

def mod(X_train, y_train, co):
    lr = GradientBoostingClassifier()
    lr.fit(X_train, y_train)

    logit_fun_lr = lr.decision_function
    proba_fun_lr = lr.predict_proba
    logit_ale_lr = ALE(logit_fun_lr,feature_names=co)
    proba_ale_lr = ALE(proba_fun_lr,feature_names=co)
    logit_exp_lr = logit_ale_lr.explain(X_train)
    proba_exp_lr = proba_ale_lr.explain(X_train)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Accumulated Local Effects plots for GB on diabets dataset')
    st.markdown (" ##### Accumulated Local Effects (ALE) - это метод вычисления эффектов признаков, основанный на статье Apley and Zhu 'Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models'. Алгоритм позволяет получить глобальные объяснения для классификационных и регрессионных моделей на табличных данных, не зависящих от модели(черный ящик). ")

    st.header('ALE plot for decision function')
    fig, ax = plt.subplots()
    plot_ale(logit_exp_lr, ax=ax, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5}, sharey=None)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    st.image(image, caption='ALE plot for decision function', use_column_width=True)

    st.header('ALE plot for probability function')
    fig, ax = plt.subplots()
    plot_ale(proba_exp_lr, ax=ax, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5})

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    st.image(image, caption='ALE plot for probability function', use_column_width=True)
    st.header('Histogram for each target')
    fig, ax = plt.subplots()
    ax.hist(X_train, label=yo)
    ax.set_xlabel(co[2])
    ax.legend()
    st.pyplot(fig)

st.button("Reset", type="primary")
if __name__ == "__main__":
    if 'multiselect' not in st.session_state:
        st.session_state.multiselect = co  # initialize with all features

    multiselect = st.multiselect('Multiselect', co, st.session_state.multiselect)
    if multiselect != st.session_state.multiselect:  # if the selection has changed
        st.session_state.multiselect = multiselect  # update the session state
        X = df[multiselect]
        y=y.to_numpy()
        X=X.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        mod(X_train, y_train, multiselect)  # rerun the function with the selected features
#https://docs.seldon.io/projects/alibi/en/stable/examples/ale_classification.html