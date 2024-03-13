import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
import streamlit
import streamlit.components.v1 as components
streamlit.set_page_config(page_title="CAM2+", page_icon="🧊",layout='wide')

df = pd.read_csv('diabetes.csv')
#df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
ebm_global = ebm.explain_global()
set_visualize_provider(InlineProvider(detected_envs=['streamlit'])) #Очень Важно
#set_visualize_provider(InlineProvider()) неработает
streamlit.markdown('### Глобальный бустер модели')
streamlit.write(show(ebm_global))

streamlit.markdown('### Локальный бустер')
ebm_local = ebm.explain_local(X_test, y_test)
streamlit.write(show(ebm_local)) #локал


from interpret.glassbox import ClassificationTree
dt = ClassificationTree(random_state=42)
dt.fit(X_train, y_train)
dt_global = dt.explain_global()
streamlit.markdown('### Глобальное дерево модели')
streamlit.write(show(dt_global))
streamlit.markdown('### Локальное дерево данных')

streamlit.write(show(dt.explain_local(X_test, y_test), 0))

from interpret.glassbox import LogisticRegression
y=y.astype('category')
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_global = lr.explain_global()
streamlit.markdown('### Глобальная LOG регрессия')
streamlit.write(show(lr_global))
streamlit.markdown('### Локальная LOG регрессия')
streamlit.write(show(lr.explain_local(X_test, y_test), 0))
#streamlit.write(show([lr_global, dt_global])) не работает
#fi = ebm_global._internal_obj['overall']
#streamlit.table(fi)
#https://github.com/interpretml/interpret/issues/423
#https://github.com/interpretml/interpret/pull/438/files