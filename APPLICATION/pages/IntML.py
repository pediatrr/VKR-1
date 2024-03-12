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
import mpld3
import streamlit.components.v1 as components
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
ebm_global = ebm.explain_global()




set_visualize_provider(InlineProvider(detected_envs=['streamlit']))
streamlit.components.v1.iframe((ebm_global.data), height=1000, width=1000, scrolling=True)
#st.write(fig_html)
#fig, ax= plt.subplots()
#fig_html = mpld3.fig_to_html(ebm_global)
#components.html(HTML(init_js + body_js).data, height=1000, width=1000, scrolling=True)
'''fi = ebm_global._internal_obj['overall']
st.table(fi)'''


#https://github.com/interpretml/interpret/issues/423
#https://github.com/interpretml/interpret/pull/438/files