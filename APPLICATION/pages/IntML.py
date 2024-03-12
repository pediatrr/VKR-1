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
import plotly.io as pio
import json
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
ebm_global = ebm.explain_global()
ebm_dict = ebm_global.data()
ebm_json = json.dumps(ebm_dict)
plotly_fig = ebm_global.visualize(5) #"work"
streamlit.write(plotly_fig)

#html = pio.to_html(plotly_fig)
set_visualize_provider(InlineProvider(detected_envs=['streamlit']))
streamlit.components.v1.html(ebm_json, height=1000, width=1000, scrolling=True)
streamlit.write(show(ebm_global))
#st.write(fig_html)
#fig, ax= plt.subplots()
#fig_html = mpld3.fig_to_html(ebm_global)
#components.html(HTML(init_js + body_js).data, height=1000, width=1000, scrolling=True)
'''fi = ebm_global._internal_obj['overall']
st.table(fi)'''


#https://github.com/interpretml/interpret/issues/423
#https://github.com/interpretml/interpret/pull/438/files