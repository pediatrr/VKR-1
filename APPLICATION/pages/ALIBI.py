import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE, plot_ale
import matplotlib.pyplot as plt
import io
from PIL import Image
data = load_iris()
feature_names = data.feature_names
target_names = data.target_names
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

logit_fun_lr = lr.decision_function
proba_fun_lr = lr.predict_proba
logit_ale_lr = ALE(logit_fun_lr, feature_names=feature_names, target_names=target_names)
proba_ale_lr = ALE(proba_fun_lr, feature_names=feature_names, target_names=target_names)
logit_exp_lr = logit_ale_lr.explain(X_train)
proba_exp_lr = proba_ale_lr.explain(X_train)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Accumulated Local Effects plots for Logistic Regression on Iris dataset')

st.header('ALE plot for decision function')
#st.pyplot(plot_ale(logit_exp_lr, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5}, sharey=None))
fig, ax = plt.subplots()
plot_ale(logit_exp_lr, ax=ax, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5}, sharey=None)

# Save the plot to a BytesIO 
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Create a PIL image object
image = Image.open(buf)

# Display
st.image(image, caption='ALE plot for decision function', use_column_width=True)


st.header('ALE plot for probability function')
#st.pyplot(plot_ale(proba_exp_lr, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5}))

fig, ax = plt.subplots()
plot_ale(proba_exp_lr, ax=ax, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5})

# Save the plot to a BytesIO object
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Create a PIL image object
image = Image.open(buf)

# Display 
st.image(image, caption='ALE plot for probability function', use_column_width=True)



st.header('Histogram for each target')
fig, ax = plt.subplots()
for target in range(3):
    ax.hist(X_train[y_train==target][:,2], label=target_names[target])
ax.set_xlabel(feature_names[2])
ax.legend()
st.pyplot(fig)
#https://docs.seldon.io/projects/alibi/en/stable/examples/ale_classification.html