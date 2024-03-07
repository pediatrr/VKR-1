import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import io
from alibi.explainers import PermutationImportance, plot_permutation_importance
from PIL import Image
st.set_page_config(page_title="CWR_alibi", page_icon="🚩")
st.header('Выберите переменные для модели', divider='rainbow')
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')
yo = y.to_frame().columns.tolist()
co = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
u=classification_report(y_true=y_test, y_pred=y_pred)
st.write([u])
"""
explainer = PermutationImportance(predictor=X,
                                  score_fns=['accuracy', 'f1'],
                                  feature_names=co,
                                  verbose=True)
exp = explainer.explain(X=X_test, y=y_test)
plot_permutation_importance(exp,
                            n_cols=2,
                            fig_kw={'figwidth': 14, 'figheight': 6})
def loss_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - f1_score(y_true=y_true, y_pred=y_pred)
explainer_loss_f1 = PermutationImportance(predictor=X,
                                          loss_fns={'1 - f1': loss_f1},
                                          feature_names=co,
                                          verbose=True)
exp_loss_f1 = explainer_loss_f1.explain(X=X_test, y=y_test)

plot_permutation_importance(exp=exp_loss_f1,
                            fig_kw={'figwidth': 7, 'figheight': 6})
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

image = Image.open(buf)
st.image(image, caption='ALE plot for decision function', use_column_width=True)"""