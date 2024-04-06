import streamlit as st
import pandas as pd
from pdpbox import pdp, get_example, info_plots
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib
st.set_page_config(page_title="PDPbox", page_icon="🚩",layout='wide')
st.header('PDPbox интерпретатор', divider='rainbow')
col1, col2 = st.columns(2)
df = pd.read_csv('diabetes.csv')
#df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')
#мультиселект
co = X.columns.tolist()
yo =  y.values.tolist()
yo = set(yo)

#targetfor=y.loc[yo]
# Делаем модель

#https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from interpret import show
from interpret.blackbox import PartialDependence
import streamlit
import streamlit.components.v1 as components
from interpret import show
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
seed = 42

y = df['Outcome']
X = df.drop(columns='Outcome')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

pca = PCA()
rf = RandomForestClassifier(random_state=seed)

blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
blackbox_model.fit(X_train, y_train)

pdp = PartialDependence(blackbox_model, X_train)
set_visualize_provider(InlineProvider(detected_envs=['streamlit'])) #Очень Важно
with col1:
    streamlit.write(show(pdp.explain_global(name='Локальный PDP'), 0))
from streamlit_shap import st_shap
import shap


with col2:
    select_p = st.selectbox('Переменная для зависимости', co)
    st_shap(shap.plots.partial_dependence(select_p, blackbox_model.predict,X_test,feature_expected_value=True,ice = False, model_expected_value = True),height=400, width=1250)
with st.expander('Взаимодействие переменных'):
    def pdp_1_1():
        multiselect = st.multiselect('Выбор признаков', co, default=["BMI",'Age'])
        multiselect_out = st.multiselect('Выбери класс предсказания', yo, default=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  
    # разбивка
        model = XGBClassifier()
        model.fit(X_train, y_train)
        # предсказываем
        y_pred = model.predict(X_test)
        #2.2
        target_67 = info_plots.InteractTargetPlot(
            df=df,
            features=multiselect,
            feature_names=multiselect,
            target='Outcome',
            num_grid_points=10,
            grid_types='equal',
            percentile_ranges=None,
            grid_ranges=None,
            cust_grid_points=None,
            show_outliers=False,
            endpoints=True,
        )

        fig, axes, summary_df = target_67.plot(
            which_classes=multiselect_out,
            show_percentile=True,
            figsize=(1200, 400),
            ncols=2,
            plot_params={"gaps": {"outer_y": 0.05}},
            engine='plotly',
            template='plotly_white',
        )
        fig
if __name__ == "__main__":
        pdp_1_1()