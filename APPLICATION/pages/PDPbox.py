import streamlit as st
import pandas as pd
from pdpbox import pdp, get_example, info_plots
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib
st.set_page_config(page_title="PDPbox", page_icon="🚩")
st.header('Выберите переменные для модели', divider='rainbow')

df = pd.read_csv('diabetes.csv')
#df['Outcome'] = df['Outcome'].astype('category')

y = df['Outcome']
X = df.drop(columns='Outcome')
#мультиселект
co = X.columns.tolist()
yo =  y.values.tolist()
yo = set(yo)
multiselect = st.multiselect('Multiselect', co, default=["BMI",'Age'])
multiselect_out = st.multiselect('Multiselect_output_class', yo, default=1)
#targetfor=y.loc[yo]
# Делаем модель
def pdp_1_1():
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
        grid_types='percentile',
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

    fig, axes, summary_df = target_67.plot(
        which_classes = multiselect_out,
        show_percentile=True,
        figsize=(1200, 400),
        ncols=2,
        plot_params={"gaps": {"outer_y": 0.05, "top": 0.1}},
        engine='plotly',
        template='plotly_white',
    )
    fig
if __name__ == "__main__":
    pdp_1_1()
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

streamlit.write(show(pdp.explain_global(), 0))

