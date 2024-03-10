import streamlit as st
import pandas as pd
from pdpbox import pdp, get_example, info_plots
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib

st.set_page_config(page_title="PDPbox", page_icon="🚩")
st.header('Выберите переменные для модели', divider='rainbow')

df = pd.read_csv('diabetes.csv')

y = df['Outcome']
X = df.drop(columns='Outcome')

co = X.columns.tolist()
yo =  y.values.tolist()
yo = set(yo)

default_values = ['Insulin', 'BMI']
default_values_out = [0, 1]
col1, col2 = st.columns(2)

with col1:
    multiselect = st.multiselect('Переменные', co, default=default_values)
with col2:
    multiselect_out = st.multiselect('Класс', yo, default=default_values_out)

def pdp_2_1():
    selected_classes = df[df['Outcome'].isin(multiselect_out)]
    y_selected = selected_classes['Outcome']
    X_selected = selected_classes.drop(columns='Outcome')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    target_67 = info_plots.InteractTargetPlot(
        df=selected_classes,
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
    fig_1, axes, summary_df = target_67.plot(
        which_classes = multiselect_out,
        show_percentile=True,
        figsize=(1200, 600),
        ncols=2,
        plot_params={"gaps": {"outer_y": 0.05, "top": 0.05}},
        engine='plotly',
        template='plotly_white',
    )
    fig_1
    return model, selected_classes

def pdp_2_2(model, selected_classes):
    predict_67_25 = info_plots.InteractPredictPlot(
        model=model,
        df=selected_classes,
        model_features=co,
        features=multiselect, 
        feature_names=multiselect,
        pred_func=None,
        n_classes=None,
        num_grid_points=10,
        grid_types='percentile',
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
        predict_kwds={},
        chunk_size=-1,
    )
    fig_2, axes, summary_df = predict_67_25.plot(
        which_classes=multiselect_out,
        show_percentile=False,
        figsize=None,
        ncols=2,
        annotate=False,
        plot_params={"gaps": {"inner_y": 0.06}},
        engine='plotly',
        template='plotly_white',
    )
    fig_2
    return predict_67_25

def pdp_2_3(model, selected_classes):
    pdp_67_25 = pdp.PDPInteract(
        model=model,
        df=selected_classes,
        model_features=co,
        features=multiselect,
        feature_names=multiselect,
    )
    return pdp_67_25
def pdp_1_3(model, selected_classes):
    predict_67_next = info_plots.PredictPlot(
    model=model,
    df=selected_classes,
    model_features=co,
    feature= multiselect[0],
    feature_name= multiselect[0],
    pred_func=None,
    n_classes=None,
    num_grid_points=10,
    grid_type='percentile',
    percentile_range=None,
    grid_range=None,
    cust_grid_points=None,
    show_outliers=False,
    endpoint=True,
    predict_kwds={},
    chunk_size=-1,
)
    return predict_67_next

if __name__ == "__main__":
    model, selected_classes = pdp_2_1()
    predict_67_25 = pdp_2_2(model, selected_classes)
    pdp_67_25 = pdp_2_3(model, selected_classes)
    predict_67_next =  pdp_1_3(model,selected_classes)
    fig_3, axes = pdp_67_25.plot(
        plot_type="grid",
        to_bins=True,
        plot_pdp=False,
        show_percentile=True,
        which_classes=multiselect_out,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params={"gaps": {"outer_y": 0.2}},
        engine="matplotlib",
        template="plotly_white",
    )
    fig_3
    fig_4, axes = pdp_67_25.plot(
    plot_type="contour",
    to_bins=True,
    plot_pdp=True,
    show_percentile=False,
    which_classes=multiselect_out,
    figsize=None,
    dpi=300,
    ncols=2,
    plot_params=None,
    engine="plotly",
    template="plotly_white",
)
fig_4

fig_5, axes, summary_df = predict_67_next.plot(
    show_percentile=True,
    figsize=None,
    ncols=2,
    plot_params=None,
    engine='plotly',
    template='plotly_white',
)
fig_5
#https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb



