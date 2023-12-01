import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import *
from pycaret.classification import *
mpl.rcParams['figure.dpi'] = 300
import streamlit as st
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].astype('category')
st.dataframe(df)
df.info()
df.head(3)
classf = setup(data = df, target='Outcome', train_size = 0.8,
normalize = False, session_id = 3934)
get_config('X').head()
# best = compare_models(sort = 'Accuracy')
model = create_model('lr', fold = 3)
predictions = predict_model(model)
predictions.head(10)
plot_model(model, 'confusion_matrix', scale=0.2)
plot_model(model, 'boundary', scale = 0.3)
st.image(plot_model(model, 'feature', scale = 0.95))
plot_model(model, 'error', verbose=True)
