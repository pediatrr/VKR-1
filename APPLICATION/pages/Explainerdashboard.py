import streamlit as st
import pandas as pd
import numpy as np
import explainerdashboard
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

df = pd.read_csv('diabetes.csv')

y = df['Outcome']
X = df.drop(columns='Outcome')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create the explainer and the dashboard
explainer = ClassifierExplainer(model, X_test, y_test)
db = ExplainerDashboard(explainer)
db.run(port=8050, mode='external')

# In your Streamlit app
def app():
    st.title("This is the machine learning page")
    dashboardurl = 'http://10.8.8.18:8501'
    st.components.v1.iframe(dashboardurl, width=None, height=900, scrolling=True)

app()

#https://github.com/oegedijk/explainerdashboard