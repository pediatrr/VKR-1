import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alibi.explainers import IntegratedGradients
from alibi.utils import visualize_image_attr
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
import shap
from streamlit_shap import st_shap
#from tqdm.notebook import tqdm
st.set_page_config(page_title="INTG_img", page_icon="🧊",layout='wide')

vismethod = st.selectbox('Выберите метод визуализации', ['blended_heat_map', 'heat_map'])
visattribute= st.radio('Выберите параметры:', ['positive', 'negative','absolute_value','all'])
def Igalibi():
    model = keras.models.load_model('resnet50_pneumonia_model.h5')
    image_path = ('C:/Users/Андр/Documents/GitHub/VKR/APPLICATION/NORMAL2-IM-1427-0001.jpeg')
    image_path = ('C:/Users/Андр/Documents/GitHub/VKR/APPLICATION/person1946_bacteria_4874.jpeg')
    image_path = ('C:/Users/1/Documents/GitHub/VKR/APPLICATION/person1946_bacteria_4874.jpeg')
    image = load_img(image_path, target_size=(224, 224))
    img = np.array(image) / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    predicted_class = int(round(prediction[0][0]))

    label_dict = {0: "Pneumonia", 1: "Normal"}
    predicted_label = label_dict[predicted_class]

    st.write(f"The predicted label is: {predicted_label}")

    n_steps = 50
    method = "gausslegendre"
    internal_batch_size = 50
    ig  = IntegratedGradients(model,
                            n_steps=n_steps, 
                            method=method,
                            internal_batch_size=internal_batch_size)
    img = img.reshape(224, 224, 3)
    instance = np.expand_dims(img, axis=0)
    baselines = np.random.random_sample(instance.shape)
    predictions = model(instance).numpy().argmax(axis=1)
    explanation = ig.explain(instance, 
                            baselines=baselines, 
                            target=predictions)
    #explanation.meta
    #explanation.data.keys()
    attrs = explanation.attributions[0]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    visualize_image_attr(attr=None, original_image=img, method='original_image',
                        title='Original Image', plt_fig_axis=(fig, ax[0]), use_pyplot=False)

    visualize_image_attr(attr=attrs.squeeze(), original_image=img, method=vismethod,
                        sign=visattribute, show_colorbar=False, title='Overlaid Attributions',
                        plt_fig_axis=(fig, ax[1]), use_pyplot=True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    st.image(image, caption='Отображение выбранных аттрибутов', use_column_width=True)
    from shap.maskers import Image as ShapImage

    # define the masker
    background = np.zeros((224, 224, 3))  # use a background of zeros
    masker = ShapImage(background)
    
    # create the explainer with the masker
    expln_shap = shap.Explainer(model=model, masker=masker, output_names=[1])
    img = img.reshape(1, 224, 224, 3)
        # compute shap values
    shap_values = expln_shap(img,max_evals=50)
    st_shap(shap.image_plot(shap_values),height=400, width=1250)
    #import torch
if st.button('Старт',type="primary"):
    Igalibi()    
#https://github.com/SeldonIO/alibi/blob/0039fbd84fa5c12ce699741beb1bcd60d5ca72a0/doc/source/examples/integrated_gradients_imagenet.ipynb