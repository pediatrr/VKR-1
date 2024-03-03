import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alibi.explainers import IntegratedGradients
from alibi.datasets import load_cats
from alibi.utils import visualize_image_attr
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # True
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import io
from PIL import Image
# Example usage
model = ResNet50V2(weights='imagenet')
print("ResNet50V2 model loaded successfully.")

image_shape = (224, 224, 3)
data, labels = load_cats(target_size=image_shape[:2], return_X_y=True)
print(f'Images shape: {data.shape}')
data = (data / 255).astype('float32')
i = 2
plt.imshow(data[i])
model = ResNet50V2(weights='imagenet')
n_steps = 50
method = "gausslegendre"
internal_batch_size = 50
ig  = IntegratedGradients(model,
                          n_steps=n_steps, 
                          method=method,
                          internal_batch_size=internal_batch_size)
instance = np.expand_dims(data[i], axis=0)
predictions = model(instance).numpy().argmax(axis=1)
explanation = ig.explain(instance, 
                         baselines=None, 
                         target=predictions)
explanation.meta
explanation.data.keys()

baselines = np.random.random_sample(instance.shape)
explanation = ig.explain(instance,
                         baselines=baselines,
                         target=predictions)
attrs = explanation.attributions[0]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
visualize_image_attr(attr=None, original_image=data[i], method='original_image',
                    title='Original Image', plt_fig_axis=(fig, ax[0]), use_pyplot=False);

visualize_image_attr(attr=attrs.squeeze(), original_image=data[i], method='blended_heat_map',
                    sign='all', show_colorbar=True, title='Overlaid Attributions random',
                     plt_fig_axis=(fig, ax[1]), use_pyplot=True);
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = Image.open(buf)
st.image(image, caption='CAT', use_column_width=True)

#https://github.com/SeldonIO/alibi/blob/0039fbd84fa5c12ce699741beb1bcd60d5ca72a0/doc/source/examples/integrated_gradients_imagenet.ipynb