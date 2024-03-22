import numpy as np
import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, Input, Reshape, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

def normalize(arr):
    arr_min = np.min(arr)
    return (arr - arr_min) / (np.max(arr) - arr_min)

from tensorflow import keras
model = keras.models.load_model('resnet50_pneumonia_model.h5')
#image_path = ('C:/Users/Андр/Documents/GitHub/VKR/APPLICATION/NORMAL2-IM-1427-0001.jpeg')
image_path = ('C:/Users/1/Documents/GitHub/VKR/APPLICATION/person1946_bacteria_4874.jpeg')
image = load_img(image_path, target_size=(224, 224))
img = np.array(image) / 255.0
img = img.reshape(1, 224, 224, 3)
prediction = model.predict(img)
predicted_class = int(round(prediction[0][0]))  # Assuming 1 is for "Normal" and 0 is for "Pneumonia"

label_dict = {0: "Pneumonia", 1: "Normal"}
predicted_label = label_dict[predicted_class]

st.write(f"The predicted label is: {predicted_label}")

n_steps = 50
method = "gausslegendre"
ig  = IntegratedGradients(model,
                          n_steps=n_steps,
                          method=method)
# Calculate attributions for the first 10 images in the test set
nb_samples = 1
X_test_sample = img
predictions = model(X_test_sample).numpy().argmax(axis=1)
explanation = ig.explain(X_test_sample,
                         baselines=None,
                         target=predictions)
explanation.meta
explanation.data.keys()
attrs = explanation.attributions[0]
normalized_attrs = normalize(attrs)
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
image_ids = [0]
vmin, vmax = np.min(normalized_attrs), np.max(normalized_attrs)

for row, image_id in enumerate(image_ids):
    # original images
    ax[row, 0].imshow(img[image_id].squeeze(), cmap='gray')
    ax[row, 0].set_title(f'Prediction: {predictions[image_id]}')

    # attributions
    attr = normalized_attrs[image_id]
    im = ax[row, 1].imshow(attr.squeeze(), vmin=vmin, vmax=vmax, cmap='viridis')

    # positive attributions
    attr_pos = attr.clip(0, 1)
    im_pos = ax[row, 2].imshow(attr_pos.squeeze(), vmin=vmin, vmax=vmax, cmap='viridis')

    # negative attributions
    attr_neg = attr.clip(-1, 0)
    im_neg = ax[row, 3].imshow(attr_neg.squeeze(), vmin=vmin, vmax=vmax, cmap='viridis')

ax[0, 1].set_title('Attributions');
ax[0, 2].set_title('Positive attributions');
ax[0, 3].set_title('Negative attributions');

for ax in fig.axes:
    ax.axis('off')

fig.colorbar(im, cax=fig.add_axes([0.95, 0.25, 0.03, 0.5]))
st.write(fig)
st.write(f"Attributions range from {np.min(attrs)} to {np.max(attrs)}")