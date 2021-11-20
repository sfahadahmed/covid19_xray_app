# Libraries
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

@st.cache(allow_output_mutation=True)
def load_model():
    f = open('model.json', 'r')
    json = f.read()
    f.close()

    model = model_from_json(json)
    model.load_weights('model_weights.h5')
    return model

# Title
st.write("# COVID Patient Identification using X-ray Image")

# Title image
st.image(Image.open('title.png'), use_column_width=True)

# Load model
model = load_model()

# File Uploader
file_image = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if(file_image is not None):
    image = Image.open(file_image)
    image = ImageOps.grayscale(image)

    if image is not None:
        img_array = np.array(image)
        img = tf.expand_dims(img_array, axis=0)

        st.image(image, caption="This patient is COVID <RESULT> (Accuracy <PERCENTAGE>%)", use_column_width=False)

        st.write(model.predict(img_array))

