# Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageOps
import cv2

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

    if image is not None:
        image = ImageOps.fit(image, (299,299), Image.ANTIALIAS)
        #st.image(image, caption="This patient is COVID <RESULT> (Accuracy <PERCENTAGE>%)", use_column_width=False)
        st.image(image, use_column_width=False)

        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.expand_dims(image, axis=0)

        st.write(model.predict(image))

