import io
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = os.path.join('model', 'model_32.h5')

@st.cache(allow_output_mutation=True)
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess(image_data):
    # Convert all the images to size 299x299 as expected by the inception v3 mode
    img = Image.open(io.BytesIO(image_data)).resize((299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess(image)  # preprocess the image
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    fea_vec = model_new.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
    return fea_vec


def create_caption(photo):
    image = encode(photo).reshape((1, 2048))
    in_text = 'startseq'
    max_length = 34
    model = model_load()
    with open('logfiles/wordtoix.json', 'r') as fp:
        wordtoix = json.load(fp)
    with open('logfiles/ixtoword.json', 'r') as fp:
        ixtoword = json.load(fp)
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = ixtoword[str(yhat)]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def cs_image_captioning():
    st.markdown("<h1 style='text-align: center; color: black;'>Image Captioning</h1>",
                unsafe_allow_html=True)

    Image = st.file_uploader('Upload image here', type=['jpg', 'jpeg', 'png'])
    my_expander = st.expander(label='🙋 Upload help')
    with my_expander:
        st.markdown('Filetype to upload : **JPG, JPEG, PNG**')
    if Image is not None:
        col1, col2 = st.columns([1, 2])
        Image = Image.read()
        with col1:
            col1.subheader("Uploaded Image")
            st.image(Image)
            caption = create_caption(Image)
        with col2:
            col2.subheader("Caption")
            # caption = ""
            st.code(caption, language="markdown")

    return None


def main():
    st.set_page_config(
        page_title='Image Captioning',
        layout="wide",
    )
    cs_image_captioning()
    return None


if __name__ == '__main__':
    main()
