import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from digitclassifier import get_classifier
from tensorflow.keras.models import load_model
import os as os
import cv2
import matplotlib.pyplot as plt

#st.session_state.model = load_model(os.path.join('models','idgitclassifiermodel.h5'))
#Load in the model
if 'model' not in st.session_state:
    st.session_state.model = load_model(os.path.join('models','idgitclassifiermodel.h5'))
# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 12)

stroke_color = st.sidebar.color_picker("Stroke color hex: ")


st.title("Hand Writing Digit Classifier")
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color='white',
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


#if canvas_result.image_data is not None:
#    st.image(canvas_result.image_data)
#if canvas_result.json_data is not None:
#    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#    for col in objects.select_dtypes(include=['object']).columns:
#       objects[col] = objects[col].astype("str")
#    st.dataframe(objects)


if st.button("Guess"):
    if canvas_result.image_data is not None:
        #read canvas data as grayscale image
        canvas_image = np.array(canvas_result.image_data)

        #convert to grayscale
        gray_scale = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)
        #Make the background black to match the format of the dataset
        img_reversed = cv2.bitwise_not(gray_scale)
        st.image(img_reversed) #show the image reversed

        #reduce the size
        img_resized = cv2.resize(img_reversed, (28, 28), interpolation=cv2.INTER_AREA)
        
        #expand to correct shape
        img_reshaped = np.expand_dims(img_resized/255,2)
        img_reshaped = np.expand_dims(img_reshaped,0)

        #make prediction
        y_pred =st.session_state.model.predict(img_reshaped)
        y_pred = np.argmax(y_pred, axis=1)

        st.write(f'I think you drew a {y_pred}')

