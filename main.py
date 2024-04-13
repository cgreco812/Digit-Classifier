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


#Load in the model

model = load_model(os.path.join('models','idgitclassifiermodel.h5'))
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
if st.button("Guess"):
    #get canvas data and convert it to a numpy array
    canvas_numpy_array = np.array(canvas_result.image_data)
    #convert it to gray scale
    x = np.dot(canvas_numpy_array[...,:3], [0.2989, 0.5870, 0.1140])
    #resize image
    x = cv2.resize(x, (28, 28), interpolation=cv2.INTER_AREA)
    image = x
    st.write(image.shape)
    fig = plt.figure(figsize=(28, 28))
    plt.imshow(x, cmap='gray')
    st.write(fig)
    #expand to correct shape
    #x = np.expand_dims(x/255,2)
    x = np.expand_dims(x,0)
    st.write(x.shape)
    #make prediction
    y_pred = model.predict(x)
    #y_pred = np.argmax(y_pred, axis=1)
    st.write(y_pred)


        #make prediction
