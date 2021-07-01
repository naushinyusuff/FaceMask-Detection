%%writefile app.py
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


st.set_option("deprecation.showfileUploaderEncoding",False)
st.title("Face-Mask Detector")
st.write(f'Upload a picture to find out if you or anyone is wearing a mask or not:')
st.write(f"(And also the % of protection, to ensure you are wearing it correctly!)")
clf= load_model("model.h5")
uploadedImage = st.file_uploader("Select an Image",type="jpg")
if uploadedImage is not None: 
  img=Image.open(uploadedImage)  
  st.image(img,caption="This is the uploaded image")

  if st.button("Predict"):
    
    img = img.resize((224,224))
    img = np.array(img,dtype = "float32")
    img = preprocess_input(img)
    img = img.reshape(1,224,224,3) 
    y_out = clf.predict(img)
    if y_out<=0.5:
      st.header("Mask not detected")
      st.header(f"Please wear a mask!")
    else:
      y_out = int(y_out*100)
      st.header("Mask detected!")  
      st.header(f"Protection score:{y_out} ")
      st.header(f"Good Job!")
      
