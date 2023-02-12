from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import uvicorn
from fastapi import FastAPI
import numpy as np
from keras.utils.image_utils import load_img
from keras.utils.image_utils import img_to_array
import matplotlib.pyplot as plt
from PIL import Image


model = load_model('C:/Users/Dell_owner/Downloads/facialexp.hdf5')





classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

emotion_app = FastAPI()

def emotion_UI():
    st.title("Predict Type of Emotion")
    image = Image.open('C:/Users/Dell_owner/Downloads/EmotionalExpression.png')
    st.image(image, caption='Express your Emotion')
    uploaded_file = st.file_uploader("Choose a image file")
    ok = st.button("Predict type of Emotion")
    try:
        if ok == True:  # if user pressed ok button then True passed
            im1 = load_img(uploaded_file, target_size=(48, 48))
            x = img_to_array(im1)
            x = x * (1. / 255)
            x = np.expand_dims(x, axis=0)
            result = classes[np.argmax(model.predict(x))]
            st.image(im1)
            print(result)

           # classindx = predict_image(im1)
            if result == "angry":
                st.success("Emotion types is Angry")
            elif result == "disgust":
                st.success("Emotion types is Disgust")
            elif result == "fear":
                st.success("Emotion types is fear")
            elif result == "happy":
                st.success("Emotion types is Happy")
            elif result == "sad":
                st.success("Emotion types is Sad")
            elif result == "surprise":
                st.success("Emotion types is Surprise")
            elif result == "neutral":
                st.success("Emotion types is Neutral")
    except Exception as e: # all error
            st.info(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(lcp3_app, host="0.0.0.0", port=8000,log_level='info')