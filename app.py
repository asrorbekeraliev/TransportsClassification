import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('Transportni klassifikatsiya qiluvchi model')

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'jpg', 'svg'])

if file:
    # PIL orqali convert qilish
    img = PILImage.create(file)

    # modelni yuklab olish
    model = load_learner('transport_model.pkl') 

    # model orqali rasmni predict qilish
    prediction, prediction_id, probibility = model.predict(img)

    # Natijalarni chop etish

    st.success(f"Prediction: {prediction}")
    st.info(f"Probibility: {probibility[prediction_id]*100:.2f} %")
    st.image(file)

    # PLotting
    fig = px.bar(x=probibility*100, y=model.dls.vocab)
    st.plotly_chart(fig)
