
import streamlit as st
import pickle
import numpy as np
from PIL import Image

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price")
#image = Image.open('image.jpg')
#st.image(image,  width = 700)
html_temp = """
<div style="background-color:DarkSlateGray;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit Laptop Price Prediction ML App </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['Os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.success("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
    
    
if st.button("About ML App"):
    st.text("Regression model to predict the laptop price based on the different features in the training data")
    st.text("Built with Streamlit")
    
                      
               
               
if st.button("About Author"):
    st.text("Name : Upendra Kumar") 
    st.text("Email : upendra.kumar48762@gmail.com") 
    st.text("Oragnization : Data Science Intern at ineuron.ai")                
               