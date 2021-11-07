import streamlit as st
import numpy as np
import pandas as pd
import home
import plots
import predict


st.set_page_config(page_title = 'Flower Prediction',
                    page_icon = ':flower:',
                    layout = 'centered',
                    initial_sidebar_state = 'auto'
                    )

iris_df = pd.read_csv("iris-species.csv")
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

pages = {
	        "Home": home,
	        "Visualise Data": plots, 
	        "Predict": predict
         }

st.sidebar.title('Navigation')
user_choice = st.sidebar.radio("Go to", tuple(pages.keys()))
if user_choice == "Home":
    home.app(iris_df) 
else:
    selected_page = pages[user_choice]
    selected_page.app(iris_df) 


import base64

@st.cache(allow_output_mutation=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('test.png')