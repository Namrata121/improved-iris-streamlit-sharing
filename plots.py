import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def app(df):
	  
  st.header('Visualise data')
  #st.set_option('deprecation.showPyplotGlobalUse', False)
  feature1 = st.selectbox("Select the x-axis values:",('SepalLengthCm','SepalWidthCm'))
  feature2 = st.selectbox("Select the y-axis values:",('PetalLengthCm','PetalWidthCm')) 

  st.subheader(f"Scatter plot between {feature1} and {feature2}")
  plt.figure(figsize = (12, 6))
  sns.scatterplot(x = feature1, y = feature2, data = df, hue='Species')
  st.pyplot()
