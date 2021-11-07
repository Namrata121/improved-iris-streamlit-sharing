import streamlit as st


def app(df):
  st.markdown("<p style='color:brown;font-family:cursive;font-size:35px'>Iris Flower Species Prediction App",unsafe_allow_html = True)

  st.text("""
    This web app allows a user to predict the type of iris flower.
  """
  )

  st.header("View Data") 
  with st.beta_expander("View Raw Dataset"):
    st.table(df)

  if st.checkbox("Show summary"):
    st.table(df.describe())

  
