import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


def app(df):

    X = df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    svc_model = SVC(kernel = 'linear')
    svc_model.fit(X_train, y_train)


    rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
    rf_clf.fit(X_train, y_train)
     
    log_reg = LogisticRegression(n_jobs = -1)
    log_reg.fit(X_train, y_train)

    def load_image(img):
        im = Image.open(img)
        image = np.array(im)
        return image
    img1 = load_image('iris-setosa.jpeg')
    img2 = load_image('Iris_virginica1.jpg')
    img3 = load_image('iris-versicolor.jpg')

    @st.cache(suppress_st_warning=True)
    def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
        species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        species = species[0]
        if species == 0:
          
          return "Iris-setosa"
        elif species == 1:
          
          return "Iris-virginica"
        else:
          
          return "Iris-versicolor"

    st.markdown("<p style='color:turquoise;font-size:30px'>Iris Flower Species Prediction App",unsafe_allow_html = True)
    #st.title("Iris Flower Species Prediction App")
    st.sidebar.title("Iris Flower Features")       
     
    s_len = st.sidebar.slider("Sepal Length", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
    s_wid = st.sidebar.slider("Sepal Width", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()))
    p_len = st.sidebar.slider("Petal Length", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
    p_wid = st.sidebar.slider("Petal Width", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()))

    classifier = st.sidebar.selectbox("Classifier", ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))


    if st.sidebar.button("Predict"):
      if classifier =='Support Vector Machine':
        species_type = prediction(svc_model, s_len, s_wid, p_len, p_wid)
        score = svc_model.score(X_train, y_train)
      
      elif classifier =='Logistic Regression':
        species_type = prediction(log_reg, s_len, s_wid, p_len, p_wid)
        score = log_reg.score(X_train, y_train)
      
      else:
        species_type = prediction(rf_clf, s_len, s_wid, p_len, p_wid)
        score = rf_clf.score(X_train, y_train)

      if species_type == "Iris-setosa":
        st.image(img1)
        st.write("Species predicted:", species_type)
      elif species_type == "Iris-virginica":
        st.image(img2)
        st.write("Species predicted:", species_type)
      elif species_type == "Iris-versicolor":
        st.image(img3)
        st.write("Species predicted:", species_type)

      st.write("Accuracy score of this model is:", score)
      lst = [svc_model , rf_clf, log_reg ]

      if classifier =='Support Vector Machine':
        model = lst[0]

      elif classifier =='Random Forest Classifier':
        model = lst[1]
  
      elif classifier == 'Logistic Regression':
        model = lst[2]
      label = ['Iris-setosa','Iris-virginica','Iris-versicolor' ]
      pred = model.predict(X_train)
      cm = (confusion_matrix(y_train,pred))
      cr = classification_report(y_train,pred)

      st.title("Evaluation of :")
      st.write(classifier)
      st.subheader("Confusion Matrix")
      plt.figure(figsize = (8, 5))
      ax = sns.heatmap(cm, annot = True,cmap="YlGnBu" , xticklabels = label , yticklabels = label )  
      #bottom, top = ax.get_ylim()                    # Getting the top and bottom margin limits.
      #ax.set_ylim(bottom +0.5 , top-0.5 )           # Increasing the bottom and decreasing the bottom margins respectively.
      ax.set_yticklabels(labels=ax.get_yticklabels(), va='center')
      st.pyplot()
      
      st.subheader("Classification Report")
      st.write("Classification_Report: ",cr)

