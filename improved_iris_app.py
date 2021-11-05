# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image
img1 = load_image('iris-setosa.jpeg')
img2 = load_image('Iris_virginica.jpg')
img3 = load_image('iris-versicolor.jpg')

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using 'map()'.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0' , '1', and '2'.

# Creating X and y variables
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating the Logistic Regression model 
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating the Random Forest Classifier model 
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

# Create a function 'prediction()' which accepts model, SepalLength, SepalWidth, PetalLength, PetalWidth as input and returns species name.
@st.cache()
def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
  	species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  	species = species[0]
  	if species == 0:
		
  		return "Iris-setosa"
	        
  	elif species == 1:
		
  		return "Iris-virginica"
	        
  	else:
		
  		return "Iris-versicolor"
	        

# Add title widget
st.sidebar.title("Iris Flower Species Prediction App")      

# Add 4 sliders and store the value returned by them in 4 separate variables. 
s_len = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
s_wid = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
p_len = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
p_wid = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

# Add a select box in the sidebar with label 'Classifier' 
# and with 3 options passed as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.
classifier = st.sidebar.selectbox("Classifier", ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

# when 'Predict' button is clicked, check which classifier is chosen and call 'prediction()' function.
# Store the predicted in a variable 'species_type' accuracy score of model in 'score' variable. 
# Print the value of 'species_type' and 'score' variable using 'st.text()' function.
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
	
	
#from PIL import Image

#def load_image(img):
#    im = Image.open(img)
#    image = np.array(im)
#    return image
#img = load_image('iris-setosa.jpeg')
#st.image(img)

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

