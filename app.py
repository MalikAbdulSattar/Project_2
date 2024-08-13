import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st



#Collecting Data
dataset=pd.read_csv("diabetes.csv")
# 0 ----> for Non Diabetes
# 1 ----> for  Diabetes
dataset["Outcome"].value_counts()
dataset.groupby("Outcome").mean()
X=dataset.drop(columns="Outcome",axis=1)
Y= dataset["Outcome"]
scaler= StandardScaler()
scaler.fit(X)
standarized_data=scaler.transform(X)
X=standarized_data
Y= dataset["Outcome"]
#Train test split data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
classifier=svm.SVC(kernel="linear")
#Traing The support vector machine 
classifier.fit(X_train,Y_train)
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
#print("Accuracy on training data : ", training_data_accuracy)
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
#print("Accuracy on testing data : ", test_data_accuracy)


# Buulding Websites    
st.title("Diabates Prediction")
input_data=st.text_input("Search Here..")
if st.button("Predict"):
    input_data = input_data.strip()
    input_list = [float(X) for X in input_data.split(',') if X]
    input_data_as_array = np.asarray(input_list)
    input_data_reshaped= input_data_as_array.reshape(1,-1)
    std_input_data=scaler.transform(input_data_reshaped)

    prediction=classifier.predict(std_input_data)
    if prediction[0]==0:
      st.write("The Person is non-Diabiatic")
    else:
      st.write("The Person is Diabatic")




    

