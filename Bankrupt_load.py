# -*- coding: utf-8 -*- Using model for prediction

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import ppscore as pps
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pickle
import streamlit as st 

st.title('Model Deployment Group-1: Bankrupt Model-LR')

   
df = pd.read_csv('bankruptcy-prevention.csv',sep=';')
df.rename(columns={' management_risk':"management_risk"},inplace=True)
df.rename(columns={' financial_flexibility':"financial_flexibility"},inplace=True) 
df.rename(columns={' credibility':"credibility"},inplace=True)
df.rename(columns={' competitiveness':"competitiveness"},inplace=True)
df.rename(columns={' operating_risk':"operating_risk"},inplace=True)
df.rename(columns={' class':"Class"},inplace=True)

#Changing the data type - Converting the target variable into integer format from the object format
label_encoder = preprocessing.LabelEncoder()
df["Class"]=label_encoder.fit_transform(df.iloc[::,6:7:])

df2 = df.drop(['industrial_risk','management_risk','operating_risk'],axis = 1)
st.subheader('User Input parameters')
st.write(df[['financial_flexibility','credibility','competitiveness']])


# load the model from disk
load_model = pickle.load(open('log_model.sav', 'rb'))

prediction = load_model.predict(df2.iloc[::,0:3:])
prediction_proba = load_model.predict_proba(df2.iloc[::,0:3])

#st.subheader('Predicted Result')
#st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Predictions')
st.write(prediction)
output=pd.concat([df2,pd.DataFrame(prediction_proba)],axis=1)

output.to_csv('output.csv')