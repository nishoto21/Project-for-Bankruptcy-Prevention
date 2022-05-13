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

data = pd.read_csv('bankruptcy-prevention.csv',sep=';')
data.head()

#As column names contains spaces so removing spaces from the column names by renaming it
data.rename(columns={' management_risk':"management_risk"},inplace=True)
data.rename(columns={' financial_flexibility':"financial_flexibility"},inplace=True) 
data.rename(columns={' credibility':"credibility"},inplace=True)
data.rename(columns={' competitiveness':"competitiveness"},inplace=True)
data.rename(columns={' operating_risk':"operating_risk"},inplace=True)
data.rename(columns={' class':"Class"},inplace=True)

#Changing the data type - Converting the target variable into integer format from the object format
label_encoder = preprocessing.LabelEncoder()
data["Class"]=label_encoder.fit_transform(data.iloc[::,6:7:])

#Forming new data by taking important features into consideration
new_data = data.drop(['industrial_risk','management_risk','operating_risk'],axis = 1)

#Splitting the data and forming the model
x = new_data.iloc[:,:-1].values
y = new_data.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.3,random_state= 7)
log_model = LogisticRegression(max_iter=500,random_state = 7)
log_model.fit(X_train, Y_train)

# save the model to disk
pickle.dump(log_model, open('log_model.sav', 'wb'))


# load the model from disk
load_model = pickle.load(open('log_model.sav', 'rb'))
result = load_model.score(x, y)
print(result)