# Importing Required library
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import requests

# Loading BMI_dict pkl file
BMI_dict = pickle.load(open('BMI_dict.pkl','rb'))
# Dict file convert into dataframe
df = pd.DataFrame(BMI_dict)
X = df.drop(['Cholesterol'],axis=1)
y = df[['Cholesterol']]
# Model selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1000)

st.title('Cholesterol Prediction')

# Random Forest Reggressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=5, max_depth=5,criterion='squared_error',random_state=165,
                           bootstrap=True)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)


Weight = st.selectbox(
     'Select Weight of Animal',
     np.array(df['Weight']))
BMI = st.selectbox(
     'Select BMI',
     np.array(df['BMI']))

prediction = rf.predict([[Weight,BMI]])
predict=str(np.round(prediction[0]))
pred=str(predict)




if st.button('Predicted Cholesterol'):
    pred



