"""Diabetes_Prediction_(2068 & 2048).ipynb"""

#upload the csv file in google colab
from google.colab import files
uploaded = files.upload()

import numpy as np
import pandas as pd
df=pd.read_csv('diabetes.csv')
df

#Cases of Non-diabetic and diabetic
df['Outcome'].value_counts()

df['Pregnancies'].value_counts()

df['Glucose'].value_counts()

df['BloodPressure'].value_counts()

df['SkinThickness'].value_counts()

df['Insulin'].value_counts()

df['BMI'].value_counts()

df['DiabetesPedigreeFunction'].value_counts()

df['Age'].value_counts()

df.describe()

#Sepating the data and the labels
x=df.drop(columns='Outcome',axis=1)
y=df['Outcome']
print(x)
print(y)

#Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
ac=accuracy_score(y_test,y_pred)*100
cm=np.array(confusion_matrix(y_test,y_pred))
print(ac)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred1=knn.predict(x_test)
ac1=accuracy_score(y_test,y_pred1)*100
cm1=np.array(confusion_matrix(y_test,y_pred1))
print(ac1)
print(cm1)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred2=lr.predict(x_test)
ac2=r2_score(y_test,y_pred2)*100
print(ac2)

from sklearn.ensemble import RandomForestClassifier
x=df.drop(columns='Outcome',axis=1)
y=df['Outcome']
rf=RandomForestClassifier(max_depth=3)
rf.fit(x,y)
new_data=pd.DataFrame({'Pregnancies':1,'Glucose':85,'BloodPressure':66,'SkinThickness':29,
                       'Insulin':0,'BMI':26.6,'DiabetesPedigreeFunction':0.351,'Age':31,},index=[0])
p=rf.predict(new_data)
if p[0]==0:
  print('Non-Diabetic')
else:
  print('Diabetic')

import matplotlib.pyplot as plt
df=pd.DataFrame(df,columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
# plot the dataframe
df.plot(x="Outcome", y=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"], kind="bar", figsize=(15,10))
# print bar graph
plt.show()

#The distribution of the outcome variable in the data was examined and visualized.
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data=df,x='Outcome')
plt.show()