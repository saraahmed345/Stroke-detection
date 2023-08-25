#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and Exploring the Dataset

# In[506]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st


# In[507]:


df= pd.read_csv(r'C:\ML\stroke-data(2).csv')


# # Definining Exploratory Data Analysis with an overview of the whole project

# In[508]:


df.head()


# In[509]:


df.info()


# In[510]:


df.describe()


# In[511]:


df.head(25)


# In[512]:


sns.countplot(x='gender', data=df, hue="smoking_status", palette='rainbow')


# In[513]:


sns.kdeplot(df['stroke'], shade=True, color='m')


# In[514]:


sns.kdeplot(df['bmi'], shade=True, color='m')


# In[515]:


sns.displot(df['hypertension'], kde=False, color='m',height=6)


# In[516]:


sns.displot(df['heart_disease'], kde=False, color='b',height=6)


# # Checking missing values and Outliers & Creating visual methods to analyze the data

# In[517]:


df['bmi'].isnull().value_counts()


# In[518]:


#precet of missing data in bmi
print ('Missing Precentage in BMI' , (201/5110)*100)


# In[519]:


df['bmi'].describe()


# In[520]:


#median of bmi is 28.1


# In[521]:


df['bmi'].fillna(value=28.1, inplace=True)


# In[522]:


df.duplicated().value_counts()


# In[523]:


columns =['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']
for i in columns:
    print('**',i,'**','\n',df[i].value_counts() ,'\n\n\n')


# In[524]:


df['work_type'] = df['work_type'].replace('children', 'Never_worked' )


# In[525]:


df['work_type'].value_counts()


# In[526]:


sns.countplot(x='work_type', data=df, hue="stroke", palette='rainbow')    
#Those who work have more precentage to have a stroke


# In[527]:


sns.histplot(df['bmi'])


# In[528]:


sns.boxplot(x="stroke", y="avg_glucose_level", data=df,palette='rainbow')


# In[529]:


sns.boxplot(x="stroke", y="bmi", data=df,palette='rainbow')


# In[530]:


sns.boxplot(x="stroke", y="age", data=df,palette='rainbow')


# In[531]:


sns.pairplot(df[['avg_glucose_level', 'age', 'stroke']],hue='stroke',palette='magma')


# In[532]:


df.corr()
sns.heatmap(df.corr(),annot=True)


# In[533]:


df['smoking_status'].value_counts()


# In[534]:


df=df.drop(columns=['id', 'smoking_status','gender','Residence_type','work_type','ever_married'])


# In[535]:


df.head()


# # create models to fit the data

# In[536]:


x = df.iloc[:, :-1]
y = df.iloc[:, 5]
print(x)


# In[537]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state =0)


# In[538]:


classifier1 = LogisticRegression(random_state = 0)


# In[539]:


classifier1.fit(X_train, y_train)


# In[540]:


y_predict = classifier1.predict(X_test)


# In[541]:


cm=confusion_matrix(y_test, y_predict)
cm


# In[542]:


sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')


# In[543]:


report=classification_report(y_test, y_predict,zero_division=0)
print(report)


# In[544]:


print("The accuracy of precentage of logestic regression is: ", accuracy_score(y_test, y_predict)*100)


# In[545]:


#Features that plays viatl role
#bmi
#age
#hypertension
#heart_disease
#avg_glucose


# In[546]:


TP=1626
TN=0
FP=78
FN=0


# In[547]:


Error=(FP+FN)/(TP+TN+FN+FP)
print("Error =" ,Error)


# In[548]:


from sklearn.linear_model import SGDClassifier


# In[549]:


clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=500)


# In[550]:


clf.fit(X_train, y_train)


# In[551]:


y_predict2 = clf.predict(X_test)


# In[552]:


cm=confusion_matrix(y_test, y_predict2)
cm


# In[553]:


print("The accuracy of precentage of SGD is: ", accuracy_score(y_test, y_predict2)*100)


# # save and load models using joblib library

# In[554]:


import joblib


# In[558]:


joblib_file1="Drug_Model"

joblib.dump(classifier1,joblib_file1)


# In[559]:


joblib_file2="Drug_Model2"
joblib.dump(clf,joblib_file2)


# In[560]:


loaded_model1=joblib.load(open(joblib_file1,'rb'))


# In[561]:


loaded_model2=joblib.load(open(joblib_file2,'rb'))


# In[562]:


y_predict1 =loaded_model1.predict(X_test)


# In[564]:


y_Predict2 =loaded_model2.predict(X_test)


# In[565]:


result1=np.round(accuracy_score(y_test, y_predict),2)


# In[566]:


result2=np.round(accuracy_score(y_test, y_predict2),2)


# In[569]:


print(result1)
print(result2)


# In[ ]:




