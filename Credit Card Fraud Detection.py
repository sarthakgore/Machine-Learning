#!/usr/bin/env python
# coding: utf-8

# # Importing Dependancies

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Importing Dataset

# In[ ]:


data = pd.read_csv(r"C:\Users\ADMIN\Downloads\archive\creditcard.csv")


# In[5]:


data


# In[6]:


data.head()


# In[6]:


data.info()


# # Finding null values

# In[ ]:


data.isnull().sum()


# # Distribuation of Transactions

# In[8]:


data["Class"].value_counts()


# In[ ]:


#This is highly unbalanced data


# # Seprating of data

# In[11]:


#Let us consider 0 --> Normal Transaction
    #            1 -->fraud Transaction


# In[12]:


legit = data[data.Class==0]
Fraud = data[data.Class==1]


# In[14]:


print(legit.shape)
print(Fraud.shape)


# In[ ]:


#Stastical measures of data


# In[15]:


legit.Amount.describe()


# In[16]:


Fraud.Amount.describe()


# In[17]:


legit_sample=legit.sample(n=492)


# In[ ]:


#Let's bulid a new dataset


# In[20]:


new_dataset= pd.concat([legit_sample,Fraud],axis=0)


# In[22]:


new_dataset.head()


# In[23]:


new_dataset["Class"].value_counts()


# # Splitting data into Features and Target

# In[27]:


x=new_dataset.drop(columns="Class",axis=1)
y=new_dataset["Class"]


# In[25]:


print(x)


# In[28]:


print(y)


# # Split the data into Traning and Testing Data

# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[31]:


print(x.shape,x_train.shape,x_test.shape)


# # Now let's Train the Model

# In[32]:


model=LogisticRegression()


# In[33]:


model.fit(x_train,y_train)


# In[ ]:


#Training model


# In[34]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[ ]:


#Calculating Accuracy


# In[36]:


print("Accuracy on traing data:",training_data_accuracy)


# In[ ]:


#Testing model


# In[39]:


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[ ]:


#Calculating Accuracy


# In[40]:


print("Accuracy score on test data:",test_data_accuracy)


# In[ ]:


#Thank you

