#!/usr/bin/env python
# coding: utf-8

# In[71]:


import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[4]:


ds=load_boston()


# In[5]:


ds


# In[6]:


ds.keys()


# In[8]:


ds.data


# In[10]:


ds.target


# In[11]:


ds.feature_names


# In[12]:


ds.DESCR


# In[13]:


df=pd.DataFrame(data=ds.data, columns=ds.feature_names)


# In[14]:


df


# In[15]:


df['target']=pd.DataFrame(data=ds.target)


# In[16]:


df 


# In[17]:


df.info()


# In[19]:


df.isnull().sum()


# # Start of EDA Process

# In[20]:


sns.heatmap(df.isnull())


# In[22]:


df.describe()


# In[25]:


df.shape


# In[26]:


df.size


# In[29]:


df.skew()


# In[31]:


df['CRIM'].plot.box()


# In[32]:


df['ZN'].plot.box()


# In[34]:


df.plot(kind='box',subplots=True,layout=(2,7))


# In[35]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[39]:


thershold=3
print(np.where(z>3))


# In[43]:


df['ZN'].plot.hist()


# In[44]:


from scipy.stats import boxcox
df['CRIM']=boxcox(df['CRIM'],0)


# In[45]:


df['CRIM'].plot.hist()


# In[46]:


plt.scatter(df['CRIM'],df['target'])


# In[47]:


plt.scatter(df['B'],df['target'])


# In[48]:


x=df['RM']
y=df['target']
plt.scatter(x,y)
plt.show()


# In[49]:


sns.pairplot(df)


# In[57]:


corr_hmap=df.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
plt.show()


# In[61]:


df_new=df[(z<3).all(axis=1)]


# In[63]:


df_new.shape


# In[59]:


df.shape


# In[64]:


df=df_new
df.shape


# In[65]:


x=df.iloc[:,0:-1]


# In[66]:


x.head


# In[67]:


y=df.iloc[:,-1]
y.head()


# In[68]:


x.shape


# In[69]:


y.shape


# In[72]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33, random_state=42)


# In[73]:


x_train.shape


# In[74]:


y_train.shape


# In[75]:


x_test.shape


# In[77]:


y_test.shape


# In[79]:


lm=LinearRegression()


# In[80]:


lm.fit(x_train,y_train)


# In[83]:


lm.coef_


# In[84]:


lm.intercept_


# In[85]:


lm.score(x_train,y_train)


# In[87]:


pred=lm.predict(x_test)
print('Predicted result price:',pred)
print('Actual Price',y_test)


# In[88]:


print('Error:')
print('Mean absolute error:',mean_absolute_error(y_test,pred))
print('Mean squared error:',mean_squared_error(y_test,pred))
print('Root Mean squared error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[90]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[ ]:




