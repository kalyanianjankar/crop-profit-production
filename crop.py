#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('E:\DATA_MINING\project\crop.csv')


# In[3]:


df.describe()


# In[4]:


df.head()


# In[5]:


one_hot=pd.get_dummies(df['crop'])


# In[6]:


df.drop('crop',axis=1)


# In[7]:


df=df.join(one_hot)


# In[8]:


df


# In[9]:


df=df.sample(frac=1)


# In[10]:


df.drop('crop',axis=1)


# In[11]:


df.dropna()


# In[12]:


boxplot = df.boxplot(column=['area_harvested'])


# In[13]:


boxplot = df.boxplot(column=['yieldd'])


# In[14]:


boxplot = df.boxplot(column=['production'])


# In[15]:


df['area_harvested'] = np.log((1+df['area_harvested']))
df['yieldd'] = np.log((1+df['yieldd']))
df['production'] = np.log((1+df['production']))


# In[16]:


df


# In[17]:


features=['area_harvested','yieldd','Barley', 'Coconuts', 'Groundnuts, with shell', 'Jute', 'Millet',
       'Oilseeds nes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Sugar cane',
       'Tea', 'Wheat']


# In[18]:


X=df[features]


# In[19]:


y=df['production']


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)


# In[21]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)


# In[22]:


from matplotlib import pyplot as plt
plt.scatter(df.area_harvested,df.yieldd)
plt.xlabel('area_harvested')
plt.ylabel('yieldd')


# In[23]:


y_pred=regressor.predict(X_test)


# In[24]:



res=pd.DataFrame({'actual':y_test,'predicted':y_pred})


# In[25]:


res


# In[26]:


from sklearn.metrics import r2_score


# In[27]:


r2_score(y_test, y_pred)


# In[28]:


plt.plot(X_test, y_test, color="cornflowerblue",
         label="max_depth=2", linewidth=2)


# In[29]:


plt.plot(X_test, y_pred, color="yellowgreen", label="max_depth=5", linewidth=2)


# In[30]:


plt.plot(y_test, y_pred, color="yellowgreen", label="max_depth=5", linewidth=2)


# In[31]:


X_grid = np.arange(min(y_test), max(y_pred), 1)


# In[32]:


X_grid = X_grid.reshape((len(X_grid), 1))


# In[33]:


plt.scatter(y_test, y_pred, color = 'red')


# In[34]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[35]:


df.info()


# In[36]:


from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from IPython.display import Image 
from io import StringIO 
import pydotplus
from sklearn import preprocessing
from sklearn import tree
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\ASHISH\\Anaconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin\\graphviz'


# In[37]:


def plot_decision_tree(clf,feature_name,target_name):
    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_name,  
                         class_names=target_name,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())


# In[38]:


clf = tree.DecisionTreeRegressor(criterion='mse',max_depth=5)


# In[39]:


clf = clf.fit(X_train,y_train)


# In[40]:


plot_decision_tree(clf, X_train.columns,df.columns[1])


# In[41]:


plot_decision_tree(clf, X_train.columns,df.columns[3])


# In[ ]:




