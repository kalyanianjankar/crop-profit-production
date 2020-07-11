#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# In[2]:


#load data and remove the not required attributes
df=pd.read_csv('datafile1.csv')
df.drop(['Cost of Cultivation (`/Hectare) A2+FL'], axis=1, inplace=True)
df.drop(['State'], axis=1, inplace=True)
df.drop(['Crop'], axis=1, inplace=True)


# In[3]:


# dimension of data
df.shape


# In[4]:


# sample data
df.head()


# In[5]:


#check if any null values are present or not
df.info()


# In[6]:


# data descriptive statistics
df.describe()


# In[7]:


#data visualization
#boxplot

boxplot = df.boxplot(column=['Cost of Cultivation (`/Hectare) C2'])


# In[8]:


boxplot = df.boxplot(column=['Yield (Quintal/ Hectare) '])


# In[9]:


boxplot = df.boxplot(column=['Cost of Production (`/Quintal) C2'])


# In[10]:


#histogram
df.hist(figsize=(7,7))


# In[11]:


#kurtosis for outliers detection
df.kurtosis()


# In[12]:


#rename the columns
df = df.rename(columns={'Cost of Cultivation (`/Hectare) C2': 'CC'})
df = df.rename(columns={'Yield (Quintal/ Hectare) ': 'Yield'})
df = df.rename(columns={'Cost of Production (`/Quintal) C2': 'CP'})


# In[13]:


#log transformation to remove outliers
df['CC'] = np.log((1+df['CC']))
df['Yield'] = np.log((1+df['Yield']))
#df['CP'] = np.log((1+df['CP']))


# In[14]:


df


# In[15]:


#check for outliers after log tranformation
boxplot = df.boxplot(column=['CC'])


# In[16]:


boxplot = df.boxplot(column=['Yield'])


# In[17]:


print(df.kurtosis())


# In[18]:


# plot correlation matrix
correlations = df.corr()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,3,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()


# In[19]:


#scatter plot
plt.scatter(df.CP,df.CC)
plt.xlabel('Cost of Production')
plt.ylabel('Cost of Cultivation')


# In[20]:


plt.scatter(df.CP,df.Yield)
plt.xlabel('Cost of Production')
plt.ylabel('Yield')


# In[21]:


# select feature to train the model
features=['CC','Yield']
X=df[features]


# In[22]:


# output variable
y=df['CP']


# In[23]:


#split the data into training (80%) and testing (20%) data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)


# In[24]:


#train the DecisionTreeRegressor on the training data
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)


# In[25]:


# predict the output of testing data
y_pred=regressor.predict(X_test)


# In[26]:


#compare the actual and predicted data
res=pd.DataFrame({'actual':y_test,'predicted':y_pred})
res


# In[27]:


#performance metrics
# R^2 score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[28]:


# MAE, MSE, RMSE
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[29]:


model_score = regressor.score(X_test,y_test)
model_score


# In[ ]:





# In[30]:





# In[ ]:





# In[ ]:





# In[38]:


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
os.environ["PATH"] += os.pathsep + 'C:\\Users\\kajal\\Anaconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin\\graphviz'


# In[39]:


def plot_decision_tree(clf,feature_name,target_name):
    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_name,  
                         class_names=target_name,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())


# In[43]:


clf = tree.DecisionTreeRegressor(criterion='mse',max_depth=5)


# In[44]:




clf = clf.fit(X_train,y_train)




# In[45]:



plot_decision_tree(clf, X_train.columns,df.columns[1])


# In[ ]:





# In[42]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:





# In[ ]:





# In[54]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




