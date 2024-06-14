#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
from sklearn.cluster import KMeans

import numpy as np


# In[3]:


data=pd.read_csv("C:/Users/Tara/Downloads/happy_score.csv")
data


# In[7]:


import matplotlib.pyplot as plt
happy=data['happyScore']
inequality=data['income_inequality']
income=data['avg_income']


# In[16]:


plt.scatter(income,happy,s=inequality*10,alpha=0.3)
plt.xlabel('income')
plt.ylabel('happy_score')


# In[22]:


income_happy=np.column_stack((income,happy))
income_happy


# In[27]:


km_res=KMeans(n_clusters=2).fit(income_happy)
km_res.cluster_centers_
clusters=km_res.cluster_centers_
plt.scatter(income,happy)
plt.scatter(clusters[:,0],clusters[:,1],s=1000)


# In[28]:


'''people having high income are happy
also people having low income seem to be quite happy too'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




