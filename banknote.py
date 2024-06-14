#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''This dataset is about distinguishing genuine and forged banknotes. Data were extracted from images that were taken from genuine and forged banknote-like specimens. the purpose of focusing on K-Means model, I only picked out two variables to build the models, which are Variance (of Wavelet Transformed image) and Skewness (of Wavelet Transformed image).'''


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv("C:/Users/Tara/Downloads/bank.csv")
data


# In[5]:


data.isnull().sum()


# In[10]:


plt.scatter(data.V1,data.V2)
plt.title("scatter plot of varianvce vs skewness")


# In[11]:


#fitting the model 
import numpy as np
from sklearn.cluster import KMeans


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming data is your DataFrame with 'V1' and 'V2' columns

n_iter = 3  # Number of iterations
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 grid of plots

for i in range(n_iter):
    km = KMeans(n_clusters=2, max_iter=3)  # Initialize KMeans with 2 clusters and 3 max iterations
    km.fit(data)  # Fit the model to the data
    centroids = km.cluster_centers_  # Get the cluster centroids

    # Scatter plot for cluster 1
    ax[i].scatter(data[km.labels_ == 0]['V1'], data[km.labels_ == 0]['V2'], label='cluster 1')
    # Scatter plot for cluster 2
    ax[i].scatter(data[km.labels_ == 1]['V1'], data[km.labels_ == 1]['V2'], label='cluster 2')
    # Scatter plot for centroids
    ax[i].scatter(centroids[:, 0], centroids[:, 1], c='r', marker='*', s=100, label='centroid')

    ax[i].legend()  # Add legend
    ax[i].set_xlabel('V1')  # Set x-axis label
    ax[i].set_ylabel('V2')  # Set y-axis label
    ax[i].set_title(f'Iteration {i + 1}')  # Set title for each subplot

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plots


# In[ ]:


#sice results are similar it means kmeans is stable.


# In[14]:


clusters = KMeans(2)
clusters.fit(data)
data['clusterid'] = clusters.labels_
plt.scatter(data['V1'],data['V2'],c=data['clusterid'])
plt.show()


# In[21]:


'''The purple one belongs to the Fake Notes while the yellow belongs to Genuine notes.'''


# In[15]:


clusters.cluster_centers_


# In[16]:


data.groupby( 'clusterid' ).describe()

