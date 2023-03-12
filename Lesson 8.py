# Задание 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[3]:


from sklearn.datasets import load_boston


# In[4]:


boston = load_boston()
data = boston["data"]


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


feature_names = boston["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.head()


# In[7]:


target = boston["target"]

Y = pd.DataFrame(target, columns=["price"])
Y.head()


# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[11]:


from sklearn.manifold import TSNE


# In[12]:


tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)


# In[13]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.show()


# Задание 2

# In[14]:


from sklearn.cluster import KMeans


# In[15]:


model = KMeans(n_clusters=3, random_state=42, max_iter=100)
labels_train = model.fit_predict(X_train_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)
plt.show()


# In[16]:


print('Первый кластер: ', Y_train[labels_train == 0].mean())
print('Второй кластер:', Y_train[labels_train == 1].mean())
print('Третий кластер:', Y_train[labels_train == 2].mean())


# In[17]:


print('Первый кластер: ', X_train['CRIM'][labels_train == 0].mean())
print('Второй кластер:', X_train['CRIM'][labels_train == 1].mean())
print('Третий кластер:', X_train['CRIM'][labels_train == 2].mean())


# Задание 3

# In[18]:


scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[19]:


X_test_tsne = tsne.fit_transform(X_test_scaled)


# In[20]:


labels_test = model.fit_predict(X_test_scaled)

plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=labels_test)
plt.show()


# In[21]:


print('Первый кластер: ', X_test['CRIM'][labels_test == 0].mean())
print('Второй кластер:', X_test['CRIM'][labels_test== 1].mean())
print('Третий кластер:', X_test['CRIM'][labels_test== 2].mean())