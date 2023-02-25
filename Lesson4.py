
import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


config InlineBackend.figure_format = 'svg'


# Задание 1

# In[6]:


x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]


# In[7]:


plt.plot(x, y)
plt.show()


# In[8]:


plt.scatter(x, y)
plt.show


# Задание 2

# In[9]:


t = np.linspace(0, 10, 51)
print(t)
f = np.cos(t)
print(f)


# In[10]:


plt.plot(t, f, color='blue')
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show()


# Задание 3

# In[11]:


x = np.linspace(-3, 3, 51)
print(x)


# In[12]:


y1 = x**2
print(y1)


# In[13]:


y2 = 2 * x + 0.5
print(y2)


# In[14]:


y3 = -3 * x - 1.5
print(y3)


# In[15]:


y4 = np.sin(x)
print(y4)


# In[16]:


fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax1.set_title('График $y_1$')
ax2.set_title('График $y_2$')
ax3.set_title('График $y_3$')
ax4.set_title('График $y_4$')
ax1.set_xlim([-5, 5])
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


# In[17]:


import pandas as pd
plt.style.use('fivethirtyeight')


# In[22]:


url = "http://localhost:8888/edit/Desktop/creditcard.csv"
creditcard = pd.read_csv(url)


# In[ ]:


class_list = creditcard['Class'].value_counts()
print(class_list)
