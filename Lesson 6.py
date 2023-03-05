Задание 1

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()
data = boston["data"]


# In[4]:


feature_names = boston["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.head()


# In[5]:


target = boston["target"]

Y = pd.DataFrame(target, columns=["price"])
Y.head()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


lr = LinearRegression()


# In[10]:


lr.fit(X_train, Y_train)


# In[11]:


y_pred_lr = lr.predict(X_test)
check_test_lr = pd.DataFrame({
    "Y_test": Y_test["price"], 
    "Y_pred_lr": y_pred_lr.flatten()})

check_test_lr.head()


# In[12]:


from sklearn.metrics import mean_squared_error

mean_squared_error_lr = mean_squared_error(check_test_lr["Y_pred_lr"], check_test_lr["Y_test"])
print(mean_squared_error_lr)


# Задание 2

# In[13]:


from sklearn.ensemble import RandomForestRegressor


# In[14]:


clf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)


# In[15]:


clf.fit(X_train, Y_train.values[:, 0])


# In[16]:


y_pred_clf = clf.predict(X_test)
check_test_clf = pd.DataFrame({
    "Y_test": Y_test["price"], 
    "Y_pred_clf": y_pred_clf.flatten()})

check_test_clf.head()


# In[17]:


mean_squared_error_clf = mean_squared_error(check_test_clf["Y_pred_clf"], check_test_clf["Y_test"])
print(mean_squared_error_clf)


# In[18]:


print(mean_squared_error_lr, mean_squared_error_clf)


# Задание 3

# In[19]:


print(clf.feature_importances_)


# In[20]:


feature_importance = pd.DataFrame({'name':X.columns, 
                                   'feature_importance':clf.feature_importances_}, 
                                  columns=['feature_importance', 'name'])
feature_importance


# In[21]:


feature_importance.nlargest(2, 'feature_importance')


# Задание 4

# In[22]:


df = pd.read_csv('../Lesson04/creditcard.csv.zip', compression='zip')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[ ]:


df['Class'].value_counts(normalize=True)


# In[ ]:


df.info()


# In[ ]:


pd.options.display.max_columns=100


# In[ ]:


df.head(10)


# In[ ]:


X = df.drop('Class', axis=1)


# In[ ]:


y = df['Class']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)


# In[ ]:


print('X_train ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)


# In[ ]:


parameters = [{
    'n_estimators': [10, 15], 
    'max_features': np.arange(3, 5), 
    'max_depth': np.arange(4, 7)
}]


# In[ ]:


clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3,
)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# In[ ]:


clf = RandomForestClassifier(max_depth=6, max_features=3, n_estimators=15)

clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict_proba(X_test)


# In[ ]:


y_pred_proba = y_pred[:, 1]


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


roc_auc_score(y_test, y_pred_proba)