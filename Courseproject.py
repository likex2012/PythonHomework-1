# # Load Dataset

# In[ ]:


## Load packages


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy import stats


# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# In[ ]:


##Function


# In[2]:


def optimizing_df(df):
    for col in df.columns:
        if df[col].dtypes.kind == 'i' or df[col].dtypes.kind == 'u':
            if df[col].min() >= 0:
                df[col] = pd.to_numeric(df[col], downcast='unsigned')
            else:
                df[col] = pd.to_numeric(df[col], downcast='integer')

        elif df[col].dtypes.kind == 'f' or df[col].dtypes.kind == 'c':
            df[col] = pd.to_numeric(df[col], downcast='float')

        elif df[col].dtypes.kind == 'O':
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')

    return df


# # Load data

# In[3]:


train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')
print("Data is loaded!")


# In[ ]:


print(f"Train:\t{train.shape[0]}\t sales and {train.shape[1]} features")
print(f'Test:\t{test.shape[0]}\t sales and {test.shape[1]} features')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info(memory_usage='deep')


# In[ ]:


test.info(memory_usage='deep')


# # Memory use optimization

# In[ ]:


train['Rooms'] = train['Rooms'].astype('int64')
test['Rooms'] = test['Rooms'].astype('int64')


# In[ ]:


train['HouseFloor'] = train['HouseFloor'].astype('int64')
test['HouseFloor'] = test['HouseFloor'].astype('int64')


# In[ ]:


train = optimizing_df(train)
test = optimizing_df(test)


# In[ ]:


train.info(memory_usage='deep')


# In[ ]:


test.info(memory_usage='deep')


# # Data checking

# In[ ]:


all_data = pd.concat((train, test), sort=False).reset_index(drop=True)
all_data.drop(['Price'], axis=1, inplace=True)
print(f'all_data size is : {all_data.shape}')


# In[ ]:


all_data.describe().transpose()


# ## Fix Rooms

# In[ ]:


all_data.loc[all_data['Rooms'] > 6]


# In[ ]:


all_data.loc[all_data['Rooms'] == 0]


# In[ ]:


def df_fix_room(df):
    info_by_district_id = df.groupby(['DistrictId', 'HouseYear'], as_index=False).agg(
        {'Rooms': 'sum', 'Square': 'sum'}).rename(
        columns={'Rooms': 'sum_roos_dr', 'Square': 'sum_square_dr'})

    info_by_district_id['mean_square_per_room_in_dr'] = info_by_district_id['sum_square_dr']         / info_by_district_id['sum_roos_dr']
    info_by_district_id.drop(
        ['sum_square_dr', 'sum_roos_dr'], axis=1, inplace=True)

    df = pd.merge(df, info_by_district_id, on=[
                  'DistrictId', 'HouseYear'], how='left')

    df['mean_square_per_room_in_dr'] = df['mean_square_per_room_in_dr'].fillna(
        df['mean_square_per_room_in_dr'].mean())

    df.loc[df['Rooms'] > 6, 'Rooms']         = (df.loc[df['Rooms'] > 6, 'Square']
           // df.loc[df['Rooms'] > 6, 'mean_square_per_room_in_dr']).astype('int')

    df.loc[df['Rooms'] == 0, 'Rooms']         = (df.loc[df['Rooms'] == 0, 'Square']
           // df.loc[df['Rooms'] == 0, 'mean_square_per_room_in_dr']).astype('int')

    df.loc[df['Rooms'] == 0, 'Rooms'] = 1
    return df


# ## Fix Square

# In[ ]:


all_data.loc[all_data['Square'] > 200].nlargest(20, 'Square')


# In[ ]:


sns.distplot(all_data['Square'], fit=norm)

mu, sigma = norm.fit(all_data['Square'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')

plt.legend(
    [f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'])
plt.ylabel('Frequency')
plt.title('Square distribution')

# QQ-plot
fig = plt.figure()
res = stats.probplot(all_data['Square'], plot=plt)
plt.show()


# In[ ]:


def df_fix_square_manual(df):
    df.loc[df['Square'] > 400, 'Square'] = df.loc[df['Square'] > 400, 'Square'] / 10
    return df


# In[ ]:


def df_fix_square(df):
    info_by_district_id = df.groupby(['DistrictId', 'Rooms', 'HouseYear'], as_index=False).agg(
        {'Square': 'mean'}).rename(
        columns={'Square': 'mean_square_rooms_dr'})

    df = pd.merge(df, info_by_district_id, on=[
        'DistrictId', 'Rooms', 'HouseYear'], how='left')

    df.loc[abs(df['Square'] - df['mean_square_rooms_dr']) > 2 * sigma, 'Square']         = df.loc[abs(df['Square'] - df['mean_square_rooms_dr']) > 2 * sigma, 'Rooms']         * df.loc[abs(df['Square'] - df['mean_square_rooms_dr']) > 2 * sigma, 'mean_square_per_room_in_dr']
    return df


# In[ ]:


def prepare_lifesquare(df):
    df.loc[df['Square'] < df['LifeSquare'],
           'LifeSquare'] = df.loc[df['Square'] < df['LifeSquare'], 'Square']
    return df


def fillna_life_square(df):
    df['LifeSquare'] = df['LifeSquare'].fillna(df['LifeSquare'].mean())
    return df


# ## Fix HouseYear

# In[ ]:


all_data.loc[all_data['HouseYear'] > 2020]


# In[ ]:


def df_fix_house_year_manual(df):
    df.loc[df['HouseYear'] == 20052011, 'HouseYear'] = int((2005 + 2011) / 2)
    df.loc[df['HouseYear'] == 4968, 'HouseYear'] = 1968
    return df


# # Data Processing

# ## Load packages

# In[ ]:


import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


# # Target Variable¶

# ## Normal distribution of the target variable 

# In[ ]:


sns.distplot(train['Price'], fit=norm)

mu, sigma = norm.fit(train['Price'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')

plt.legend(
    [f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

# QQ-plot
fig = plt.figure()
res = stats.probplot(train['Price'], plot=plt)
plt.show()


# ## Log distribution of the target variable

# In[ ]:


price_log = np.log1p(train['Price'])
sns.distplot(price_log, fit=norm)

mu, sigma = norm.fit(train['Price'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')

plt.legend(
    [f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

# QQ-plot
fig = plt.figure()
res = stats.probplot(price_log, plot=plt)
plt.show()


# ## Distribution of variable House Year

# In[ ]:


plt.figure(figsize=(18, 6))
sns.barplot(train['HouseYear'], train['Price'])
plt.xticks(rotation=90)
plt.title('Distribution of variable House Year')


# ## Distribution of variable District Id

# In[ ]:


plt.figure(figsize=(18, 6))
sns.barplot(train['DistrictId'], train['Price'])
plt.xticks(rotation=90)
plt.title('Distribution of variable District Id')


# # Features engineering

# ## Missing Data

# In[ ]:


all_data = pd.concat((train, test), sort=False).reset_index(drop=True)
all_data.drop(['Price'], axis=1, inplace=True)
print(f'all_data size is : {all_data.shape}')


# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(
    all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data


# In[ ]:


def df_del_missing(df):
    df_na = (df.isnull().sum() / len(df)) * 100

    df_na = df_na.drop(
        df_na[df_na == 0].index).sort_values(ascending=False)
    df_na = list(df_na.index)
    df.drop(df_na, axis=1, inplace=True)
    return df


# ## Data Correlation

# In[ ]:


corrmat = train.loc[:, train.columns != 'Id'].corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[ ]:


corrmat = train.loc[:, train.columns != 'Id'].corrwith(
    train['Price']).abs().sort_values(ascending=False)[1:]
plt.bar(corrmat.index, corrmat.values)
plt.title('Correlation to Price')
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


train.head()


# ## Cluster

# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


train_cluster = train.copy()


# In[ ]:


train_cluster = df_fix_house_year_manual(train_cluster)


# In[ ]:


train_cluster_scaled = pd.DataFrame(scaler.fit_transform(
    train_cluster.loc[:, ['HouseYear', 'Price']]), columns=['HouseYear', 'Price'])


# In[ ]:


inertias = []

for i in range(2, 10):
    temp_model = KMeans(n_clusters=i, random_state=100)
    temp_model.fit(train_cluster_scaled)
    temp_inertia = temp_model.inertia_
    inertias.append(temp_inertia)

plt.plot(range(2, 10), inertias)
plt.title('Inertia')

plt.show()


# In[ ]:


plt.scatter(train_cluster_scaled['HouseYear'], train_cluster_scaled['Price'])
plt.xlabel('HouseYear')
plt.ylabel('Price')
plt.show()


# In[ ]:


kmeans_model = KMeans(n_clusters=5, random_state=100)


# In[ ]:


train_labels = kmeans_model.fit_predict(train_cluster_scaled)


# In[ ]:


plt.scatter(train_cluster_scaled['HouseYear'],
            train_cluster_scaled['Price'], c=train_labels)

plt.xlabel('HouseYear')
plt.ylabel('Price')

plt.title('Train data')


# In[ ]:


agglomerative_clustering_model = AgglomerativeClustering(n_clusters=5)


# In[ ]:


train_cluster['cluster_year'] = agglomerative_clustering_model.fit_predict(
    train_cluster_scaled)


# In[ ]:


plt.scatter(train_cluster['HouseYear'],
            train_cluster['Price'], c=train_cluster['cluster_year'])
plt.xlabel('HouseYear')
plt.ylabel('Price')
plt.title('Train')


# In[ ]:


def add_cluster_year(df):
    df_scaled = pd.DataFrame(scaler.fit_transform(
        df.loc[:, ['HouseYear']]), columns=['HouseYear'])
    df['cluster_year'] = agglomerative_clustering_model.fit_predict(df_scaled)
    return df


# ## Mean price by Rooms and Mean price by DistrictId and Rooms

# In[ ]:


def add_mean_price(df, df_train=train):
    price = df_train['Price'].mean()
    price_mean_by_rooms = df_train.groupby(['Rooms'], as_index=False).agg({'Price': 'mean'}).        rename(columns={'Price': 'mean_price_by_rooms'})

    price_mean_by_distr_rooms = df_train.groupby(['DistrictId', 'Rooms'], as_index=False).agg({'Price': 'mean'}).        rename(columns={'Price': 'mean_price_dr'})

    df = pd.merge(df, price_mean_by_distr_rooms, on=[
                  'DistrictId', 'Rooms'], how='left')
    df = pd.merge(df, price_mean_by_rooms, on='Rooms', how='left')
    df['mean_price_dr'] = df['mean_price_dr'].fillna(df['mean_price_by_rooms'])
    df['mean_price_dr'] = df['mean_price_dr'].fillna(price)
    df['mean_price_by_rooms'] = df['mean_price_by_rooms'].fillna(price)
    return df


# ## Large district

# In[ ]:


def add_distr_info(df):
    distr_info = df['DistrictId'].value_counts().reset_index().        rename(columns={"index": "DistrictId", "DistrictId": 'large_district'})
    df = pd.merge(df, distr_info, on='DistrictId', how='left')
    df['large_district'] = df['large_district'].fillna(1)
    return df


# # Modelling

# ## Function

# In[ ]:


def data_prepare(df, df_train=train):
    df = df_fix_square_manual(df)
    df = df_fix_house_year_manual(df)
    df = df_fix_room(df)
    df = df_fix_square(df)
    df = prepare_lifesquare(df)
    df = fillna_life_square(df)
    df = df_del_missing(df)
    df = add_cluster_year(df)
    df = add_mean_price(df, df_train)
    df = add_distr_info(df)
    df = pd.get_dummies(df)
    df.drop('mean_square_per_room_in_dr', axis=1, inplace=True)
    df.drop('mean_square_rooms_dr', axis=1, inplace=True)
    optimizing_df(df)
    return df


def model_test(model, name, test, valid):
    model_pred = model.predict(test)
    r2 = r2_score(valid, model_pred)
    mse = mean_squared_error(valid, model_pred)
    plt.scatter(valid, (model_pred - valid))
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.title(name)
    plt.legend([f'R2= {r2:.4f} and mse= {mse:.0e}'])
    plt.axhline(0, color='red')
    plt.show()


def model_top_deviation(model, test, valid):
    model_pred = model.predict(test)
    model_test = test.copy()
    model_test['Price'] = model_pred
    model_test['Price_test'] = valid
    model_test['SD'] = abs(model_test['Price']
                           - model_test['Price_test'])
    return model_test.nlargest(10, 'SD')


# ## Load packages

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


# ## Data processing

# In[ ]:


print(train.columns)


# In[ ]:


features = list(train.loc[:, train.columns != 'Id'].corrwith(
    train['Price']).abs().sort_values(ascending=False)[1:].index)

target = 'Price'


# In[ ]:


train[features].head()


# In[ ]:


models_dict = {}


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train[features], train[target], test_size=0.3, random_state=42)


# In[ ]:


X_train = data_prepare(X_train, train)
X_test = data_prepare(X_test, train)


# In[ ]:


X_train.info()


# In[ ]:


X_test.info()


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ## Linear Regression

# In[ ]:


line_regression_model = LinearRegression()
line_regression_model.fit(X_train, y_train)


# In[ ]:


models_dict['Linear Regression'] = line_regression_model


# ## Test Linear Regression

# In[ ]:


model_test(line_regression_model, 'Linear Regression', X_test, y_test)


# In[ ]:


model_top_deviation(line_regression_model, X_test, y_test)


# # Random Forest Regressor

# In[ ]:


random_forest_regressor_model = RandomForestRegressor()
random_forest_regressor_model.fit(X_train, y_train)


# In[ ]:


models_dict['Random Forest Regressor'] = random_forest_regressor_model


# ## Test Random Forest Regressor

# In[ ]:


model_test(random_forest_regressor_model,
           'Random Forest Regressor', X_test, y_test)


# In[ ]:


model_top_deviation(random_forest_regressor_model, X_test, y_test)


# # Gradient Boosting Regressor

# In[ ]:


gradient_boosting_regressor_model = GradientBoostingRegressor()
gradient_boosting_regressor_model.fit(X_train, y_train)


# In[ ]:


models_dict['Gradient Boosting Regressor'] = gradient_boosting_regressor_model


# ## Test Gradient Boosting Regressor

# In[ ]:


model_test(gradient_boosting_regressor_model,
           'Gradient Boosting Regressor', X_test, y_test)


# In[ ]:


model_top_deviation(gradient_boosting_regressor_model, X_test, y_test)


# # LassoCV

# In[ ]:


lasso_cv_model = LassoCV()
lasso_cv_model.fit(X_train, y_train)


# In[ ]:


models_dict['LassoCV'] = lasso_cv_model


# ## Test LassoCV

# In[ ]:


model_test(lasso_cv_model, 'LassoCV', X_test, y_test)


# In[ ]:


model_top_deviation(lasso_cv_model, X_test, y_test)


# In[ ]:


all_data.loc[all_data['KitchenSquare'] < 3]


# # LGBMRegressor

# In[ ]:


lgbm_regressor_model = LGBMRegressor()
lgbm_regressor_model.fit(X_train, y_train)


# ## Test LGBMRegressor

# In[ ]:


model_test(lgbm_regressor_model, 'LGBMRegressor', X_test, y_test)


# In[ ]:


model_top_deviation(lgbm_regressor_model, X_test, y_test)


# ## Tunning LGBMRegressor

# In[ ]:


lgbm_regressor_model.get_params


# In[ ]:


np.arange(0.01, 0.05, 0.01)


# In[ ]:


parameters = [{
    'max_bin': np.arange(90, 120, 10),
    'n_estimators': np.arange(4000, 7000, 1000),
    'learning_rate': np.arange(0.01, 0.05, 0.01)
}]


# In[ ]:


clf = GridSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_grid=parameters,
    scoring='neg_mean_squared_error',
    cv=4,
    n_jobs=-1,
)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


cv_results = pd.DataFrame(clf.cv_results_)
param_columns = [
    column
    for column in cv_results.columns
    if column.startswith('param_')
]

score_columns = ['mean_test_score']

cv_results = (cv_results[param_columns + score_columns]
              .sort_values(by=score_columns, ascending=False))

cv_results.head(10)


# In[ ]:


clf.best_params_


# ## Test tunning LGBMRegressor

# In[ ]:


lgbm_regressor_model = LGBMRegressor(
    max_bin=110,
    num_leaves=4,
    n_estimators=4000,
    learning_rate=0.01
)
lgbm_regressor_model.fit(X_train, y_train)


# In[ ]:


model_test(lgbm_regressor_model, 'LGBMRegressor', X_test, y_test)


# In[ ]:


models_dict['LGBMRegressor'] = lgbm_regressor_model


# ## XGBRegressor

# In[ ]:


xgboost_model = XGBRegressor()
xgboost_model.fit(X_train, y_train)


# In[ ]:


models_dict['XGBRegressor'] = xgboost_model


# In[ ]:


model_test(xgboost_model, 'XGBRegressor', X_test, y_test)


# In[ ]:


model_top_deviation(xgboost_model, X_test, y_test)


# # Result

# ## Load packages

# In[ ]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# ## Function

# In[ ]:


def models_r2(models, test, valid):
    scores = pd.DataFrame(columns=['name', 'r2', 'mse'])
    for name, model in models.items():
        test_pred = model.predict(test)
        r2 = r2_score(valid, test_pred)
        mse = mean_squared_error(valid, test_pred)
        scores = scores.append(
            {'name': name, 'r2': r2, 'mse': mse}, ignore_index=True)
    scores.sort_values('r2', ascending=False, inplace=True)
    return scores


# ## Choosing the best model

# In[ ]:


models_score_test = models_r2(models_dict, X_test, y_test)
models_score_train = models_r2(models_dict, X_train, y_train)


# In[ ]:


models_score_test[['name', 'r2']]


# In[ ]:


r2_max_test = models_score_test['r2'].max()
r2_max_train = models_score_train['r2'].max()
plt.barh(models_score_test['name'], models_score_test['r2'],
         alpha=0.5, color='red', label=f'Test  Data: R2 max: {r2_max_test:.4f}')
plt.barh(models_score_train['name'], models_score_train['r2'],
         alpha=0.5, color='grey', label=f'Train Data: R2 max: {r2_max_train:.4f}')
plt.title('R2')
plt.legend()
plt.axvline(0.6, color='red')
plt.axvline(r2_max_test, color='yellow')
plt.show()


# In[ ]:


mse_min_test = models_score_test['mse'].min()
mse_min_train = models_score_train['mse'].min()
plt.barh(models_score_test['name'], models_score_test['mse'],
         alpha=0.5, color='red', label=f'Test  Data MSE min: {mse_min_test:.0e}')
plt.barh(models_score_train['name'], models_score_train['mse'],
         alpha=0.5, color='grey', label=f'Train Data MSE min: {mse_min_train:.0e}')
plt.title('Mean squared error')
plt.legend(loc=2)
plt.axvline(mse_min_test, color='yellow')
plt.show()


# In[ ]:


best_model = models_dict['LGBMRegressor']


# In[ ]:


pd.DataFrame({'name': list(X_train.columns),
              'importances': list(best_model.feature_importances_)})


# In[ ]:


model_test(best_model, 'best_model', X_test, y_test)


# # Output Files

# In[ ]:


test = data_prepare(test)


# In[ ]:


test_features = list(X_train.columns)


# In[ ]:


test[test_features].info()


# In[ ]:


test['Price'] = best_model.predict(test[test_features])


# In[ ]:


price_log = np.log1p(test['Price'])
sns.distplot(price_log, fit=norm)

mu, sigma = norm.fit(test['Price'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')

plt.legend(
    [f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

# QQ-plot
fig = plt.figure()
res = stats.probplot(price_log, plot=plt)
plt.show()


# In[ ]:


test[['Id', 'Price']].to_csv('DRubtsov_predictions.csv', index=None)