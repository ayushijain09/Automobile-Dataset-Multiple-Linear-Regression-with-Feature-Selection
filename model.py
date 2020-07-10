#!/usr/bin/env python
# coding: utf-8

# In[229]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[230]:


import warnings
warnings.filterwarnings('ignore')


# In[231]:


# Reading the dataset
cars = pd.read_csv("CarPrice_Assignment.csv")


# In[232]:


cars.head()


# In[233]:


#Summary of dataset
print(cars.info())


# ### Data Exploration

# In[234]:


#Remove the variable such as car_ID that straight away makes no sense, since it wont be adding any value in determining the price
cars = cars.drop(['car_ID'],axis=1)


# ##### To perform linear regression, we need to figure out on which feature the target variable depends on. We can a make a pair wise plot on numerical variables

# In[235]:


cars.head()


# In[236]:


cars_num = cars.select_dtypes(include=['float64','int64'])


# In[237]:


cars_num.head()


# In[238]:


#pairplot
plt.figure(figsize=(20, 10))
sns.pairplot(cars_num)
plt.show()


# #####  Very difficult to infer! Sometimes, when we have a lot of features, it is better to use correlation matrix or heatmap instead of pairplot. 

# In[239]:


corMatrix = cars_num.corr()
corMatrix


# ##### or maybe lets plot the correlations on heatmap for better visualization!

# In[240]:


plt.figure(figsize=(16,8))
sns.heatmap(corMatrix, cmap="YlGnBu", annot=True)    #cmap = color map 
plt.show()


# ###### You can infer from the last row that how wheelbase, carlength, carwidth, curbweight, enginesize, horsepower are highly correlated with price, while, peakrpm, citympg, highwaympg are negatively correlated with price.
# 
# Also, there is a high correlation among the independent variables which leads to the problem of Multicollinearity. This always needs to be handled if its too severe.

# ### Data Cleaning
# 
# 1. Instead of the car name, we can keep company names to reduce categories

# In[241]:


cars['CarName']


# In[242]:


cars['carCompany'] = cars['CarName'].apply(lambda x: x.split(" ")[0])


# In[243]:


cars['carCompany']


# In[244]:


cars['carCompany'].astype('category').value_counts()


# #### Few companies are misspelled! Lets correct them.

# In[245]:


# volkswagen
cars.loc[(cars['carCompany'] == "vw") | (cars['carCompany'] == "vokswagen"), 'carCompany'] = 'volkswagen'

# porsche
cars.loc[cars['carCompany'] == "porcshce", 'carCompany'] = 'porsche'

# toyota
cars.loc[cars['carCompany'] == "toyouta", 'carCompany'] = 'toyota'

# nissan
cars.loc[cars['carCompany'] == "Nissan", 'carCompany'] = 'nissan'

# mazda
cars.loc[cars['carCompany'] == "maxda", 'carCompany'] = 'mazda'


# In[246]:


cars['carCompany'].astype('category').value_counts()


# #### Now that we have company names, lets drop carName column

# In[247]:


cars=cars.drop('CarName', axis=1)


# In[248]:


cars.head()


# #### There are 2 columns 'doornumber' and 'columnnumber' are numeric types which are mentioned as words. Lets change them into numeric data type.

# In[249]:


cars['cylindernumber'].astype('category').value_counts()


# In[250]:


cars['doornumber'] = cars['doornumber'].map({'four':4,'two':2})
cars['cylindernumber'] = cars['cylindernumber'].map({'four':4,'two':2,'six':6,'five':5,'eight':8,'three':3,'twelve':12})


# #### Now, lets focus on categorical variables. Models cant learn from so we need to make dummy variables to convert them into numeric types.

# In[251]:


cars_category = cars.select_dtypes(include=['object'])
cars_category.head()


# In[252]:


cars_dummy = pd.get_dummies(cars_category, drop_first=True)
cars_dummy.head()


# #### Drop Categorical columns and add new dummies

# In[253]:


cars = cars.drop(list(cars_category), axis=1)


# In[254]:


cars = pd.concat([cars,cars_dummy], axis=1)


# In[255]:


cars.head()


# ### Now we have all features as numeric data types.

# #### Before building a model lets scale all the numerical features that we had before. The categorical features dont need to be scaled.
# But we can only scale the training set features and not the test set. Lets first split the dataset.

# In[256]:


from sklearn.cross_validation import train_test_split
df_train, df_test = train_test_split(cars, train_size=0.7, test_size=0.3, random_state=100)


# In[257]:


cars_num.columns


# #### Scaling

# In[258]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale_features = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
       'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price','doornumber','cylindernumber']     ##doornumber and cylindernumber are numeric types as well that were converted from object type
df_train[scale_features] = scale.fit_transform(df_train[scale_features])


# In[259]:


df_train.head()


# In[260]:


missingCol = cars.count().idxmin()
missingCol


# In[261]:


linearmodel = LinearRegression()


# In[262]:


# Split the train and test dataset into X and y

y_train = df_train.pop('price')
X_train = df_train

df_test[scale_features] = scale.fit_transform(df_test[scale_features])

y_test = df_test.pop('price')
X_test = df_test


# In[263]:


linearmodel.fit(X_train,y_train)
predictions = linearmodel.predict(X_test)


# In[266]:


RMSE = np.sqrt(mean_squared_error(y_test, predictions))
R2 = r2_score(y_test, predictions)
print('R2:',R2,'RMSE:',RMSE)


# In[267]:


# Comparison between predicted and actual values

df = pd.DataFrame()
df['y'] = y_test
df['pred'] = predictions
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




