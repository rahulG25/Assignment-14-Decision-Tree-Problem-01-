#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


df = pd.read_csv("Fraud_check.csv")
df.head()


# In[4]:


df.tail()


# In[5]:


#dropping first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


# In[7]:


#Creating new cols TaxInc and dividing 'Taxable.Income' for Risky and Good labels
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])


# In[8]:


print(df)


# Lets assume: taxable_income <= 30000 as “Risky=0” and others are “Good=1”

# In[9]:


#After creating new col. TaxInc its also made dummies variable
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# In[10]:


df.tail(10)


# In[11]:


#plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')


# In[12]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[13]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[14]:


# Declaring features & target
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# In[17]:


##Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"


# In[18]:


##Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)


# In[19]:


df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode


# In[20]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


# In[22]:


##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[23]:


## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[25]:


##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[26]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[27]:


model.n_outputs_


# In[31]:


model.oob_score_


# In[36]:


#Predictions on train data
prediction = model.predict(x_train)


# In[37]:


#Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)


# In[38]:


np.mean(prediction == y_train)


# In[39]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[40]:


#Prediction on test data
pred_test = model.predict(x_test)


# In[42]:


#Accuracy
acc_test =accuracy_score(y_test,pred_test)
#78.333%


# In[44]:


# In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO


# In[50]:


tree = model.estimators_[5]


# In[54]:


dot_data = StringIO()
export_graphviz (tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)


# In[55]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# # Building Decision Tree Classifier using Entropy Criteria

# In[56]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[57]:


from sklearn import tree


# In[58]:


#PLot the decision tree
tree.plot_tree(model);


# In[59]:


colnames = list(df.columns)
colnames


# In[60]:


fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[61]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[62]:


preds


# In[63]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[64]:


# Accuracy 
np.mean(preds==y_test)


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[65]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[66]:


model_gini.fit(x_train, y_train)


# In[67]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# # Decision Tree Regression Example

# In[68]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[69]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[71]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[72]:


#Find the accuracy
model.score(X_test,y_test)

