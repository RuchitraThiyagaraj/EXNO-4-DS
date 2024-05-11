# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
~~~
#DEVELOPED BY : RUCHITRA.T
#REG NO : 212223110043
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
missing=data[data.isnull().any(axis=1)]
data2=data.dropna(axis=0)
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
new_data=pd.get_dummies(data2, drop_first=True)
columns_list=list(new_data.columns)
features=list(set(columns_list)-set(['SalStat']))
y=new_data['SalStat'].values
x=new_data[features].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print("Missclassified samples: %d"%(test_y !=prediction).sum())
~~~
# Output: 
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/a98e1dc5-8b89-439c-877e-66e0e8f5626d)
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/3de78350-4a4d-41f2-bdb8-e7b65e373a4c)
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/8bf7a7af-e808-43bd-8c96-a7536030caa1)
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/211ec793-098a-46c3-9860-0232fb78c734)
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/03f5930c-e904-4ab0-8bab-1758c817eb9b)
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/7de4689f-0a84-4275-89f3-b59887c070ba)
![image](https://github.com/RuchitraThiyagaraj/EXNO-4-DS/assets/154776996/9e8aa563-7be5-411e-acf8-7311f203fdbf)


# RESULT:
Hence given data is read and performed Feature Scaling and Feature Selection process and saved the data to a file.
