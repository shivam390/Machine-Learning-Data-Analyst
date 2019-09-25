import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
import os
os.chdir('/Users/shivamgautam/Downloads/edureka ML and AI/practise work/')
df=pd.read_csv('train.csv')
df.head()
df["datetime"] = pd.to_datetime(df.datetime)
df["date"]=df.datetime.dt.date
# ### Question 1: Create the following features from the column "datetime" in the above dataframe
# #### 1) Hour: Hour in the datetime, e.g., datetime = 5/2/2012 19:00, hour = 19
# #### 2) weekday: e.g., datetime = 5/2/2012 19:00, weekday = Wednesday	
# #### 3) Month: e.g., datetime = 5/2/2012 19:00, Month = May
# #### 4) date: e.g., datetime = 5/2/2012 19:00, date = 5/2/2012
df['Hour']=df.datetime.dt.hour
df['Weekday']=df.datetime.dt.weekday_name
df['Month']=df.datetime.dt.month_name()
df.head()
# ### Question 2: Find the average number of bookings in each month across years and sort the data by average number of bookings in descending order
df['year']=df.datetime.dt.year
booking_sum=df.groupby(['year','Month'])['Total_booking'].sum()
yr_month_booking=booking_sum.sort_values(ascending=False)
yr_month_booking
## Question 3 
# #### 1) Convert categorical variables into one-hot encoded features
# #### 2) Concat one hot enoded features into original dataframe
# #### 3) Remove the original columns from the dataframe
categoryvariablelist=["Weekday","Month","season","weather"]
df.dtypes
le = LabelEncoder()
ohe = OneHotEncoder()
for var in categoryvariablelist:
    print(var)
    df[var]=df[var].astype('category')    
    df[var] = df[[var]].apply(lambda col:le.fit_transform(col))
    ohe = OneHotEncoder(categorical_features=var,sparse=False)
df.dtypes
# Categorical boolean mask
#categorical_feature_mask = X.dtypes==object
# filter categorical columns using mask and turn it into a list
#categorical_cols = X.columns[categorical_feature_mask].tolist()
#categorical_feature_mask = df.dtypes==object

#categorical_cols = df.columns[categorical_feature_mask].tolist()
#categorical_cols

# instantiate OneHotEncoder
#ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 
# categorical_features = boolean mask for categorical columns
# sparse = False output an array not sparse matrix

df=df.drop(['datetime','workingday','temp','atemp','windspeed'],axis=1)

x_ohe = ohe.fit_transform(df)

dfnew = pd.get_dummies(df, columns = ["Weekday","Month","season","weather"])




