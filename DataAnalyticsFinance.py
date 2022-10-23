# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 10:16:05 2022

@author: londoncm
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

# Change path
current_path = os.getcwd()
current_path

# Adding the path where is the data located

os.chdir(r'C:\Users\LONDONCM\Documents\Personal\Python course\Finance Project') 

os.listdir()


# Read data

df = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx', 1 )
df.head()

df.shape

# Seeing if there are null values
df.isnull().sum()


# Remove data

df.drop(['ID', 'ZIP Code'], axis= 1, inplace=True) 

df.columns    

# Summmary to get description about data 



fig = px.box(df, y=['Age', 'Experience', 'Income', 'Family', 'Education'])
fig.show()
                    
# ------------ Understanding Data & data -preprocessing------------

# Visualise distribution of data

df.skew()

df.dtypes

df.hist(figsize=(20,20))

sns.displot(df['Experience'])

df['Experience'].mean()

Negative_exp = df[df['Experience']<0]
Negative_exp.head()

sns.displot(Negative_exp['Age'])

# There are errors in the data because appears negative experience for certain age.

Negative_exp['Experience'].mean()

Negative_exp.size

print('There are {} records which has negative values for experience, approx {} %'.format(Negative_exp.size, ((Negative_exp.size/df.size)*100)))

data=df.copy()
data.head()

# Reeplace negative data with the mean

data['Experience'] = np.where(data['Experience']<0,data['Experience'].mean(),data['Experience'])

data[data['Experience']<0]


# Correlation of data and Analyzing Education status of customers

# Correlation of data

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)

# Age and experience are highly correlated. It means is fine to go with experience and drop experience to avoid multicollinearity.

data.drop(['Experience'], axis=1)
data.head()


# Analyzing Education status of customers

data['Education'].unique()
    
def mark(x):
    if x==1:
        return 'Undergrad'
    elif x==2:
        return 'Graduate'
    else:
        return 'Advanced/Professional'

data['Edu_mark']=data['Education'].apply(mark)

data.head()

EDU_dis= data.groupby('Edu_mark')['Age'].count()
EDU_dis
    

fig2 = px.pie(data,values=EDU_dis, names=EDU_dis.index,title='Pie Chart')
fig2.show()

# Explore the account holder distribution

data.columns

def Security_CD(row):
    if(row['Securities Account']==1) & (row['CD Account']==1):
        return 'Holds Securities & Deposit'
    elif (row['Securities Account']==0) & (row['CD Account']==0):
        return 'Does not Holds Securities or Deposit'
    elif (row['Securities Account']==1) & (row['CD Account']==0):
        return 'Holds only Securities'
    elif (row['Securities Account']==0) & (row['CD Account']==1):
        return 'Holds only Deposit'

data['Account_holder_category']=data.apply(Security_CD, axis=1)

data.head()

values = data['Account_holder_category'].value_counts()
values.index

fig3 = px.pie(data,values=values, names=values.index,title='Pie Chart')
fig3.show()
    

# Analyse customer on the basis of their Education Status, income & Personal Loan Status

fig4 = px.box(data,x='Education',y='Income',facet_col='Personal Loan')


plt.figure(figsize=(12,8))
sns.displot(data[data['Personal Loan']==0]['Income'],label='Income with no personal loan')
sns.displot(data[data['Personal Loan']==1]['Income'],label='Income with personal loan')
plt.legend()

# Automate the Analysis.

def plot(col1,col2,label1,label2,title):
    plt.figure(figsize=(12,8))
    sns.displot(data[data['col2']==0]['col1'],hist=False,label=label1)
    sns.displot(data[data['col2']==1]['col1'],hist=False,label=label2)
    plt.legend()
    plt.title(title)

plot('Income','Personal Loan','Income with no personal loan','Income with personal loan','Income Distribution')

plot('Mortgage','Personal Loan','Mortgage of customers with no personal loan','Mortgage of customers with personal loan','Mortgage of customers Distribution')


# Analyse categories of customers on the basis of Security account, online, Account_holder & credit card.

data.columns
col_names=['Securities Account','Online','Account_holder_category','CreditCard']

for i in col_names:
    plt.figure(figsize=(10,5))
    sns.countplot(x=i,hue='Personal Loan',data=data)


# How age of a person is going to be a factor in available loan?

sns.scatterplot(data['Age'],data['Personal Loan'],hue=data['Family'])

import scipy.stats as stats

Ho='Age does not have to impact on availing personal loan'
Ha='Age does have to impact on availing personal loan'

Age_no = np.array(data[data['Personal Loan']==0]['Age'])
Age_yes = np.array(data[data['Personal Loan']==0]['Age'])

t,p_value = stats.ttest_ind(Age_no,Age_yes,axis=0)

if p_value<0.05:
    print(Ha, 'as the p_value is less than 0.05 with a value of{} '.format(p_value))
else:
    print(Ho, 'as the p_value is greater than 0.05 with a value of{} '.format(p_value))
    

# Automate above stuffs

def Hypothesis(col1,col2,Ho,Ha):
    arr1 = np.array(data[data[col1]==0][col2])
    arr2 = np.array(data[data[col1]==0][col2])
    t,p_value = stats.ttest_ind(arr1,arr2,axis=0)
    if p_value<0.05:
        print('{}, as the p_value is less than 0.05 with a value of{} '.format(Ha,p_value))
    else:
        print('{}, as the p_value is greater than 0.05 with a value of{} '.format(Ho,p_value))
        
    
Hypothesis('Personal Loan','Age',Ho='Age does not have to impact on availing personal loan',Ha='Age does have to impact on availing personal loan')


# Does income of a person hace an impact on availing loan?

Hypothesis(col1='Personal Loan',col2='Income',Ho='Income does not have to impact on availing personal loan',Ha='Income does have to impact on availing personal loan')

# Does the family size makes them to avoid loan?

Hypothesis(col1='Personal Loan',col2='Family',Ho='Family does not have to impact on availing personal loan',Ha='Family does have to impact on availing personal loan')





















