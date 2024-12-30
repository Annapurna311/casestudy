# EDA on 'titanic.csv'
  # Why do EDA:*Model building  *Analysis and reporting  *Validation assumption  *Handle Missing Values
  #          Feature Engineering  *Detecting outliers

# Column Types: *Numerical, *Categorical, *Mixed 
"""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read csv
df = pd.read_csv('titanic.csv')
print(df.head())
print(df.info())

'''   Column Types:
Numerical- Age, Fare, PassengerId
Categorical- Survived, Pclass, Sex, Sibsp, Parch, Embarked
Mixed- Name, Ticket, Cacin  '''

## Steps of doing Univariate Analysis on Numerical Columns    ('Age')-Column
'''print(df['Age'].describe())         # Descriptive Statistics
df['Age'].plot(kind='hist', bins=20)           # Visualition
plt.show()
print(df['Age'].skew())       # Skewness Calculation
df['Age'].plot(kind='box')
plt.show()
print(df[df['Age']>65])        # Checking Outliers
print(df['Age'].isnull().sum())          # Checking Missing Values
print(df['Age'].isnull().sum()/len(df['Age']))       # % of Missing Values  '''
# Conclission='Age' is normally distributed, 20% of the value are missing, There are some outliers.

# ('Fare')-Column
'''print(df['Fare'].describe())
df['Fare'].plot(kind='kde')
plt.show()
print(df['Fare'].skew())  
df['Fare'].plot(kind='box')
plt.show()
print(df[df['Fare']>250])             # Checking Outliers
print(df['Fare'].isnull().sum())'''
# Conclusion: The data is highly(positively) skewed, Fare col actually contains the group fare 
#   and not the individual fare(This might be an issue),  We need to create a new col called individual fare.

## Steps of doing Univariate Analysis on Categorical Columns    ('Survived')-col
'''print(df['Survived'].describe())              # Descriptive Statistics
print(df['Survived'].value_counts())
df['Survived'].value_counts().plot(kind='bar')         # Visualition
plt.show()
df['Survived'].value_counts().plot(kind='pie', autopct='%0.1f%%')
plt.show()
print(df['Survived'].isnull().sum())           # Missing Values      '''
# Conclusion: More % of the Survived.

# (Pclass)-column
'''print(df['Pclass'].describe())              # Descriptive Statistics
print(df['Pclass'].value_counts())
df['Pclass'].value_counts().plot(kind='bar')              # Visualition
plt.show()
df['Pclass'].value_counts().plot(kind='pie', autopct='%0.1f%%')
plt.show()
print(df['Pclass'].isnull().sum())   '''             # Missing Values
# Conclusion: Surprisingly less people traveled in Pclass 2 and more people traveled in Pclass 1.

# ('Sex')-column
'''print(df['Sex'].describe())
print(df['Sex'].value_counts())
df['Sex'].value_counts().plot(kind='bar')
plt.show()
df['Sex'].value_counts().plot(kind='pie', autopct='%0.1f%%')        # In Percentage
plt.show()
print(df['Sex'].isnull().sum())  '''
# Conclusion: Male are  travelling more than female. there is no null values.

# ('Sibsp')-column
'''print(df['SibSp'].describe())
print(df['SibSp'].value_counts())
df['SibSp'].value_counts().plot(kind='bar')
plt.show()
df['SibSp'].value_counts().plot(kind='pie', autopct='%0.1f%%')
plt.show()
print(df['SibSp'].isnull().sum())'''
# Conclusion: More peole are travelling single.

# ('Parch')-column
'''print(df['Parch'].describe())
print(df['Parch'].value_counts())
df['Parch'].value_counts().plot(kind='bar')
plt.show()
df['Parch'].value_counts().plot(kind='pie', autopct='%0.1f%%')
plt.show()
print(df['Parch'].isnull().sum()) '''
# Conclusion: Parch and SibSp columns can be merged to from a new col call family_size.
#             Create a new col called is_alone.

# ('Embarked)-column
'''print(df['Embarked'].describe())
print(df['Embarked'].value_counts())
df['Embarked'].value_counts().plot(kind='bar')
plt.show()
df['Embarked'].value_counts().plot(kind='pie', autopct='%0.1f%%')
plt.show()
print(df['Embarked'].isnull().sum())  '''

## Steps of doing Univariate Analysis on Mixed Columns 
# Conclusion: Need to Feature Engeneering in the mixed column

## Steps of doing Bivariate Analysis 
'''  type of relationship:
  1. Numerical - Numerical             #Graphs:scatterplot(regression plots),2D histplot,2D KDEplots
  2. Numerical - Categorical           #Graphs:barplot,boxplot,kdeplot,violinplot even scatterplot
  3. Categorical - Categorical  '''    #Graphs:heatmap,stacked barplots,treemaps

# Categorical - Categorical
#     col-'Survived' = 'Pclass'
'''print(pd.crosstab(df['Survived'],df['Pclass'],normalize='columns')*100)     #In %
sns.heatmap(pd.crosstab(df['Survived'],df['Pclass'],normalize='columns')*100)
plt.show()'''
# Conclusion: Pclass 3 was risky to travel, Pclass 1 is safe to travel.

#     col-'Survived' = 'Sex'
'''print(pd.crosstab(df['Survived'],df['Sex']))
sns.heatmap(pd.crosstab(df['Survived'],df['Sex']))
plt.show() '''
# Conclusion: Females are more saved than males.

#      col-'Survived' = 'Emarked'
'''print(pd.crosstab(df['Survived'],df['Embarked'],normalize='columns')*100)
sns.heatmap(pd.crosstab(df['Survived'],df['Embarked'],normalize='columns')*100)
plt.show()
print(pd.crosstab(df['Sex'],df['Embarked'],normalize='columns')*100)
print(pd.crosstab(df['Pclass'],df['Embarked'],normalize='columns')*100) '''
# Conclusion: Pclass1 people are save compared to others.

#  Numerical - Categorical
#        col-'Survived' = 'Age'
'''df[df['Survived'] == 1]['Age'].plot(kind='kde', label='Survived')
df[df['Survived'] == 0]['Age'].plot(kind='kde', label='Not Survived')
plt.legend()
plt.show()  '''

# Feature Engineering on Fare col
'''print(df['SibSp'].value_counts())
print(df[df['SibSp'] == 8])
df['family_size'] = df['SibSp'] + df['Parch'] + 1
print(df['family_size'])
print(df)  

# family_type
# 1 -> alone
# 2-4 -> small
# >5 -> large
def transform_family_size(num):
  if num == 1:
    return 'alone'
  elif num>1 and num <5:
    return 'small'
  else:
    return 'large'
df['family_type'] = df['family_size'].apply(transform_family_size)      # Create additional col-'family_type'
print(df['family_type'])
print(df)

print(pd.crosstab(df['Survived'],df['family_type'],normalize='columns')*100)
df['Surname'] = df['Name'].str.split(',').str.get(0)      # Create additional col - 'Surname'
print(df['Surname'])
print(df)
print(df['Cabin'].value_counts())
print(df['Cabin'].fillna('M'))'''

# Multivariate Analysis
sns.pairplot(df)
plt.show()    """

###------------###------------###-------------###--------------###------------###---------------###--------------###

# EDA on "telecom_cust.csv"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('telecom_cust_churn.csv')
'''print(df.head())
print(df.info())'''

# replacing blanks with 0 and no total charges are recorded
'''df['TotalCharges'] = df['TotalCharges'].replace(" ","0")
df['TotalCharges'] = df['TotalCharges'].astype("float")
print(df['TotalCharges'])

print(df.isnull().sum())
print(df.describe())
print(df['customerID'].duplicated().sum())'''

# Converted 0 and 1 values of SeniorCitizen to yes/no to make easier to understand.
'''def conv(value):
  if value == 1:
    return "yes"
  else:
    return "no"
df['SeniorCitizen'] = df['SeniorCitizen'].apply(conv)
print(df['SeniorCitizen'])
print(df.head(30))'''

'''ax = sns.countplot(x='Churn', data=df)
ax.bar_label(ax.containers[0])
plt.title('Count of Customer Churn')
plt.show()'''

'''plt.figure(figsize = (3,4))
gb = df.groupby('Churn').agg({'Churn':'count'})
plt.pie(gb['Churn'], labels = gb.index, autopct = '%1.2f%%')
plt.title('Percentage of Customeres Churned', fontsize = 10)
plt.show()'''
# from the given chart we can conclude that 26.54% of our customers have churned out.
# not lets explore the reason behind it.

'''plt.figure(figsize = (3, 2))
sns.countplot(x='gender', data=df, hue='Churn')
plt.title('Churn by Gender')
plt.show()'''

'''plt.figure(figsize = (3, 2))
sns.countplot(x='SeniorCitizen', data=df, hue='Churn')
plt.title('Churn by SeniorCitizen')
plt.show()'''

'''plt.figure(figsize = (9, 4))
sns.histplot(x='tenure', data=df, bins=50, hue='Churn')
plt.show()'''
# People who have used our services for a long timr have stayed and people who have used our services
  #1 0r 2 months have churned.
