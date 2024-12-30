# Case Study: Predicting Student Performance Based on Study and Lifestyle Habits
# Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Reading Dataset
df = pd.read_csv('student_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# 1.Data Exploration and Preprocessing
# o	Handle missing values through imputation or removal.
'''print(df.isnull().sum())
# No null values

# o	Standardize or normalize numerical features (e.g., Study Hours, Sleep Hours, etc.).
# Standardize numerical features
numerical_features = ['Study Hours', 'Attendance Rate (%)', 'Extracurricular Hours',
                       'Part-time Work Hours', 'Sleep Hours', 'Social Media Usage (hrs/day)']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print(df[numerical_features].head())

# o	Encode categorical variables, such as Parental Education Level, if they are present.
from sklearn.preprocessing import LabelEncoder
# Check if 'Parental Education Level' exists in the dataset
if 'Parental Education Level' in df.columns:
    # Perform Label Encoding
    label_encoder = LabelEncoder()
    df['Parental Education Level Encoded'] = label_encoder.fit_transform(df['Parental Education Level'])
    print("Label Encoded Data:")
    print(df[['Parental Education Level', 'Parental Education Level Encoded']].head())
else:
    print("'Parental Education Level' column not found in the dataset.")  '''

# 2. Exploratory Data Analysis (EDA)
# o	Visualize relationships between Final Exam Score and other features using scatter plots or correlation matrices.
'''numerical_features = ['Study Hours', 'Attendance Rate (%)', 'Extracurricular Hours',
                       'Part-time Work Hours', 'Sleep Hours', 'Social Media Usage (hrs/day)']
sns.pairplot(df, x_vars=numerical_features, y_vars='Final Exam Score', kind='reg')
plt.show()
# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# o	Analyze the distribution of scores, study hours, and attendance rates.
# List of variables to analyze
variables = ['Scores', 'Study Hours', 'Attendance Rate (%)']
# Set up the plot
plt.figure(figsize=(16, 5))
for i, var in enumerate(variables, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[var], kde=True, bins=20, color='orange')
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()    

# o  Check for outliers that might affect the model’s performance, 
# such as extreme values in Part-time Work Hours or Social Media Usage.
outlier_features = ['Part-time Work Hours', 'Social Media Usage']  # Adjust based on your dataset
for feature in outlier_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=feature, color='lightblue')
        plt.title(f'Boxplot of {feature}')
        plt.xlabel(feature)
        plt.show()
    else:
        print(f"'{feature}' column not found in the dataset.")  '''

# 3.Feature Engineering
"""  	Study-to-Social Ratio: Ratio of study hours to social media usage.
     	Sleep-to-Activity Ratio: Ratio of sleep hours to extracurricular plus part-time work hours.
"""
# Apply transformations to skewed data if needed, such as a log transformation on hours-based features.
# Study-to-Social Ratio
df['Study-to-Social Ratio'] = df['Study Hours'] / (df['Social Media Usage'] + 1)

# Sleep-to-Activity Ratio
df['Sleep-to-Activity Ratio'] = df['Sleep Hours'] / (df['Extracurricular Hours'] + df['Part-time Work Hours'] + 1)
print(df['Study-to-Social Ration'])
