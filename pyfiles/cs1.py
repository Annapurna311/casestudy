import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# read data
df = pd.read_csv('student_data.csv')
print(df.head())
print(df.info())

# 1. Data Exploration and Preprocessing
# Handle missing values through imputation or removal.
print(df.isnull().sum())

# Standardize or normalize numerical features (e.g., Study Hours, Sleep Hours, etc.).
  # Select numerical columns
'''numerical_features = ['Study Hours', 'Sleep Hours']  # Update based on your dataset
# Standardization
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
# Optionally, normalization (if required for specific algorithms)
# normalizer = MinMaxScaler()
# df[numerical_features] = normalizer.fit_transform(df[numerical_features])
print("After Standardization:")
print(df[numerical_features].head())

# Encode categorical variables, such as Parental Education Level, if they are present.
from sklearn.preprocessing import LabelEncoder
# Check if 'Parental Education Level' exists in the dataset
if 'Parental Education Level' in df.columns:
    # Perform Label Encoding
    label_encoder = LabelEncoder()
    df['Parental Education Level Encoded'] = label_encoder.fit_transform(df['Parental Education Level'])

    print("Label Encoded Data:")
    print(df[['Parental Education Level', 'Parental Education Level Encoded']].head())
else:
    print("'Parental Education Level' column not found in the dataset.") ''' 

# 2. Exploratory Data Analysis (EDA)
# Visualize relationships between Final Exam Score and other features using scatter plots or correlation matrices.
# Scatter plots for numerical features vs. Final Exam Score
'''numerical_features = ['Study Hours', 'Sleep Hours']  # Replace with your numerical feature names
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=feature, y='Final Exam Score')
    plt.title(f'Relationship between {feature} and Final Exam Score')
    plt.xlabel(feature)
    plt.ylabel('Final Exam Score')
    plt.show()

# Analyze the distribution of scores, study hours, and attendance rates.
# Analyze the Distribution of Final Exam Scores
plt.figure(figsize=(8, 5))
sns.histplot(df['Final Exam Score'], kde=True, color='blue', bins=20)
plt.title('Distribution of Final Exam Scores')
plt.xlabel('Final Exam Score')
plt.ylabel('Frequency')
plt.show()

# Analyze the Distribution of Study Hours
plt.figure(figsize=(8, 5))
sns.histplot(df['Study Hours'], kde=True, color='green', bins=20)
plt.title('Distribution of Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Frequency')
plt.show()  '''

# Analyze the Distribution of Attendance Rates
'''plt.figure(figsize=(8, 5))
sns.histplot(df['Attendance Rate (%)'], kde=True, color='orange', bins=20)
plt.title('Distribution of Attendance Rates')
plt.xlabel('Attendance Rate (%)')
plt.ylabel('Frequency')
plt.show()
# Summary Statistics
features = ['Final Exam Score', 'Study Hours', 'Attendance Rate (%)']  # Adjust based on your dataset
print(df[features].describe())'''

# Check for outliers that might affect the model’s performance, such as extreme values in Part-time Work Hours or Social Media Usage.
# List of features to check for outliers
# List of features to check for outliers
'''outlier_features = ['Part-time Work Hours', 'Social Media Usage']  # Adjust based on your dataset
for feature in outlier_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=feature, color='lightblue')
        plt.title(f'Boxplot of {feature}')
        plt.xlabel(feature)
        plt.show()
    else:
        print(f"'{feature}' column not found in the dataset.")'''

# 3. Feature Engineering
"""  	Study-to-Social Ratio: Ratio of study hours to social media usage.
     	Sleep-to-Activity Ratio: Ratio of sleep hours to extracurricular plus part-time work hours.
"""
# Apply transformations to skewed data if needed, such as a log transformation on hours-based features.
# Check skewness of hours-based features
'''hours_features = ['Study Hours', 'Social Media Usage', 'Part-time Work Hours', 'Sleep Hours']
for feature in hours_features:
    if feature in df.columns:
        skewness = df[feature].skew()
        print(f"Skewness of {feature}: {skewness}")
    else:
        print(f"'{feature}' column not found in the dataset.")

# Apply Log Transformation
for feature in hours_features:
    if feature in df.columns:
        # Apply log transformation to features with significant skewness
        df[f'Log_{feature}'] = np.log1p(df[feature])  # Use log1p to handle zero values
        print(f"Applied log transformation to {feature}.")
# Display the first few rows of the dataset with new features
print(df.head())

# Visualize distributions of new features
new_features = ['Study-to-Social Ratio', 'Sleep-to-Activity Ratio']
transformed_features = [f'Log_{feature}' for feature in hours_features if f'Log_{feature}' in df.columns]
for feature in new_features + transformed_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, bins=20, color='cyan')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()'''

# 4.	Split the Data
#Divide the data into training and testing sets to validate model performance.
# Define features and target
X = df.drop(columns=['Final Exam Score'])
y = df['Final Exam Score']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")  

# 5.Build the Linear Regression Model
# o	Fit a linear regression model using the training data, with Final Exam Score as the target variable.
model = LinearRegression()
model.fit(X_train, y_train)
