#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


# Load the dataset
df = pd.read_csv(r"C:\Users\rites\OneDrive\Desktop\csv\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display the first few rows and check the structure of the dataset
df.head()


# In[12]:


df.info()


# In[11]:


# Data Cleaning and Preparation
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values or impute them based on the context of your dataset
df.dropna(inplace=True)

# Convert relevant columns to appropriate data types if needed (e.g., dates, categorical variables)
# Example:
# df['join_date'] = pd.to_datetime(df['join_date'])

# Check for duplicates and remove if necessary
df.drop_duplicates(inplace=True)


# In[15]:


#Exploratory Data Analysis
# Summary statistics
print(df.describe())

# Visualize distributions and relationships
# Example:
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Example of correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[24]:


#Performance Metrics Analysis
# Example: Average performance score by department
avg_performance_by_dept = df.groupby('Department')['PerformanceRating'].mean().reset_index()

# Plotting average performance by department
plt.figure(figsize=(8, 4))
sns.barplot(x='Department', y='PerformanceRating', data=avg_performance_by_dept)
plt.title('Average PerformanceRating by Department')
plt.xlabel('Department')
plt.ylabel('Average PerformanceRating')
plt.xticks(rotation=15)
plt.show()


# In[36]:


import statsmodels.api as sm

df = pd.read_csv(r"C:\Users\rites\OneDrive\Desktop\csv\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# One-hot encode 'EducationField'
df = pd.get_dummies(df, columns=['EducationField'], drop_first=True)

# Prepare variables for regression
X = df[['MonthlyIncome', 'Age'] + [col for col in df.columns if 'EducationField_' in col]]
y = df['PerformanceRating']

# Add constant to X
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())


# In[41]:


# Example: Interactive dashboard or detailed report using Power BI or Python dashboards
# This step involves creating visualizations and a narrative based on your findings.
# Use libraries like Power BI or build dashboards using Python (e.g., Dash).

# Example of creating a dashboard or report using Python libraries
# - Create multiple plots and arrange them
# - Export or display in a Jupyter Notebook or as HTML

# Example:
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Scatter plot of salary vs performance score
axes[0, 0].scatter(df['MonthlyIncome'], df['PerformanceRating'])
axes[0, 0].set_title('MonthlyIncome vs PerformanceRating')
axes[0, 0].set_xlabel('MonthlyIncome')
axes[0, 0].set_ylabel('PerformanceRating')

# Plot 2: Box plot of performance score by education level
sns.boxplot(x='Education', y='PerformanceRating', data=df, ax=axes[0, 1])
axes[0, 1].set_title('PerformanceRating by Education')
# Plot 3: Distribution of age
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Age')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('BusinessTravel')

# Plot 4: Bar plot of average performance score by department
sns.barplot(x='Department', y='PerformanceRating', data=avg_performance_by_dept, ax=axes[1, 1])
axes[1, 1].set_title('Average PerformanceRating by Department')
axes[1, 1].set_xlabel('Department')
axes[1, 1].set_ylabel('Average PerformanceRating')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

