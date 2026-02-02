import pandas as pd
from pathlib import Path


df = pd.read_csv("C:/Users/computer world/Desktop/Projects/Customer Churn/data/customer_churn_dataset.csv")

df.shape
df.head()
df.tail()
df.describe()

print("Churn Counts:")
print(df['churn'].value_counts())

print("\nChurn Ratio:")
print(df['churn'].value_counts(normalize=True))

print("\nDuplicate Rows:")
print(df.duplicated().sum())

X = df.drop('churn', axis=1)
y = df['churn']

print(X.isnull().sum().sum())  
print(X.dtypes)              

df.to_csv("C:/Users/computer world/Desktop/Projects/Customer Churn/data/clean_churn.csv", index=False) 
print("Data saved to clean_churn.csv")
