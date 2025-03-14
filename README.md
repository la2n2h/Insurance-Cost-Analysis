# Insurance-Cost-Analysis

dataset : https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?resource=download&select=kc_house_data.csv

![image](https://github.com/user-attachments/assets/d2a96a43-dd56-4037-baf0-a485ee8351b3)

# DATA PREPARE

### Importing Required Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
```

### Download Dataset
```
df = pd.read_csv('D:/Data anlysis- working sheet/python/data/kc_house_data.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv')
```

### Save raw data into computer
```
df.to_csv('D:/Data anlysis- working sheet/python/data/insurance_cost.csv')
```
# DATA PROCESS
### identify and handle mistakes
```
# Print the first 10 rows of the dataframe
df.head()
```
![image](https://github.com/user-attachments/assets/f53355d9-97e2-4276-b2d5-ec308b3593af)

The raw data have no column titles yet.

```
# insert column titles
df.columns = ['Age', 'Gender', 'BMI', 'No_of_children', 'Smoker', 'Region', 'Charges']
```
```
# check size of data
df.shape
```
![image](https://github.com/user-attachments/assets/041fd69d-4852-4da4-aa51-6fc53ae79215)
the dataset have 2771 rows and 7 column

```
# check data type
df.dtypes
```
![image](https://github.com/user-attachments/assets/623f84be-ff14-48d2-8627-b006d4bfd476)
Age, Smoker: object, should change to int

```
# change data type
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('Int64')
df['Smoker'] = pd.to_numeric(df['Smoker'], errors='coerce').astype('Int64')
```

```
# change ? into np.nan in Smoker colume
df['Smoker'] = df['Smoker'].replace({'?': np.nan})
```

```
# check null data
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a6d57b64-7704-47ad-9f33-af7f62a01202)

Age have 4 missing value  and Smoker have 7 missing value

```
# fill mean value for Age and fill most frequency value for Smoker
df['Age'] = df['Age'].fillna(df['Age'].mean().astype(int))
df['Smoker'] = df['Smoker'].fillna(df['Smoker'].value_counts().idxmax())
```
### Overview of the Data after clean
```
# Print the statistical description of the dataset, including that of 'object' data types
df.describe(include ='all')
```
![image](https://github.com/user-attachments/assets/70eeb905-2bb5-4d1a-a1cd-11dc50155a65)


Number of observations (count): 2,771 rows, no NaN values.
Mean (mean): The average value for each column.
Standard deviation (std): Measures how spread out the data is.
Min - Max: The smallest and largest values.
Percentiles (25%, 50%, 75%): Show data distribution.

Analysis of Each Column
Column: Age
Mean: 39.12 years.
Range: 18 - 64 years.
Distribution:
50% of people are between 26 and 51 years old.
Median (50%): 39 years → Fairly balanced data.
Std (14.08) → Some variation but not extreme.
Insight: No obvious outliers, distribution looks normal.

Column: Gender (1: Male, 2: Female)
Mean: 1.51 → Close to 1.5, indicating nearly equal male & female ratio.
Std: 0.5 → Two groups are evenly distributed.
Insight: Gender distribution is balanced.

Column: BMI (Body Mass Index)
Mean: 30.7 (Overweight according to BMI standards).
Min - Max: 15.96 → 53.13 (Possible cases of underweight or severe obesity).
Std: 6.13 → Noticeable variation.
Insight: Noticeable variation can be happen due to this data about insurance cost.

Column: No_of_children (Number of Children)
Mean: 1.1 → Most people have 0 to 2 children.
Max: 5 children → Seems reasonable.
Std: 1.21 → Not highly variable.
Insight: Data appears reasonable, no extreme values.

Column: Smoker (0: yes, 1: No)
Mean: 0.203 → About 20% of the dataset consists of smokers.
Std: 0.40 → Clear distinction between smokers and non-smokers.
Min - Max: Only 0 and 1, no incorrect values.
Insight: Most people in the dataset are smokers.

Column: Region (Living Region - 1 to 4)
Mean: 2.56 → No strong bias toward any specific region.
Std: 1.13 → Regions are fairly evenly distributed.
Insight: Checking value_counts() will provide a clearer regional distribution.

Column: Charges (Medical Costs - Target Variable)
Mean: $13,260.
Cost Range: $1,121 → $63,770 (Very large gap).
Std: $12,153 → Huge cost variations.
Percentiles:
25%: $4,687
50%: $9,304
75%: $16,516
Max: $63,770 (Very high, potential outliers)

### Normalization & Standardization the data
Age – Values range from 18 to 64, normalization reduces the impact of large differences.

BMI – Values range widely (15.96 → 53.13), so scaling helps balance feature importance.

Charges – Large range of medical costs ($1,121 → $63,770), making normalization essential.

```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age', 'BMI', 'Charges']] = scaler.fit_transform(df[['Age', 'BMI', 'Charges']])
```




