# Insurance-Cost-Analysis

dataset : https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?resource=download&select=kc_house_data.csv

| Parameter |Description| Content type |
|---|----|---|
|age| Age in years| integer |
|gender| Male or Female|integer (1 or 2)|
| bmi | Body mass index | float |
|no_of_children| Number of children | integer|
|smoker| Whether smoker or not | integer (0 or 1)|
|region| Which US region - NW, NE, SW, SE | integer (1,2,3 or 4 respectively)| 
|charges| Annual Insurance charges in USD | float|

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
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv')
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
expected_columns = ['Age', 'Gender', 'BMI', 'No_of_children', 'Smoker', 'Region', 'Charges']
if df.shape[1] == len(expected_columns):
    df.columns = expected_columns
else:
    print(" Warning: Number of columns does not match, check data!")

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

### Normalization, Standardization and bin the data
Age – Values range from 18 to 64, normalization reduces the impact of large differences.

BMI – Values range widely (15.96 → 53.13), so scaling helps balance feature importance.

Charges – Large range of medical costs ($1,121 → $63,770), making normalization essential.

```
df['Age_nor'] = df['Age']/ df['Age'].max()
df['BMI_nor'] = df['BMI']/ df['BMI'].max()
df['Charges_nor'] = df['Charges']/ df['Charges'].max()
```
```
# rounde value to nearest 2 decimal places
df[['Age_nor','BMI_nor', 'Charges', 'Charges_nor']] = np.round(df[['Age_nor','BMI_nor', 'Charges', 'Charges_nor']],2)
```
```
Age_bins =np.linspace(min(df['Age']), max(df['Age']),4)
Age_labels = ['Young', ' Middle-age', 'Senior']
df['Age_group'] = pd.cut(df['Age'], bins = Age_bins, labels=Age_labels, include_lowest = True)
BMI_bins =np.linspace(min(df['BMI']), max(df['BMI']),5)
BMI_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['BMI_group'] = pd.cut(df['BMI'], bins = BMI_bins, labels=BMI_labels, include_lowest = True)
Charges_bins =np.linspace(min(df['Charges']), max(df['Charges']),4)
Charges_labels = ['Low-cost', 'Medium-cost', 'High-cost']
df['Charges_group'] = pd.cut(df['Charges'], bins = Charges_bins, labels=Charges_labels, include_lowest = True)
```

```
print('Age_bins:', Age_bins)
print('BMI_bins:', BMI_bins)
print('Charges_bins:', Charges_bins)
df.head()
```
![image](https://github.com/user-attachments/assets/ce973757-b237-411d-b2d0-77f8892190c6)

### save data for analysis and visualization
```
df.to_csv('D:/Data anlysis- working sheet/python/data/insurance_cost_final.csv')
```

# DATA ANALYSIS

### Descriptive Statistics

Use visualizations to understand the distribution of key variables
```
num_cols = ['Age', 'BMI', 'Charges']
for col in num_cols:
    plt.figure(figsize=(6, 4)) 
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
```
![image](https://github.com/user-attachments/assets/0d0bb9c2-99f5-4fd0-8d20-e5a9b0aef753)
![image](https://github.com/user-attachments/assets/4c62afc4-6a6d-42f6-bf69-3541b6ea156b)
![image](https://github.com/user-attachments/assets/f3e83eee-bbbf-4783-aa0d-dd2d5c6884a1)

Age Distribution
Ages range from 18 to about 65 years old.
There is a large concentration at age 18, indicating many young customers in the dataset.
The distribution is relatively even for other ages, with no particular age dominating except for 18-year-olds.
Insight: There may be many students or young individuals in the insurance dataset.

BMI Distribution
The BMI follows a normal distribution, peaking around 27 - 30.
Most customers have a BMI between 20 - 40, meaning they mostly fall into the normal weight or overweight categories.
A small number of customers have a BMI above 45, which could indicate severe obesity.
Insight: Focusing on high BMI groups could help assess its impact on insurance costs.

Insurance Charges Distribution
The distribution is right-skewed, with most customers paying under 15,000 in insurance costs.
A small number of customers have very high costs (above 50,000 - 60,000), possibly due to health conditions like smoking or pre-existing diseases.
This indicates a huge disparity in insurance costs among different customer groups.
Insight: Further analysis is needed to examine factors like smoking (Smoker), BMI, or age that contribute to these high costs.

Summary:
Age: Many young customers (18 years old), while other ages are evenly distributed.
BMI: Most customers have a BMI between 20 - 40, following a normal distribution.
Insurance Costs: Significant disparity, with some customers paying extremely high fees.

### Check Correlations Between Variables
```
numeric_df = df.select_dtypes(include=['float64', 'int64'])
numeric_df.corr()
```
![image](https://github.com/user-attachments/assets/aa86edbb-e4ca-4777-a0d9-bd613287f2e0)

This table represents the correlation coefficients between different variables in the dataset. The correlation ranges from -1 to 1:

Close to 1 → Strong positive relationship (when one variable increases, the other also increases).
Close to -1 → Strong negative relationship (when one variable increases, the other decreases).
Close to 0 → No clear relationship.

Factors Affecting Insurance Charges (Charges)
Variables most correlated with Charges
Smoker (0.789141) → Smoking has a very strong impact on insurance costs.
Age (0.298892) → Older individuals tend to have higher insurance costs.
BMI (0.199906) → BMI has a positive correlation, but its impact is not as significant as smoking or age.
No_of_children (0.066551) → The number of children has very little effect on insurance costs.
Gender (0.062959) and Region (0.054018) → Almost no impact on insurance costs.

Conclusion:
Smoking is the most significant factor influencing insurance costs.
Age and BMI also affect the cost, but to a lesser extent.
Gender, region, and the number of children have minimal impact on insurance charges.

Relationships Between Other Variables
BMI and Region (0.271200) → There is a moderate correlation, possibly due to lifestyle or dietary differences across regions.
Age and BMI (0.112859) → Age does not significantly influence BMI.
Gender and Smoker (0.083125) → There is no major difference in smoking habits between genders.

### Comparison of Insurance Charges Between Smokers, Age and BMI group
```
num_cols = ['Age_group', 'BMI_group', 'Smoker']
for col in num_cols:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col], y=df['Charges'], palette='Set1')
    plt.show()

```
![image](https://github.com/user-attachments/assets/463dcab0-6e48-454a-b484-598a04443494)
![image](https://github.com/user-attachments/assets/3cc65adf-24c9-4161-afb1-06d23cc8f531)
![image](https://github.com/user-attachments/assets/021fef99-e3b1-4871-a6bc-bf83f1dd19a8)

Smoker vs. Charges
Smokers have significantly higher insurance charges compared to non-smokers.
The median insurance cost for smokers is much higher, with greater dispersion.
There are many outliers in the non-smoker group, but their values are still considerably lower than those of smokers.

Many outliers appear at the higher end of the cost distribution (above $20,000).
Possible reasons: Some non-smokers might have severe health conditions or other risk factors driving up costs.
However, even the highest non-smoker outliers are significantly lower than regular smoker costs.

BMI Group vs. Charges
The Overweight group has the highest average insurance costs and the largest spread.
The Obese group tends to have lower costs compared to the Overweight group, possibly due to data distribution or other influencing factors.
The remaining groups have relatively similar costs but still exhibit many outliers.

These groups show a moderate number of outliers, mostly in the $30,000+ range.
Potential reasons: Some individuals may have serious medical conditions despite a normal BMI, leading to higher costs.
Overweight:

This group has the widest spread of costs, including some extreme high-cost outliers above $60,000.
The higher variation suggests that overweight individuals have a more unpredictable range of health risks.

Obese:
Surprisingly, the outliers in this group are fewer than in the overweight category.
This could indicate that obese individuals may have more consistent health risks, leading to less variation in costs.

Age Group vs. Charges
The Senior group has higher insurance costs than the other groups.
The Young group has the lowest insurance costs but contains many outliers.
The Middle-age group has a higher spread compared to the Young group but lower than the Senior group.

young Group:

Many high-cost outliers, despite the overall lower median.
Possible explanation: Some younger individuals might have chronic conditions or require expensive treatments.
Middle-age Group:

Outliers appear but are slightly less extreme than in the younger group.
Insurance costs are increasing but still somewhat stable.
Senior Group:

Most seniors have high insurance costs, but outliers still exist, suggesting some individuals require extraordinary medical expenses beyond the usual age-related factors.
Key Insight:
Even though older individuals generally have higher costs, the young group has unexpected high-cost outliers, likely due to critical illnesses or accidents.


Summary:
Smoking has a strong impact on insurance costs, with significantly higher charges for smokers.
BMI also influences costs, particularly with the Overweight group incurring higher expenses.
Older age correlates with increased insurance costs, especially in the Senior group.

The limitation of the dataset is that it does not provide explicit health data for further investigation. Therefore, we temporarily focus on Smoking, BMI, and Age to analyze their impact on insurance charges.


###

### verify whether age, BMI, or other factors contribute to outliers in the non-smoker group
```
# Filter non-smokers
non_smoker_df = df[df["Smoker"] == 0]

# Calculate IQR
Q1 = non_smoker_df["Charges"].quantile(0.25)
Q3 = non_smoker_df["Charges"].quantile(0.75)
IQR = Q3 - Q1

# Define outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
outlier_threshold = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
outliers = non_smoker_df[(non_smoker_df["Charges"] < outlier_threshold[0]) | 
                          (non_smoker_df["Charges"] > outlier_threshold[1])]

print(outliers)
```
![image](https://github.com/user-attachments/assets/e565c993-1386-4582-aecd-1e1017d33da3)

### Scatter Plot of Age vs. Charges for Non-Smokers
```
plt.figure(figsize=(8,5))
sns.scatterplot(data=non_smoker_df, x="Age", y="Charges", color="blue", alpha=0.6, label="Non-Outliers")
sns.scatterplot(data=outliers, x="Age", y="Charges", color="red", label="Outliers")
plt.title("Age vs. Charges (Non-Smokers)")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/42457070-87dc-4d00-bce0-523d74ecd539)
The scatter plot shows that older individuals generally have higher insurance charges. However, some individuals of varying ages exhibit abnormally high costs, suggesting the influence of additional factors such as BMI or medical history.

### Scatter Plot of BMI vs. Charges for Non-Smokers
```
plt.figure(figsize=(8,5))
sns.scatterplot(data=non_smoker_df, x="BMI", y="Charges", color="blue", alpha=0.6, label="Non-Outliers")
sns.scatterplot(data=outliers, x="BMI", y="Charges", color="red", label="Outliers")
plt.title("BMI vs. Charges (Non-Smokers)")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/77bab224-03bf-4298-a987-ced9ae748e94)
The distribution of BMI and charges reveals that individuals with both normal and high BMI levels can have outlier charges.
This suggests that BMI alone is not the primary reason for these outliers

# MODEL DEVELOPMENT
### Fit a linear regression model that may be used to predict the charges value, just by using the smoker attribute of the dataset. Print the 
 score of this model.
 ```
# initialize a linear regression model,
lm = LinearRegression()
```
```
# create a linear function with 'smoker' as predictor variable and 'charges' as the response varialbe
x = df[['Smoker']] # predictor variable
y = df[['Charges']] # response variable
lm.fit(x, y) #Fit the linear model using highway-mpg
print(lm.score(x, y))
```
![image](https://github.com/user-attachments/assets/3490b055-0127-42f6-8232-d58195c7e9a0)
Result: R² = 0.6227
This means that the "Smoker" variable alone explains 62.27% of the variance in insurance costs. This is a relatively high explanatory power, indicating that smoking has a strong impact on insurance charges.

```
z = df[["Age", "Gender", "BMI", "No_of_children", "Smoker", "Region"]]
lm.fit(z,y)
print(lm.score(z, y))
```
![image](https://github.com/user-attachments/assets/b124df7c-e4a7-4f2f-81b3-1a1f370f4a58)
Result: R² = 0.7506
By adding more independent variables (Age, Gender, BMI, Number of Children, Smoker, Region), the model can now explain 75.06% of the variance in insurance charges.

This indicates that, apart from smoking, BMI and age also play significant roles in determining insurance costs.
However, the improvement from 62.27% → 75.06% suggests that smoking remains the most influential factor.

### Create a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() to create a model that can predict the charges value using all the other attributes of the dataset
```
# y and z use the same values as defined in previous cells 
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
z = z.astype(float)
pipe.fit(z,y)
ypipe=pipe.predict(z)
print(r2_score(y,ypipe))
```
![image](https://github.com/user-attachments/assets/193fd2c5-812f-45d6-b799-231fce9474de)
Result: R² = 0.8454
When applying Polynomial Regression, the explanatory power increases to 84.54%.
This suggests that the relationship between variables and insurance costs is not entirely linear, meaning that BMI or Age might have nonlinear effects on charges.
Data standardization (StandardScaler()) also helps improve accuracy.

### Scatter Plot vs Polynomial Regression Curve
```
columns = ['Age', 'BMI']
for col in columns:
    x = df[[col]]
    y = df[['Charges']]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Biến đổi thành đa thức
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x_scaled)

    # Fit model
    lm = LinearRegression()
    lm.fit(x_poly, y)

    # Dự báo
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    x_range_scaled = scaler.transform(x_range)
    x_range_poly = poly.transform(x_range_scaled)
    y_pred = lm.predict(x_range_poly)

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[col], y=df["Charges"], alpha=0.5, label="Actual Data")
    plt.plot(x_range, y_pred, color='red', label="Polynomial Regression")
    plt.xlabel(col)
    plt.ylabel('Charge')
    plt.legend()
    plt.show()
```
![image](https://github.com/user-attachments/assets/7e8ba459-5d22-4f4b-968b-b68377ae2492)
![image](https://github.com/user-attachments/assets/ccb32e13-5330-46bf-aea7-efb5eaafd137)

### Comparision Actual vs Predicted Charges
```
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y.values.ravel(), y=ypipe.ravel(), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Đường 45 độ
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges - Polynomial Regression")
plt.show()
```
![image](https://github.com/user-attachments/assets/86fce3cd-1f14-46b0-bc3f-b85ca243a852)

# Model Refinement
```
# Z and Y hold same values as in previous cells
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)

# x_train, x_test, y_train, y_test hold same values as in previous cells
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))

# x_train, x_test, y_train, y_test hold same values as in previous cells
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))
```
![image](https://github.com/user-attachments/assets/37369586-2f16-4b2d-bb0b-f6468fe63833)

Ridge Regression (R² = 0.725): A decent linear model, but not optimal.
Polynomial Ridge Regression (R² = 0.820): Adding polynomial features improved prediction accuracy.

# DATA VISUALIZATION
View the interactive Tableau dashboard here: [Insurance Cost Analysis Dashboard](https://public.tableau.com/views/Insurancecost_17420293456880/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

<div class='tableauPlaceholder' id='viz1742198987827' style='position: relative'><noscript><a href='#'><img alt='INSURANCE COST ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;Insurancecost_17420293456880&#47;Story1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Insurancecost_17420293456880&#47;Story1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;Insurancecost_17420293456880&#47;Story1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1742198987827');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>

