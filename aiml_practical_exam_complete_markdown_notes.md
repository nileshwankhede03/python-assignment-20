# AIML & Data Science Practical Exam Notes

## Q2: Acquire, Visualize and Analyze the Dataset `students_result.csv`

---

# Objective

- Load dataset
- Visualize data
- Draw scatter plot
- Draw bar chart
- Find correlation

---

# Step 1: Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
```

## Explanation

### pandas

Dataset handle karne ke liye use hota hai.

### matplotlib

Graphs banane ke liye use hota hai.

---

# Step 2: Load Dataset

```python
df = pd.read_csv("students_result.csv")
print(df.head())
```

## Explanation

- `read_csv()` → CSV file load karta hai
- `head()` → first 5 rows show karta hai

---

# Step 3: Dataset Information

```python
print(df.info())
print(df.describe())
```

## Explanation

### info()

- Datatypes
- Null values
- Total columns

### describe()

- Mean
- Max
- Min
- Standard deviation

---

# Step 4: Scatter Plot

```python
plt.scatter(df['StudyHours'], df['ExamScore'])

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")

plt.show()
```

## Marathi Explanation

Scatter plot 2 variables cha relation dakhavto.

- X-axis → Study Hours
- Y-axis → Exam Score

Study hours vadhle ki marks vadhatat ka te samajte.

---

# Step 5: Bar Chart

```python
df['StudyHours'].value_counts().plot(kind='bar')

plt.xlabel("Study Hours")
plt.ylabel("Count")
plt.title("Study Hours Count")

plt.show()
```

## Marathi Explanation

Pratyek study hour kiti vela ala te show karto.

---

# Step 6: Correlation

```python
correlation = df['StudyHours'].corr(df['ExamScore'])

print("Correlation:", correlation)
```

## Marathi Explanation

Correlation 2 variables madhla relation strong aahe ka te sangto.

Values:

- +1 → Positive relation
- 0 → No relation
- -1 → Negative relation

---

# Viva Questions

## What is correlation?

2 variables madhla relation measure karto.

## Why scatter plot?

2 continuous variables cha relation baghnyasathi.

## Why bar chart?

Frequency show karnyasathi.

# Q3: Feature Extraction using `synthetic_machining_data.csv`

---

# Objective

- Display first rows
- Check missing values
- Extract features
- Apply PCA
- Visualization

---

# Step 1: Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

---

# Step 2: Load Dataset

```python
df = pd.read_csv("synthetic_machining_data.csv")

print(df.head())
```

---

# Step 3: Check Missing Values

```python
print(df.isnull().sum())
```

## Marathi Explanation

Konatya column madhye missing values aahet te check karto.

---

# Step 4: Extract Features

```python
X = df.select_dtypes(include=['int64','float64'])

print(X.head())
```

## Marathi Explanation

Fakta numerical columns select karto.

---

# Step 5: Standardization

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

## Marathi Explanation

Saglya values same scale var anto.

---

# Step 6: Apply PCA

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(X_pca)
```

## Marathi Explanation

PCA features kami karto pan important information thevto.

Example:

- 10 columns → 2 columns

---

# Step 7: Visualization

```python
plt.scatter(X_pca[:,0], X_pca[:,1])

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Visualization")

plt.show()
```

---

# Viva Questions

## What is PCA?

Dimensionality reduction technique.

## Why PCA?

Features kami karun speed ani performance improve karto.

# Q4: Feature Selection using `cancer_dataset.csv`

---

# Objective

- Split dataset
- Plot heatmap
- Select important features

---

# Step 1: Import Libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
```

---

# Step 2: Load Dataset

```python
df = pd.read_csv("cancer_dataset.csv")

print(df.head())
```

---

# Step 3: Correlation Heatmap

```python
corr = df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr, annot=True)

plt.show()
```

## Marathi Explanation

Heatmap color vaprun relation dakhavto.

Dark color = Strong relation.

---

# Step 4: Select Important Features

```python
selected_features = corr['target'].abs().sort_values(ascending=False)

print(selected_features)
```

## Marathi Explanation

Target sobat jasta relation aslele features select karto.

---

# Step 5: Train Test Split

```python
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## Marathi Explanation

Dataset divide karto:

- 80% → Training
- 20% → Testing

---

# Viva Questions

## What is feature selection?

Important features select karne.

## Why heatmap?

Correlation visualize karnyasathi.

# Q5: Classification Model using `ipl_data.csv`

---

# Objective

- Train classification model
- Make predictions
- Evaluate performance

---

# Step 1: Import Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
```

---

# Step 2: Load Dataset

```python
df = pd.read_csv("ipl_data.csv")

print(df.head())
```

---

# Step 3: Features and Target

```python
X = df.drop('result', axis=1)
y = df['result']
```

## Marathi Explanation

- X → Input features
- y → Output/Target

---

# Step 4: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

# Step 5: Train Model

```python
model = DecisionTreeClassifier()

model.fit(X_train, y_train)
```

## Marathi Explanation

Model training data varun shikto.

---

# Step 6: Prediction

```python
y_pred = model.predict(X_test)

print(y_pred)
```

---

# Step 7: Evaluation

```python
print(classification_report(y_test, y_pred))
```

---

# Viva Questions

## What is classification?

Category predict karne.

Example:

- Win/Lose
- Pass/Fail

## What is Decision Tree?

Tree structure vaprun prediction karanara algorithm.

# Q6: Regression Model using `car_price.csv`

---

# Objective

- Train regression model
- Predict prices
- Evaluate model
- Predict new data

---

# Step 1: Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

---

# Step 2: Load Dataset

```python
df = pd.read_csv("car_price.csv")

print(df.head())
```

---

# Step 3: Features and Target

```python
X = df[['Year']]

y = df['Price']
```

---

# Step 4: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

# Step 5: Train Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

---

# Step 6: Prediction

```python
y_pred = model.predict(X_test)

print(y_pred)
```

---

# Step 7: Calculate Error

```python
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
```

## Marathi Explanation

MSE prediction kiti wrong aahe te sangto.

Small MSE → Model changla aahe.

---

# Step 8: Visualization

```python
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.show()
```

---

# Step 9: Predict New Data

```python
new_data = [[2024]]

prediction = model.predict(new_data)

print("Predicted Price:", prediction)
```

---

# Viva Questions

## What is regression?

Numeric value predict karne.

Example:

- Salary
- Price
- Temperature

## Difference Between Classification and Regression

| Classification   | Regression            |
| ---------------- | --------------------- |
| Category predict | Numeric value predict |
| Pass/Fail        | Price/Salary          |

---

# Important Practical Tips

## Check Columns

```python
print(df.columns)
```

Dataset madhye exact column names baghnyasathi.

---

# Common Libraries

| Library    | Use                    |
| ---------- | ---------------------- |
| pandas     | Data handling          |
| matplotlib | Graph plotting         |
| seaborn    | Advanced visualization |
| sklearn    | Machine Learning       |

---

# Final Viva Introduction

"Today I performed data preprocessing, visualization, feature extraction, feature selection, classification and regression using Python libraries like Pandas, Matplotlib, Seaborn and Scikit-learn."
