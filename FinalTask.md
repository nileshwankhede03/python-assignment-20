# AIML Practical Exam Codes

---

# Q2: Acquire, Visualize and Analyze `students_result.csv`

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("students_result.csv")

# Display First Rows
print(df.head())

# Dataset Info
print(df.info())
print(df.describe())

# Scatter Plot
plt.scatter(df['StudyHours'], df['ExamScore'])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# Bar Chart
df['StudyHours'].value_counts().plot(kind='bar')
plt.xlabel("Study Hours")
plt.ylabel("Count")
plt.title("Study Hours Count")
plt.show()

# Correlation
correlation = df['StudyHours'].corr(df['ExamScore'])
print("Correlation:", correlation)
```

---

# Q3: Feature Extraction using `synthetic_machining_data.csv`

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Dataset
df = pd.read_csv("synthetic_machining_data.csv")

# Display Rows
print(df.head())

# Missing Values
print(df.isnull().sum())

# Extract Features
X = df.select_dtypes(include=['int64','float64'])

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(X_pca)

# Visualization
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Visualization")
plt.show()
```

---

# Q4: Feature Selection using `cancer_dataset.csv`

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("cancer_dataset.csv")

# Display Rows
print(df.head())

# Correlation Matrix
corr = df.corr()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True)
plt.show()

# Select Important Features
selected_features = corr['target'].abs().sort_values(ascending=False)
print(selected_features)

# Split Dataset
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

# Q5: Classification Model using `ipl_data.csv`

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("ipl_data.csv")

# Display Rows
print(df.head())

# Features and Target
X = df.drop('result', axis=1)
y = df['result']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print(y_pred)

# Evaluation
print(classification_report(y_test, y_pred))
```

---

# Q6: Regression Model using `car_price.csv`

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv("car_price.csv")

# Display Rows
print(df.head())

# Features and Target
X = df[['Year']]
y = df['Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print(y_pred)

# Error Calculation
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# New Prediction
new_data = [[2024]]
prediction = model.predict(new_data)
print("Predicted Price:", prediction)
```

---

# Important Command

```python
print(df.columns)
```

Use this command if column name error occurs.

