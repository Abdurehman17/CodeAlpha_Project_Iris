import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Convert to a pandas DataFrame for easier data handling
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
print(iris_df.head())  # Check the first few rows of the data
print(iris_df.describe())  # Summary statistics
print(iris_df['species'].value_counts())  # Count of each species
# Split the data into features (X) and labels (y)
X = iris_df.drop(columns='species')
y = iris_df['species']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the KNN classifier with k=3 (you can experiment with different values)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)
# Make predictions on the test data
y_pred = knn.predict(X_test)

# Print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
import seaborn as sns

sns.pairplot(iris_df, hue='species')
plt.show()
