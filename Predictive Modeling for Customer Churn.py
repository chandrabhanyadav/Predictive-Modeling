import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
import os
file_path = os.path.join(os.getcwd(), 'customer_data.csv')  # Assuming the CSV is in the same directory
data = pd.read_csv(file_path)


# Convert 'TotalCharges' to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values in 'TotalCharges'
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Drop 'customerID'
data.drop('customerID', axis=1, inplace=True)

# Convert categorical variables into numerical using get_dummies
data = pd.get_dummies(data, drop_first=True)

# Split the dataset into features and target variable
X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
