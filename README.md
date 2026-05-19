# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a dataset containing student details such as Hours Studied, Previous Score, Internship status, and Placement status.

2. Split the dataset into input features (X) and target output (y).

3. Divide the data into training and testing sets using train-test split.

4. Normalize the feature values using `StandardScaler` for better model performance.

5. Create and train the Logistic Regression model using the training data.

6. Make predictions on the test data and evaluate performance using accuracy, confusion matrix, and classification report.

7. Predict the placement status of a new student using the trained model.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Samantha Shree SV
RegisterNumber:  212225040362
*/
# Ex:No:5
# Implementation of Logistic Regression Model to Predict the Placement Status of Student

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],   # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]     # 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Step 5: Create and train Logistic Regression model
# ------------------------------
model = LogisticRegression()

model.fit(X_train_scaled, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = model.predict(X_test_scaled)

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict placement for a new student
# ------------------------------
new_student = pd.DataFrame({
    'Hours_Studied': [6],
    'Previous_Score': [68],
    'Internship': [1]
})

new_student_scaled = scaler.transform(new_student)

placement_pred = model.predict(new_student_scaled)
placement_prob = model.predict_proba(new_student_scaled)

print("\nPredicted Placement Status:",
      "Placed" if placement_pred[0] == 1 else "Not Placed")

print("Probability of Placement:",
      round(placement_prob[0][1], 2))

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

<img width="689" height="451" alt="image" src="https://github.com/user-attachments/assets/c7623167-b97e-4bc8-b666-4d988161f149" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
