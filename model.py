import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, r2_score

# Load the dataset
data = pd.read_csv("fetal_health.csv")  # Adjust path as needed

# Features and target
X = data.drop("fetal_health", axis=1)
y = data["fetal_health"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# # Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print(r2_score(y_true=y_test, y_pred=y_pred))
print("#"*60)


# Model 1: Naive Bayes
print("Naive Bayes Results:")
naive_model = GaussianNB()
naive_model.fit(X_train, y_train)
y_pred_nb = naive_model.predict(X_test)

# print(r2_score(y_test, y_pred_nb))
# print("#"*60)
# Evaluation for Naive Bayes
print("Confusion Matrix (Naive Bayes):\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))

# Model 2: Decision Tree
print("\nDecision Tree Results:")
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# print(r2_score(y_test, y_pred_tree))

# Evaluation for Decision Tree
print("Confusion Matrix (Decision Tree):\n", confusion_matrix(y_test, y_pred_tree))
print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred_tree))
