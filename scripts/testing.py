import joblib
import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load('D:/ml project/models/aviation_severity_model.pkl')
df = pd.read_csv("D:/ml project/data/cleaned/final_aviation_data.csv")
df.drop(columns = ["investigation_type"])
print("model loaded sucessfully!")

df = pd.read_csv("D:/ml project/data/cleaned/final_aviation_data.csv")
df.drop(columns = ["unamed: 0.1","investigation_table"],errors = "ignore")
print("data loaded succesfully! Rows: ", len(df))

test_rows = int(input("eneter rows you want to sample for testing:"))
print(f"sampling {test_rows} rows for testing..")
sample = df.sample(test_rows, random_state = 42)
true_labels = sample["target"]
sample_input = sample.drop(columns = ["target"])

print("\nSAMPLE INPUT")
print(sample_input)
preds= model.predict(sample_input)
probs = model.predict_proba(sample_input)[:,1]

print("\nPredictions:")
print(preds)

print("\nProbability(Fatal Risk Percentage): ")
print(probs)

print("\nTrue Labels: ")
print(true_labels.values)

from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\n MODEL TESTING COMPLETE!")