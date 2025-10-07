import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('player_stats.csv')
df = df.drop_duplicates()

# Clean ACS first (for label only)
df['acs'] = pd.to_numeric(df['acs'], errors='coerce')
df = df.dropna(subset=['acs'])
print("Rows after ACS cleaning:", len(df))

for col in ['kill', 'death', 'assist', 'kast%', 'adr', 'hs%']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Create label
threshold = np.percentile(df['acs'], 80)
df['High_Performer'] = (df['acs'] >= threshold).astype(int)

# Features (exclude 'acs' since it's used for the label)
features = df[['kill', 'death', 'assist', 'kast%', 'adr', 'hs%']]
target = df['High_Performer']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
conf_mat = confusion_matrix(y_test, preds)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_mat)

y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('static/roc_curve.png')
