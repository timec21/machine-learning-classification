import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Veri setini yükleme
data = load_breast_cancer()

# Veriyi DataFrame'e çevirme
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Hedef değişken ve özellikleri belirleme
X = df.drop(['target'], axis=1)
y = df['target']

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Modeli
num_base_classifiers = 500  
clf_rf = RandomForestClassifier(n_estimators=num_base_classifiers, max_depth=10, random_state=42)
clf_rf.fit(X_train_scaled, y_train)  
y_pred_train_rf = clf_rf.predict(X_train_scaled)  
y_pred_test_rf = clf_rf.predict(X_test_scaled)  

# Performans değerlendirme
rf_accuracy = accuracy_score(y_test, y_pred_test_rf)
rf_f1 = f1_score(y_test, y_pred_test_rf, average="weighted")
rf_precision = precision_score(y_test, y_pred_test_rf, average="weighted")
rf_recall = recall_score(y_test, y_pred_test_rf, average="weighted")

print("Random Forest Model Evaluation:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"F1 Score: {rf_f1:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}\n")

# Random Forest Classification raporu
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_test_rf))

# Random Forest Karmaşıklık Matrisi
cm_rf = confusion_matrix(y_test, y_pred_test_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Disease (0)", "Disease (1)"], 
            yticklabels=["No Disease (0)", "Disease (1)"])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
