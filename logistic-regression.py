import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

# Logistic Regression modeli
log_reg_model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
log_reg_model.fit(X_train_scaled, y_train)  # Model eğitimi
y_pred_log_reg = log_reg_model.predict(X_test_scaled)  # Test seti tahminleri

# Performans değerlendirme
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_f1 = f1_score(y_test, y_pred_log_reg, average="weighted")
log_reg_precision = precision_score(y_test, y_pred_log_reg, average="weighted")
log_reg_recall = recall_score(y_test, y_pred_log_reg, average="weighted")

print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {log_reg_accuracy:.4f}")
print(f"F1 Score: {log_reg_f1:.4f}")
print(f"Precision: {log_reg_precision:.4f}")
print(f"Recall: {log_reg_recall:.4f}")

# Lojistik Regresyon sınıflandırma raporu ve karmaşıklık matrisi
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_log_reg, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease (0)", "Disease (1)"], yticklabels=["No Disease (0)", "Disease (1)"])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()



# Logistic Regression doğruluk oranları
C_values = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
LRtrainAcc = []
LRtestAcc = []


for param in C_values:
    clf_lr = LogisticRegression(C=param, max_iter=1000)
    clf_lr.fit(X_train_scaled, y_train)
    LRtrainAcc.append(clf_lr.score(X_train_scaled, y_train))
    LRtestAcc.append(clf_lr.score(X_test_scaled, y_test))
    
    

# Logistic Regression için doğruluk oranlarını görselleştirme
plt.figure(figsize=(8, 6))  
plt.plot(C_values, LRtrainAcc, 'ro-', label="Train Accuracy (LR)")
plt.plot(C_values, LRtestAcc, 'bv--', label="Test Accuracy (LR)")
plt.legend()
plt.xlabel("C")
plt.xscale("log")  
plt.ylabel("Accuracy")
plt.title("Logistic Regression - C vs Accuracy")
plt.grid(True)  
plt.show()



