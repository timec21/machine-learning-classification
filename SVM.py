import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# SVM Modeli
svm_model = SVC(kernel="linear", C=0.1, random_state=42)
svm_model.fit(X_train_scaled, y_train)  # Model eğitimi
y_pred_svm = svm_model.predict(X_test_scaled)  # Test seti tahminleri

# SVM Modeli Değerlendirme
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average="weighted")
svm_precision = precision_score(y_test, y_pred_svm, average="weighted")
svm_recall = recall_score(y_test, y_pred_svm, average="weighted")

print("SVM Model Evaluation:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"F1 Score: {svm_f1:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}\n")

# SVM Modeli Classification raporu
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# SVM Karmaşıklık Matrisi
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease (0)", "Disease (1)"], yticklabels=["No Disease (0)", "Disease (1)"])
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
