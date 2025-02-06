import pydotplus
from IPython.display import Image
from sklearn.tree import plot_tree, export_graphviz
import json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Karar ağacı modeli 
clf_dt = DecisionTreeClassifier(random_state=42)

# GridSearch parametreleri belirleme
param_grid = {
    'criterion': ['gini', 'entropy'],               
    'max_depth': [None, 10, 20, 30, 40],             
    'min_samples_split': [2, 5, 10],                 
    'min_samples_leaf': [1, 2, 4],                   
    'max_features': [None, 'sqrt', 'log2'],          
    'splitter': ['best', 'random']                   
}

# GridSearchCV ayarları
grid_search = GridSearchCV(estimator=clf_dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# GridSearch ile modeli eğitme
grid_search.fit(X_train, y_train)


print("GridSearchCV ile bulunan en iyi parametreler: ", grid_search.best_params_)

# En iyi modelle test setinde tahmin yapma
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

# Performans metrikleri
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Karmaşıklık matrisini oluşturma ve görselleştirme
# Hastalığı tekrar edip etmeyeceğine göre sınıflandırma
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Recurrence (0)", "Recurrence (1)"], 
            yticklabels=["No Recurrence (0)", "Recurrence (1)"])
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Karmaşıklık Matrisi (Decision Tree)")
plt.show()

# Matplotlib ile ağaç görselleştirme 
plt.figure(figsize=(30, 20))  
plot_tree(best_dt, feature_names=X.columns.tolist(), 
          class_names=['No Recurrence', 'Recurrence'], 
          filled=True, fontsize=12)
plt.title("Decision Tree Visualization (Detailed and Expanded)", fontsize=16)
plt.show()

# pydotplus ve Image ile karar ağacını görselleştirme
dot_data = export_graphviz(best_dt, out_file=None, feature_names=X.columns, 
                           class_names=['No Recurrence', 'Recurrence'], 
                           filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


plt.figure(figsize=(15, 10))
plot_tree(best_dt, feature_names=X.columns.tolist(), 
          class_names=['No Recurrence', 'Recurrence'], 
          filled=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# Farklı derinlikler için doğruluk oranlarını saklamak
train_accuracies = []
test_accuracies = []
max_depths = range(1, 21)  

# Her derinlik için karar ağacı modeli çalıştırma
for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)  

    # Eğitim ve test doğruluk oranlarını hesaplama
    train_accuracies.append(clf.score(X_train, y_train))
    test_accuracies.append(clf.score(X_test, y_test))

# Derinliğe göre doğruluk oranlarını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracies, marker='o', label="Train Accuracy", linestyle='-', color='green')
plt.plot(max_depths, test_accuracies, marker='s', label="Test Accuracy", linestyle='-', color='blue')
plt.xticks(max_depths)
plt.xlabel("Maksimum Derinlik")
plt.ylabel("Doğruluk Oranı")
plt.title("Karar Ağacının Maksimum Derinliğine Göre Doğruluk Oranları")
plt.legend()
plt.grid(True)
plt.show()


