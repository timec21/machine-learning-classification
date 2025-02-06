from sklearn.neighbors import KNeighborsClassifier

# KNN modeli için GridSearch parametreleri
param_grid_knn = {
    'n_neighbors': range(1, 21),  
    'weights': ['uniform', 'distance'],  
    'metric': ['euclidean', 'manhattan', 'minkowski']  
}

# GridSearchCV ayarları
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, n_jobs=-1, verbose=2)

# GridSearch ile KNN modelini eğit
grid_search_knn.fit(X_train, y_train)


print("GridSearchCV ile bulunan en iyi parametreler (KNN): ", grid_search_knn.best_params_)

# En iyi modelle test setinde tahmin yap
best_knn = grid_search_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

# Performans metrikleri
print("Accuracy (KNN):", accuracy_score(y_test, y_pred_knn))
print("Precision Score (KNN):", precision_score(y_test, y_pred_knn))
print("Recall Score (KNN):", recall_score(y_test, y_pred_knn))
print("F1 Score (KNN):", f1_score(y_test, y_pred_knn))
print("Classification Report (KNN):\n", classification_report(y_test, y_pred_knn))

# Karmaşıklık matrisini oluşturma ve görselleştirme
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Recurrence (0)", "Recurrence (1)"], 
            yticklabels=["No Recurrence (0)", "Recurrence (1)"])
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Karmaşıklık Matrisi (KNN)")
plt.show()

print()
print()
print()
print()

# Farklı komşu sayıları (k) için doğruluk oranlarını saklamak
train_accuracies_knn = []
test_accuracies_knn = []
neighbor_range = range(1, 21)  

# Her komşu sayısı için KNN modeli çalıştırma
for k in neighbor_range:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X_train, y_train)  # Modeli eğitme

    # Eğitim ve test doğruluk oranlarını hesaplama
    train_accuracies_knn.append(clf_knn.score(X_train, y_train))
    test_accuracies_knn.append(clf_knn.score(X_test, y_test))

# Komşu sayısına göre doğruluk oranlarını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(neighbor_range, train_accuracies_knn, marker='o', label="Train Accuracy", linestyle='-', color='green')
plt.plot(neighbor_range, test_accuracies_knn, marker='s', label="Test Accuracy", linestyle='-', color='blue')
plt.xticks(neighbor_range)
plt.xlabel("Komşu Sayısı (k)")
plt.ylabel("Doğruluk Oranı")
plt.title("KNN'nin Komşu Sayısına Göre Doğruluk Oranları")
plt.legend()
plt.grid(True)
plt.show()
