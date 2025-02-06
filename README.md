# Machine Learning Classification

## Overview
This project focuses on **machine learning classification** using various supervised learning algorithms to classify breast cancer data. The dataset is obtained from `sklearn.datasets.load_breast_cancer`, and multiple classification models are implemented, including:

- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

The models are trained and optimized using **GridSearchCV** to find the best hyperparameters. Performance metrics such as accuracy, precision, recall, F1-score, and confusion matrices are analyzed and visualized.

## Features
- **Data Loading & Preprocessing**
  - Loads the **breast cancer dataset** from `sklearn.datasets`.
  - Splits data into **training** and **test** sets.
  - Standardizes features using `StandardScaler` where needed.

- **Model Training & Hyperparameter Tuning**
  - Implements **Decision Tree**, **KNN**, **Logistic Regression**, **Random Forest**, and **SVM** models.
  - Uses **GridSearchCV** for hyperparameter tuning.
  
- **Model Evaluation & Performance Metrics**
  - Computes **accuracy, precision, recall, F1-score**.
  - Generates **classification reports**.
  - Plots **confusion matrices** for better insight into classification results.

- **Data Visualization**
  - Decision tree visualization using `plot_tree` and `pydotplus`.
  - **Heatmaps for confusion matrices**.
  - Accuracy comparisons for different model parameters.

## Technologies Used
- **Python**
- **Pandas** (data handling)
- **Scikit-Learn** (machine learning models, feature scaling, model evaluation)
- **Matplotlib & Seaborn** (data visualization)
- **pydotplus** (decision tree visualization)
- **IPython.display** (image rendering for decision tree graph)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/timec21/machine-learning-classification
   cd machine-learning-classification
   ```
## Visualizations
The script generates the following:
- **Decision Tree visualization** (text-based and graphical representation).
- **Confusion matrices** for different models.
- **Accuracy comparisons** for various hyperparameters.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


