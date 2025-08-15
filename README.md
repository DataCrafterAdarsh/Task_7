# Task 7 – Support Vector Machines (SVM)

**AI & ML Internship – Binary Classification**

---

##  Objective
The goal of this task is to implement **Support Vector Machines (SVM)** for both **linear** and **non-linear** classification on the Breast Cancer dataset.  
We aim to understand:
- Margin maximization
- The kernel trick
- Hyperparameter tuning
- Model evaluation techniques

---

##  Dataset
- **Name:** Breast Cancer Dataset  
- **Rows:** 569  
- **Features:** 30 numerical  
- **Target:** `diagnosis` (`M` = 1 for malignant, `B` = 0 for benign)  
- **Source:** [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)  

---

##  Steps Performed

### 1. Data Preparation
- Dropped the `id` column  
- Encoded target values (`M` → 1, `B` → 0)  
- Split data into training and test sets (80-20 split)  
- Standardized all features using **StandardScaler**

### 2. Model Training
- **Linear SVM** (`kernel="linear"`) trained and evaluated
- **RBF SVM** (`kernel="rbf"`) trained and evaluated

### 3. Hyperparameter Tuning
- Used **GridSearchCV** to find the best values for:
  - `C` (regularization parameter)
  - `gamma` (spread parameter for RBF)
- Tuned model re-evaluated on the test set

### 4. Visualization
- Reduced data to 2 dimensions using **PCA**
- Plotted decision boundaries for both Linear and RBF SVM

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- 5-fold Cross-validation comparison for Linear and RBF kernels

---

##  Tools & Libraries Used
- **Python 3**
- **Pandas, NumPy** – data handling
- **Scikit-learn** – SVM, preprocessing, hyperparameter tuning, evaluation
- **Matplotlib** – visualization

---

##  How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
