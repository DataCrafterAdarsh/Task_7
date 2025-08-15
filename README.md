Task 7 – Support Vector Machines (SVM)

AI & ML Internship

*Objective
Implement Support Vector Machines for both linear and non-linear classification using the Breast Cancer dataset.
Understand margin maximization, kernel trick, and hyperparameter tuning.

*Dataset
Name: Breast Cancer Dataset
Rows: 569
Features: 30 numerical
Target: diagnosis (M=1, B=0)

*Steps Performed
Data Preparation
Dropped id column
Encoded target (M=1, B=0)
Standardized features

*Model Training
Linear SVM (kernel="linear")
RBF SVM (kernel="rbf")
Hyperparameter Tuning
Used GridSearchCV to find best C and gamma for RBF

*Visualization
Reduced features to 2D using PCA
Plotted decision boundaries for both kernels

*Evaluation
Accuracy, Precision, Recall, F1-score
5-fold cross-validation

*Tools Used
Python
Pandas, NumPy
Scikit-learn
Matplotlib

*How to Run
pip install pandas numpy scikit-learn matplotlib
python task7_svm.py
