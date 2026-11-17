
# Fraud Detection in Financial Transactions

## Project Overview
This project focuses on detecting fraudulent financial transactions using Machine Learning and Explainable Artificial Intelligence (XAI).

The system analyzes transaction data, performs preprocessing, trains multiple machine learning models, evaluates their performance, and identifies potentially fraudulent transactions.

## Objectives
- Analyze financial transaction data.
- Detect and handle missing values and duplicate records.
- Identify outliers using the IQR method.
- Handle imbalanced transaction data.
- Train and compare five machine learning models.
- Evaluate models using Accuracy, Precision, Recall and F1-Score.
- Explain predictions using SHAP and LIME.
- Develop an interactive fraud detection interface using Gradio.

## Technologies Used
- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- RandomOverSampler
- SHAP
- LIME
- Gradio

## Machine Learning Models
1. Random Forest
2. Logistic Regression
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Decision Tree

## Project Workflow
1. Dataset Collection
2. Data Loading and Exploration
3. Data Preprocessing
4. Feature Engineering
5. Outlier Detection using IQR
6. Duplicate Removal
7. Training and Testing Split
8. Handling Class Imbalance
9. Model Training
10. Model Evaluation and Comparison
11. Explainable AI using SHAP and LIME
12. Fraud Prediction Interface using Gradio

## Explainable AI
SHAP and LIME are used to understand the factors influencing the model's predictions and improve the interpretability of fraud detection results.

## Interactive User Interface
A Gradio-based interface allows users to enter transaction details and receive a fraud prediction from the trained Random Forest model.

## How to Run the Project
1. Open Google Colab.
2. Upload the project notebook.
3. Install the required Python libraries.
4. Provide the required transaction dataset (`train_transactions.csv`).
5. Execute the notebook cells sequentially.
6. Launch the Gradio interface.

## Dataset
The dataset is not included in this repository. Users must provide their own compatible transaction dataset.

## Project Notebook
`P_trw(1).ipynb`

## Disclaimer
This project is developed for academic and educational purposes. Predictions should not be treated as a substitute for professional financial fraud investigation.

## Author
Developed as an academic project on Fraud Detection in Financial Transactions.
