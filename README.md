# Vehicle Emission Prediction and Vehicle Type Classification Using Machine Learning

## Overview

This project uses Machine Learning techniques to analyze vehicle data, predict carbon dioxide (CO₂) emissions, and classify vehicles based on their vehicle type.

The project uses the **2026 Test Car List dataset published by the U.S. Environmental Protection Agency (EPA)**. It applies data preprocessing, Exploratory Data Analysis (EDA), statistical analysis, regression algorithms, and classification algorithms to understand vehicle characteristics and build predictive models.

The project is divided into two main tasks:

1. **CO₂ Emission Prediction:** Predicts vehicle CO₂ emissions measured in grams per mile (g/mi) using regression models.
2. **Vehicle Type Classification:** Classifies vehicles into categories such as Car, Truck, and Both using classification models.

## Objectives

- Analyze vehicle specifications and emission-related data.
- Perform data cleaning and preprocessing.
- Identify patterns and relationships through EDA.
- Study correlations between vehicle characteristics and CO₂ emissions.
- Build and compare multiple regression models.
- Build and evaluate multiple classification models.
- Identify important features influencing model predictions.

## Dataset

**Dataset Name:** 2026 Test Car List Data  
**Source:** U.S. Environmental Protection Agency (EPA)  
**Format:** Excel (.xlsx)

**Official Dataset Download:**  
[Download the 2026 EPA Test Car List](https://www.epa.gov/system/files/documents/2026-01/26-testcar-2026-01-21.xlsx)

The dataset contains vehicle specifications and test-related information, including:

- Vehicle manufacturer and make
- Engine displacement
- Rated horsepower
- Number of cylinders
- Transmission type and number of gears
- Drive system
- Equivalent test weight
- Fuel type
- Road-load coefficients
- Vehicle type
- CO₂ emissions

**Target Variables**

| Task | Target Variable |
|---|---|
| Regression | `CO2 (g/mi)` |
| Classification | `Vehicle Type` |

The regression task uses records with available CO₂ target values. The classification task uses records with available vehicle-type labels.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- OpenPyXL

## Project Workflow

### 1. Data Understanding and Exploration
- Dataset dimensions and column inspection
- Data types and descriptive statistics
- Missing-value analysis
- Duplicate-record analysis
- Numerical and categorical feature identification

### 2. Data Preprocessing
- Removal of duplicate records
- Handling of missing values
- Text-value cleaning
- Numerical feature imputation using median values
- Categorical feature imputation using most frequent values
- One-hot encoding of categorical variables
- Feature scaling where required

### 3. Exploratory Data Analysis (EDA)
- CO₂ emission distribution
- Vehicle-type distribution
- Numerical feature distributions
- Outlier analysis using the IQR method
- Correlation heatmap
- Scatter plots for selected vehicle characteristics
- CO₂ emission comparison across vehicle types

### 4. Statistical Analysis
- Pearson correlation analysis
- Ordinary Least Squares (OLS) regression

### 5. Regression Models

The following algorithms are implemented:

1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The best regression model is selected based on the lowest test RMSE.

### 6. Classification Models

The following algorithms are implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Random Forest Classifier
6. Support Vector Machine (SVM)

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Log Loss
- Multiclass ROC-AUC (One-vs-Rest)

The best classification model is selected based on the highest test Macro F1-score.

## Model Results

### Classification Performance

The Random Forest Classifier achieved the following results in the reported test evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 97.95% |
| Macro Precision | 97.67% |
| Macro Recall | 97.99% |
| Macro F1-score | 97.83% |
| Macro ROC-AUC (OVR) | 99.91% |

The regression model comparison is generated when the Python script is executed. Refer to the generated regression comparison file for the model-wise performance.

*Note: These are results from the reported evaluation run. Performance may vary if the dataset, preprocessing, or model configuration changes.*

## Visualizations

The project generates visualizations to understand the dataset and evaluate model performance, including:

- CO₂ emission distribution
- Vehicle-type distribution
- Numerical feature histograms
- Feature boxplots
- Correlation heatmap
- CO₂ emission comparison by vehicle type
- Actual vs. predicted CO₂ values
- Regression residual plot
- Regression feature importance
- Classification confusion matrix
- Multiclass ROC curve
- Classification feature importance

## Project Structure

```text
Vehicle-Emission-Prediction/
│
├── CA2predictive.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── Vehicle_Project_Results/
    ├── 01_co2_distribution.png
    ├── 02_vehicle_type_distribution.png
    ├── 03_numerical_histograms.png
    ├── 04_correlation_heatmap.png
    ├── 05_co2_by_vehicle_type.png
    ├── 06_actual_vs_predicted.png
    ├── 07_residual_plot.png
    ├── 08_regression_feature_importance.png
    ├── 09_multiclass_roc_curve.png
    ├── 10_classification_feature_importance.png
    ├── regression_model_comparison.csv
    ├── classification_model_comparison.csv
    ├── regression_dataset.csv
    └── classification_dataset.csv
```

*The results folder is generated when the script runs. Actual filenames may vary depending on the output files generated.*

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/devanandanah/Vehicle-Emission-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Vehicle-Emission-Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the official EPA Excel dataset using the link provided above.

Place the downloaded file inside your project folder.

### 5. Update the Dataset Path

Open `CA2predictive.py` and locate the `pd.read_excel()` statement.

Update the file path to match your local dataset location. For example:

```python
df = pd.read_excel("26-testcar-2026-01-21.xlsx")
```

This assumes the dataset is in the same directory as the Python script.

### 6. Run the Project

You can execute the Python file using Spyder, Anaconda, or a terminal:

```bash
python CA2predictive.py
```

The script generates model evaluation results, datasets, CSV files, and visualizations in the `Vehicle_Project_Results` folder.

## Future Enhancements

- Hyperparameter tuning and cross-validation for improved model selection.
- Feature selection and leakage assessment.
- Deployment of the trained models through a web application.
- Interactive dashboards for vehicle emission analysis.
- Real-time prediction using user-provided vehicle specifications.

## Author

**Devanandana D**

B.Tech Computer Science and Engineering  
Lovely Professional University

## Acknowledgements

- U.S. Environmental Protection Agency (EPA) for providing the vehicle dataset.
- Scikit-learn and Python open-source communities for the machine learning tools and libraries used in this project.

---

**If you find this project useful, consider giving the repository a star!**
