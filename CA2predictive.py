import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, label_binarize
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss,
    roc_auc_score, roc_curve, auc
)
# colour palette
COLORS = [
    "#C2185B",
    "#E63946",
    "#F97316",
    "#FFC857",
    "#8E245B",
    "#FF6B81"
]

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.1
)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Apply the palette
sns.set_palette(COLORS)
plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS)
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["savefig.facecolor"] = "#FFFFFF"
plt.rcParams["axes.edgecolor"] = "#D9D9D9"
plt.rcParams["text.color"] = "#333333"
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#444444"
plt.rcParams["ytick.color"] = "#444444"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Poppins", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 17
})
# Sequential map for heatmaps and confusion matrices
MAGENTA_ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "magenta_red_orange_yellow",
    ["#8E245B", "#C2185B", "#E63946", "#F97316", "#FFC857"]
)

# Diverging map for correlations: negative, neutral, positive.
CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "magenta_neutral_orange",
    ["#C2185B", "#FFF7ED", "#F97316"]
)


output_folder = "Vehicle_Project_Results"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_excel("C:/Users/Devanandana Dev/Downloads/26-testcar-2026-01-21.xlsx")

print("\nPROJECT: VEHICLE EMISSION PREDICTION AND VEHICLE TYPE CLASSIFICATION")
print("=" * 70)

print("\nFirst five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for number, column in enumerate(df.columns, start=1):
    print(number, column)

print("\nDataset information:")
df.info()

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumber of numerical columns:", len(numerical_columns))
print("Number of categorical columns:", len(categorical_columns))

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nMissing values:")
missing_values = df.isnull().sum()
missing_table = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": (missing_values / len(df)) * 100
})
missing_table = missing_table.sort_values(
    by="Missing Values", ascending=False
)
print(missing_table[missing_table["Missing Values"] > 0])

print("\nNumber of duplicate rows:", df.duplicated().sum())

print("\nNumerical descriptive statistics:")
print(df.describe().T)

print("\nCategorical descriptive statistics:")
print(df.describe(include="object").T)

# Save the initial dataset profile.
missing_table.to_csv(
    os.path.join(output_folder, "missing_value_analysis.csv")
)
df.describe(include="all").T.to_csv(
    os.path.join(output_folder, "dataset_descriptive_statistics.csv")
)

regression_target = "CO2 (g/mi)"
classification_target = "Vehicle Type"

required_columns = [regression_target, classification_target]
for column in required_columns:
    if column not in df.columns:
        raise ValueError("Required target column is missing: " + column)

print("\nCO2 target summary:")
print(df[regression_target].describe())
print("Missing CO2 values:", df[regression_target].isnull().sum())
print("Available CO2 values:", df[regression_target].notnull().sum())

print("\nVehicle Type distribution:")
print(df[classification_target].value_counts(dropna=False))
print("\nVehicle Type percentage:")
print(
    (df[classification_target].value_counts(normalize=True) * 100).round(2)
)

# Work on a copy so that the original data remains unchanged.
df = df.copy()

# Remove completely duplicated records, if any.
df = df.drop_duplicates().reset_index(drop=True)

# Remove rows where the classification target is missing.
# The inspected dataset has no missing Vehicle Type values.
df = df.dropna(subset=[classification_target])

# Clean whitespace in text columns.
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].astype("string").str.strip()
    df[column] = df[column].replace("", pd.NA)

# Check target labels.
print("\nVehicle Type labels after cleaning:")
print(df[classification_target].value_counts(dropna=False))
candidate_features = [
    "Vehicle Manufacturer Name",
    "Represented Test Veh Make",
    "Test Veh Displacement (L)",
    "Rated Horsepower",
    "# of Cylinders and Rotors",
    "Tested Transmission Type",
    "# of Gears",
    "Transmission Lockup?",
    "Drive System Description",
    "Equivalent Test Weight (lbs.)",
    "Axle Ratio",
    "N/V Ratio",
    "Shift Indicator Light Use Desc",
    "Test Fuel Type Description",
    "Target Coef A (lbf)",
    "Target Coef B (lbf/mph)",
    "Target Coef C (lbf/mph**2)",
    "Set Coef A (lbf)",
    "Set Coef B (lbf/mph)",
    "Set Coef C (lbf/mph**2)",
    "Police - Emergency Vehicle?"
]
# Use only columns that actually exist in the workbook.
feature_columns = [
    column for column in candidate_features if column in df.columns
]

# Constant columns cannot help a model distinguish between records.
constant_columns = [
    column for column in feature_columns
    if df[column].nunique(dropna=True) <= 1
]
feature_columns = [
    column for column in feature_columns
    if column not in constant_columns
]

print("\nSelected predictor columns:")
print(feature_columns)

print("\nExcluded constant predictor columns:")
print(constant_columns)

numeric_features = [
    column for column in feature_columns
    if pd.api.types.is_numeric_dtype(df[column])
]

outlier_rows = []

for column in numeric_features:
    values = df[column].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower_limit) | (values > upper_limit)
    ).sum()

    outlier_rows.append({
        "Feature": column,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": int(outlier_count)
    })
numeric_features = [
    column for column in feature_columns
    if pd.api.types.is_numeric_dtype(df[column])
]

outlier_rows = []

for column in numeric_features:
    values = df[column].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower_limit) | (values > upper_limit)
    ).sum()

    outlier_rows.append({
        "Feature": column,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": int(outlier_count)
    })
plt.figure(figsize=(9, 5))
plt.hist(
    df[regression_target].dropna(),
    bins=30,
    color=COLORS[0],
    edgecolor="white",
    linewidth=0.8
)
plt.title("Distribution of CO2 Emissions")
plt.xlabel("CO2 Emissions (g/mi)")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "01_co2_distribution.png"),
    dpi=300
)
plt.show()


plt.figure(figsize=(8, 5))
df[classification_target].value_counts().plot(kind="bar")
plt.title("Distribution of Vehicle Types")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "02_vehicle_type_distribution.png"),
    dpi=300
)
plt.show()
if numeric_features:
    df[numeric_features].hist(
        figsize=(15, 12),
        bins=25,
        color=COLORS[0],
        edgecolor="white",
        linewidth=0.6
    )
    plt.suptitle("Distribution of Numerical Vehicle Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "03_numerical_histograms.png"),
        dpi=300
    )
    plt.show()
for column in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title("Boxplot of " + column)
    plt.tight_layout()
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in column
    )
    plt.savefig(
        os.path.join(output_folder, "boxplot_" + safe_name + ".png"),
        dpi=300
    )
    plt.show()




correlation_columns = numeric_features.copy()

if regression_target in df.columns:
    correlation_columns.append(regression_target)

correlation_data = df[correlation_columns].corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_data,
    cmap=CORRELATION_CMAP,
    center=0,
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "04_correlation_heatmap.png"),
    dpi=300
)
plt.show()

print("\nCorrelation with CO2:")
if regression_target in correlation_data.columns:
    print(
        correlation_data[regression_target]
        .sort_values(ascending=False)
    )



scatter_features = [
    "Rated Horsepower",
    "Test Veh Displacement (L)",
    "Equivalent Test Weight (lbs.)"
]

for column in scatter_features:
    if column in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x=column,
            y=regression_target,
            color=COLORS[0],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.35
        )
        plt.title(column + " vs CO2 Emissions")
        plt.tight_layout()
        safe_name = "".join(
            character if character.isalnum() else "_"
            for character in column
        )
        plt.savefig(
            os.path.join(output_folder, "scatter_" + safe_name + ".png"),
            dpi=300
        )
        plt.show()



categorical_features = [
    column for column in feature_columns
    if column not in numeric_features
]

for column in categorical_features:
    counts = df[column].value_counts(dropna=False).head(12)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Top Categories: " + column)
    plt.xlabel(column)
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in column
    )
    plt.savefig(
        os.path.join(output_folder, "category_" + safe_name + ".png"),
        dpi=300
    )
    plt.show()


plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df.dropna(subset=[regression_target]),
    x=classification_target,
    y=regression_target
)
plt.title("CO2 Emissions by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("CO2 Emissions (g/mi)")
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "05_co2_by_vehicle_type.png"),
    dpi=300
)
plt.show()



# Pearson correlation is calculated for numerical variables.
pearson_correlation = df[correlation_columns].corr(method="pearson")

print("\nPearson correlation matrix:")
print(pearson_correlation)

pearson_correlation.to_csv(
    os.path.join(output_folder, "pearson_correlation.csv")
)

# OLS regression using selected numerical predictors.
# This is a separate statistical analysis, not the final ML pipeline.
ols_columns = [
    column for column in [
        "Rated Horsepower",
        "Test Veh Displacement (L)",
        "# of Cylinders and Rotors",
        "Equivalent Test Weight (lbs.)",
        "Axle Ratio",
        "N/V Ratio"
    ]
    if column in df.columns
]

ols_data = df[ols_columns + [regression_target]].dropna().copy()

if len(ols_data) > 20 and len(ols_columns) > 0:
    X_ols = sm.add_constant(ols_data[ols_columns])
    y_ols = ols_data[regression_target]

    ols_model = sm.OLS(y_ols, X_ols).fit()

    print("\nOLS Regression Summary:")
    print(ols_model.summary())

    with open(
        os.path.join(output_folder, "ols_regression_summary.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(ols_model.summary().as_text())
else:
    print("\nNot enough complete observations for the selected OLS analysis.")



def make_one_hot_encoder():
    # Supports both newer and older Scikit-learn versions.
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_preprocessor(features, scale_numeric=True):
    numeric = [
        column for column in features
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical = [
        column for column in features
        if column not in numeric
    ]

    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder())
    ])

    transformers = []

    if numeric:
        transformers.append(
            ("numerical", numeric_pipeline, numeric)
        )

    if categorical:
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )



regression_data = df.dropna(subset=[regression_target]).copy()

X_reg = regression_data[feature_columns].copy()
y_reg = regression_data[regression_target].copy()

# The split happens before fitting any imputer, encoder or scaler.
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

print("\nRegression training records:", len(X_train_reg))
print("Regression testing records:", len(X_test_reg))

regression_results = []
regression_predictions = {}
regression_models = {}



# Model 1: Simple Linear Regression
# One predictor is used to show the relationship between horsepower and CO2.

simple_feature = "Rated Horsepower"

if simple_feature not in feature_columns:
    simple_feature = numeric_features[0]

simple_preprocessor = make_preprocessor(
    [simple_feature],
    scale_numeric=False
)

simple_model = Pipeline([
    ("preprocessing", simple_preprocessor),
    ("model", LinearRegression())
])

simple_model.fit(
    X_train_reg[[simple_feature]],
    y_train_reg
)

simple_predictions = simple_model.predict(
    X_test_reg[[simple_feature]]
)

regression_models["Simple Linear Regression"] = simple_model
regression_predictions["Simple Linear Regression"] = simple_predictions


# Model 2: Multiple Linear Regression

multiple_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, False)),
    ("model", LinearRegression())
])

multiple_model.fit(X_train_reg, y_train_reg)
multiple_predictions = multiple_model.predict(X_test_reg)

regression_models["Multiple Linear Regression"] = multiple_model
regression_predictions["Multiple Linear Regression"] = multiple_predictions


# Model 3: Polynomial Regression
# Degree 2 allows the model to learn simple curved relationships.
# Polynomial expansion is applied to numerical features only.

poly_numeric_features = [
    column for column in numeric_features
    if column in feature_columns
]

poly_categorical_features = [
    column for column in categorical_features
    if column in feature_columns
]

poly_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

poly_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", make_one_hot_encoder())
])

poly_transformers = []

if poly_numeric_features:
    poly_transformers.append(
        ("numerical", poly_numeric_pipeline, poly_numeric_features)
    )

if poly_categorical_features:
    poly_transformers.append(
        ("categorical", poly_categorical_pipeline, poly_categorical_features)
    )

poly_preprocessor = ColumnTransformer(
    transformers=poly_transformers,
    remainder="drop"
)

polynomial_model = Pipeline([
    ("preprocessing", poly_preprocessor),
    ("model", LinearRegression())
])

polynomial_model.fit(X_train_reg, y_train_reg)
polynomial_predictions = polynomial_model.predict(X_test_reg)

regression_models["Polynomial Regression"] = polynomial_model
regression_predictions["Polynomial Regression"] = polynomial_predictions


# Model 4: Decision Tree Regressor

tree_regression_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, False)),
    ("model", DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ))
])

tree_regression_model.fit(X_train_reg, y_train_reg)
tree_regression_predictions = tree_regression_model.predict(X_test_reg)

regression_models["Decision Tree Regressor"] = tree_regression_model
regression_predictions["Decision Tree Regressor"] = tree_regression_predictions


# Model 5: Random Forest Regressor

random_forest_regression_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, False)),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_regression_model.fit(X_train_reg, y_train_reg)
random_forest_regression_predictions = (
    random_forest_regression_model.predict(X_test_reg)
)

regression_models["Random Forest Regressor"] = random_forest_regression_model
regression_predictions["Random Forest Regressor"] = random_forest_regression_predictions


# Model 6: Gradient Boosting Regressor

gradient_boosting_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, False)),
    ("model", GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

gradient_boosting_model.fit(X_train_reg, y_train_reg)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test_reg)

regression_models["Gradient Boosting Regressor"] = gradient_boosting_model
regression_predictions["Gradient Boosting Regressor"] = gradient_boosting_predictions




for model_name, predictions in regression_predictions.items():
    mae = mean_absolute_error(y_test_reg, predictions)
    mse = mean_squared_error(y_test_reg, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, predictions)

    regression_results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

regression_comparison = pd.DataFrame(regression_results)
regression_comparison = regression_comparison.sort_values(
    by="RMSE", ascending=True
)

print("\nREGRESSION MODEL COMPARISON")
print(regression_comparison.to_string(index=False))

regression_comparison.to_csv(
    os.path.join(output_folder, "regression_model_comparison.csv"),
    index=False
)

best_regression_name = regression_comparison.iloc[0]["Model"]
print("\nBest regression model based on test RMSE:", best_regression_name)

# Actual vs predicted plot for the best regression model.
best_regression_predictions = regression_predictions[best_regression_name]

plt.figure(figsize=(7, 6))
plt.scatter(
    y_test_reg,
    best_regression_predictions,
    color=COLORS[0],
    alpha=0.7,
    edgecolors="white",
    linewidths=0.4
)

minimum_value = min(
    y_test_reg.min(),
    best_regression_predictions.min()
)
maximum_value = max(
    y_test_reg.max(),
    best_regression_predictions.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--"
)
plt.xlabel("Actual CO2 (g/mi)")
plt.ylabel("Predicted CO2 (g/mi)")
plt.title("Actual vs Predicted CO2 - " + best_regression_name)
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "06_actual_vs_predicted.png"),
    dpi=300
)
plt.show()

# Residual plot for the best regression model.
residuals = y_test_reg - best_regression_predictions

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=best_regression_predictions,
    y=residuals,
    color=COLORS[1],
    alpha=0.7,
    edgecolor="white",
    linewidth=0.35
)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted CO2 (g/mi)")
plt.ylabel("Residuals")
plt.title("Residual Plot - " + best_regression_name)
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "07_residual_plot.png"),
    dpi=300
)
plt.show()

# Feature importance from the Random Forest regressor.
rf_regression_estimator = (
    random_forest_regression_model.named_steps["model"]
)
rf_regression_preprocessor = (
    random_forest_regression_model.named_steps["preprocessing"]
)

try:
    regression_feature_names = (
        rf_regression_preprocessor.get_feature_names_out()
    )
    regression_importance = pd.DataFrame({
        "Feature": regression_feature_names,
        "Importance": rf_regression_estimator.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    regression_importance.to_csv(
        os.path.join(output_folder, "regression_feature_importance.csv"),
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        regression_importance["Feature"][::-1],
        regression_importance["Importance"][::-1]
    )
    plt.title("Important Features - Random Forest Regressor")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "08_regression_feature_importance.png"),
        dpi=300
    )
    plt.show()
except Exception as error:
    print("\nCould not create regression feature importance:", error)


classification_data = df.dropna(subset=[classification_target]).copy()

X_class = classification_data[feature_columns].copy()
y_class = classification_data[classification_target].copy()

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

print("\nClassification training records:", len(X_train_class))
print("Classification testing records:", len(X_test_class))

classification_results = []
classification_models = {}
classification_predictions = {}
classification_probabilities = {}



# Model 1: Logistic Regression

logistic_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, True)),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Model 2: K-Nearest Neighbors

knn_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, True)),
    ("model", KNeighborsClassifier(n_neighbors=7))
])

# Model 3: Gaussian Naive Bayes

naive_bayes_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, True)),
    ("model", GaussianNB())
])

# Model 4: Decision Tree Classifier

decision_tree_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, False)),
    ("model", DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    ))
])

# Model 5: Random Forest Classifier

random_forest_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, False)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Model 6: Support Vector Machine

svm_model = Pipeline([
    ("preprocessing", make_preprocessor(feature_columns, True)),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

classification_model_list = {
    "Logistic Regression": logistic_model,
    "K-Nearest Neighbors": knn_model,
    "Gaussian Naive Bayes": naive_bayes_model,
    "Decision Tree Classifier": decision_tree_model,
    "Random Forest Classifier": random_forest_model,
    "Support Vector Machine": svm_model
}



class_labels = np.sort(y_class.unique())

for model_name, model in classification_model_list.items():

    print("\nTraining:", model_name)

    model.fit(X_train_class, y_train_class)

    predictions = model.predict(X_test_class)
    probabilities = model.predict_proba(X_test_class)

    classification_models[model_name] = model
    classification_predictions[model_name] = predictions
    classification_probabilities[model_name] = probabilities

    accuracy = accuracy_score(y_test_class, predictions)
    precision = precision_score(
        y_test_class, predictions,
        average="macro", zero_division=0
    )
    recall = recall_score(
        y_test_class, predictions,
        average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test_class, predictions,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test_class, predictions,
        average="weighted", zero_division=0
    )

    model_classes = model.named_steps["model"].classes_

    loss = log_loss(
        y_test_class,
        probabilities,
        labels=model_classes
    )

    # Multiclass ROC-AUC using one-vs-rest.
    try:
        auc_score = roc_auc_score(
            y_test_class,
            probabilities,
            labels=model_classes,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc_score = np.nan

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision (Macro)": precision,
        "Recall (Macro)": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Log Loss": loss,
        "ROC AUC (Macro OVR)": auc_score
    })

    print("\nClassification report:")
    print(
        classification_report(
            y_test_class,
            predictions,
            zero_division=0
        )
    )

    print("Log Loss:", round(loss, 4))
    print("Macro ROC-AUC:", auc_score)

    # Confusion matrix for each classifier.
    cm = confusion_matrix(
        y_test_class,
        predictions,
        labels=model_classes
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=MAGENTA_ORANGE_CMAP,
        xticklabels=model_classes,
        yticklabels=model_classes
    )
    plt.xlabel("Predicted Vehicle Type")
    plt.ylabel("Actual Vehicle Type")
    plt.title("Confusion Matrix - " + model_name)
    plt.tight_layout()

    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in model_name
    )
    plt.savefig(
        os.path.join(output_folder, "confusion_" + safe_name + ".png"),
        dpi=300
    )
    plt.show()



classification_comparison = pd.DataFrame(classification_results)

classification_comparison = classification_comparison.sort_values(
    by="Macro F1",
    ascending=False
)

print("\nCLASSIFICATION MODEL COMPARISON")
print(classification_comparison.to_string(index=False))

classification_comparison.to_csv(
    os.path.join(output_folder, "classification_model_comparison.csv"),
    index=False
)

best_classification_name = classification_comparison.iloc[0]["Model"]

print(
    "\nBest classification model based on Macro F1:",
    best_classification_name
)

# Save the best model's classification report.
best_classification_predictions = (
    classification_predictions[best_classification_name]
)

best_report = classification_report(
    y_test_class,
    best_classification_predictions,
    zero_division=0
)

with open(
    os.path.join(output_folder, "best_classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(best_report)



# Use the best classifier's probabilities and one-vs-rest curves.
best_classification_model = classification_models[best_classification_name]
best_probabilities = classification_probabilities[best_classification_name]
best_classes = best_classification_model.named_steps["model"].classes_

y_test_binary = label_binarize(
    y_test_class,
    classes=best_classes
)

plt.figure(figsize=(8, 6))

for index, class_name in enumerate(best_classes):
    if len(np.unique(y_test_binary[:, index])) < 2:
        continue

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test_binary[:, index],
        best_probabilities[:, index]
    )

    class_auc = auc(
        false_positive_rate,
        true_positive_rate
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=str(class_name) + " (AUC = " + str(round(class_auc, 3)) + ")"
    )

plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve - " + best_classification_name)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "09_multiclass_roc_curve.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------
# 18. CLASSIFICATION FEATURE IMPORTANCE
# ------------------------------------------------

rf_classifier = random_forest_model.named_steps["model"]
rf_classifier_preprocessor = random_forest_model.named_steps["preprocessing"]

try:
    classification_feature_names = (
        rf_classifier_preprocessor.get_feature_names_out()
    )

    classification_importance = pd.DataFrame({
        "Feature": classification_feature_names,
        "Importance": rf_classifier.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    classification_importance.to_csv(
        os.path.join(output_folder, "classification_feature_importance.csv"),
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        classification_importance["Feature"][::-1],
        classification_importance["Importance"][::-1]
    )
    plt.title("Important Features - Random Forest Classifier")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "10_classification_feature_importance.png"),
        dpi=300
    )
    plt.show()

except Exception as error:
    print("\nCould not create classification feature importance:", error)



regression_data.to_csv(
    os.path.join(output_folder, "regression_dataset.csv"),
    index=False
)

classification_data.to_csv(
    os.path.join(output_folder, "classification_dataset.csv"),
    index=False
)



print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED")
print("=" * 70)

print("\nOriginal dataset shape:", df.shape)
print("Regression dataset shape:", regression_data.shape)
print("Classification dataset shape:", classification_data.shape)

print("\nBest Regression Model:", best_regression_name)
print(
    regression_comparison.iloc[0][
        ["MAE", "MSE", "RMSE", "R2 Score"]
    ]
)

print("\nBest Classification Model:", best_classification_name)
print(
    classification_comparison.iloc[0][
        [
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "Macro F1",
            "Log Loss",
            "ROC AUC (Macro OVR)"
        ]
    ]
)

print("\nAll result files and graphs are saved in:")
print(os.path.abspath(output_folder))

print("\nReview the comparison tables and plots before writing conclusions.")
print("Do not report results that have not been checked and interpreted.")
