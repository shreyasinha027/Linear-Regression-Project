# Linear Regression Project

## Objective

Implement and understand Simple and Multiple Linear Regression using Python and Scikit-Learn.

## Dataset

The dataset used is **Uncleaned_DS_jobs.csv**, which contains information related to Data Science job listings.

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

## Steps Performed

### 1. Data Loading

* Imported the dataset using Pandas.
* Inspected the structure of the dataset.

### 2. Data Preprocessing

* Removed duplicate records.
* Selected numerical features.
* Handled missing values using median imputation.

### 3. Train-Test Split

* Split the dataset into training and testing sets using an 80:20 ratio.

### 4. Model Building

* Trained a Linear Regression model using `sklearn.linear_model.LinearRegression`.

### 5. Model Evaluation

Evaluated model performance using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

### 6. Visualization

Generated:

* Actual vs Predicted Values Plot
* Residual Plot

### 7. Coefficient Interpretation

Analyzed feature coefficients to understand the impact of each feature on the target variable.

## Results

The Linear Regression model was successfully trained and evaluated on the dataset. The performance metrics and visualizations provide insights into the model's predictive capability.

## Project Structure

linear-regression/

├── main.py

├── Uncleaned_DS_jobs.csv

├── README.md

└── requirements.txt


## Author

Shreya Sinha
