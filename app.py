import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the Diabetes dataset
@st.cache_data
def load_and_preprocess_data():
    diabetes_data = load_diabetes()
    X = diabetes_data.data  # Features
    y = diabetes_data.target  # Target (disease progression)
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Evaluate the model performance
def evaluate_model(y_test, y_pred, threshold=5):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    
    return {'MSE': mse, 'RMSE': rmse, 'R²': r2,}

# Streamlit app starts here
st.title("Diabetes Progression Prediction App")
st.write("""
    This application uses Lasso Regression to predict disease progression for diabetes patients.
""")

# Load the data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train the Lasso Regression model with hyperparameter tuning
lasso_model = Lasso()
lasso_param_grid = {'alpha': np.logspace(-4, 4, 100)}  # Alpha values for regularization
lasso_grid_search = GridSearchCV(lasso_model, lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid_search.fit(X_train, y_train)

# Get the best model
best_lasso_model = lasso_grid_search.best_estimator_
y_pred = best_lasso_model.predict(X_test)

# Evaluate the model
metrics = evaluate_model(y_test, y_pred)

# Display results in Streamlit
st.subheader("Model Evaluation Metrics")
st.write(f"MSE: {metrics['MSE']}")
st.write(f"RMSE: {metrics['RMSE']}")
st.write(f"R² Score: {metrics['R²']}")

# Display actual vs predicted results
st.subheader("Actual vs Predicted Values")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(results_df.head(10))  # Display the first 10 predictions

# Allow user to input values and make predictions
st.subheader("Make a Prediction")

# Define valid input ranges based on the dataset features
feature_ranges = {
    'age': (0.0, 100.0),  # Age cannot be negative
    'sex': (0.0, 1.0),    # 0 for female, 1 for male
    'bmi': (10.0, 40.0),  # Assuming typical BMI range
    'bp': (0.0, 200.0),    # Blood pressure
    's1': (0.0, 100.0),    # Total cholesterol
    's2': (0.0, 100.0),    # LDL cholesterol
    's3': (0.0, 100.0),    # HDL cholesterol
    's4': (0.0, 200.0),    # Triglycerides
    's5': (0.0, 100.0),    # Family history
    's6': (0.0, 100.0),    # Other health factor
}

input_values = []
for feature in load_diabetes().feature_names:
    min_val, max_val = feature_ranges.get(feature, (-10.0, 10.0))  # Default range if not specified
    input_val = st.number_input(f"Enter value for {feature}", min_value=float(min_val), max_value=float(max_val), value=float(min_val), step=0.1)  # Ensure step is a float
    input_values.append(input_val)

if st.button("Predict"):
    # Scale the input values the same way we scaled training data
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform([input_values])
    prediction = best_lasso_model.predict(scaled_values)
    st.write(f"Predicted Disease Progression: {prediction[0]:.2f}")
