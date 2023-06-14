import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load and preprocess the data
df = pd.read_csv('your_dataset.csv', parse_dates=['Date'], index_col='Date')
# Perform any necessary preprocessing steps, handle missing values or outliers

# Step 2: Separate the data by application
unique_applications = df['Tag_fo_application'].unique()

# Step 3: Perform ARIMA modeling for each application
for application in unique_applications:
    # Filter data for the current application
    application_data = df[df['Tag_fo_application'] == application]['direct_cost']

    # Apply Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)  # Number of splits, you can adjust this value

    for train_index, test_index in tscv.split(application_data):
        train_data = application_data.iloc[train_index]
        test_data = application_data.iloc[test_index]

        # Fit the ARIMA model
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()

        # Make predictions on the test data
        predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

        # Evaluate the model's performance
        # Compare predictions with actual values and calculate evaluation metrics
   
# FOR XGBOOST:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Step 1: Load and preprocess the data
df = pd.read_csv('your_dataset.csv', parse_dates=['Date'], index_col='Date')
# Perform any necessary preprocessing steps, handle missing values or outliers

# Step 2: Separate the data by application
unique_applications = df['Tag_fo_application'].unique()

# Step 3: Perform XGBoost modeling for each application
for application in unique_applications:
    # Filter data for the current application
    application_data = df[df['Tag_fo_application'] == application]['direct_cost']

    # Perform manual train-test split
    train_data, test_data = train_test_split(application_data, test_size=0.2)

    # Check if train_data or test_data has insufficient data points
    if len(train_data) < 2 or len(test_data) < 1:
        print(f"Not enough data for {application}. Skipping...")
        continue

    try:
        # Prepare the data
        X_train = np.array(range(len(train_data))).reshape(-1, 1)
        y_train = train_data.values

        X_test = np.array(range(len(train_data), len(train_data) + len(test_data))).reshape(-1, 1)
        y_test = test_data.values

        # Train the XGBoost model
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = model.predict(X_test)

        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.plot(application_data.index, y, label='Actual')
        plt.plot(application_data.index, predictions, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Direct Costs')
        plt.title(f'{application} - Actual vs Predicted')
        plt.legend()
        plt.show()
        
        # Calculate RMSE
        rmse = mean_squared_error(y_test, predictions, squared=False)
        print(f"RMSE for {application}: {rmse:.2f}")

    except Exception as e:
        print(f"Error occurred for {application}: {str(e)}")

# To plot things for Xgboost(Seperately):
# Plot the actual data and predictions
plt.figure(figsize=(10, 6))
plt.plot(application_data.index, y, label='Actual')
plt.plot(application_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Direct Costs')
plt.title(f'{selected_application} - Actual vs Predictions')
plt.legend()
plt.show()
