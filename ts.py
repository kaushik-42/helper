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
