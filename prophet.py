import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

# Step 1: Load and preprocess the data
df = pd.read_csv('your_dataset.csv', parse_dates=['Date'])

# Step 2: Separate the data by application
unique_applications = df['Tag_fo_application'].unique()

# Step 3: Perform Prophet modeling for each application
for application in unique_applications:
    # Filter data for the current application
    application_data = df[df['Tag_fo_application'] == application][['Date', 'direct_cost']]
    application_data = application_data.rename(columns={'Date': 'ds', 'direct_cost': 'y'})

    # Calculate the index for the train-test split
    split_index = int(len(application_data) * 0.7)

    # Split data into train and test sets
    train_data = application_data.iloc[:split_index]
    test_data = application_data.iloc[split_index:]

    # Fit the Prophet model
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(train_data)

    # Make predictions on the test data
    future = model.make_future_dataframe(periods=len(test_data), freq='15D')
    predictions = model.predict(future)

    # Select only the predictions for the test period
    test_predictions = predictions.iloc[split_index:]

    # Evaluate the model's performance
    rmse = mean_squared_error(test_data['y'], test_predictions['yhat'], squared=False)
    print(f"RMSE for {application}: {rmse:.2f}")

    
    import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, small_constant=1e-6):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) with handling for zero values.

    Parameters:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.
        small_constant (float, optional): Small constant value to be added to the denominator for handling zero values.
                                          Defaults to 1e-6.

    Returns:
        float: Mean Absolute Percentage Error (MAPE) with handling for zero values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    absolute_percentage_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), small_constant))
    mape = np.mean(absolute_percentage_errors) * 100

    return mape


def mape(y_true, y_pred):
    n = len(y_true)
    error_sum = 0
    count = 0

    for i in range(n):
        if y_true[i] != 0:
            error_sum += abs((y_true[i] - y_pred[i]) / y_true[i])
            count += 1

    if count > 0:
        mape_value = (error_sum / count) * 100
        return mape_value
    else:
        return None
