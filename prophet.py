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