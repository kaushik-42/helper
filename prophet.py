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

#############
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Step 1: Load and preprocess the data
df = pd.read_csv('your_dataset.csv', parse_dates=['Date'])

# Step 2: Separate the data by application
unique_applications = df['Tag_fo_application'].unique()

# Step 3: Define the parameter grid for tuning
seasonality_prior_scales = [0.01, 0.1, 1, 10]
changepoint_prior_scales = [0.001, 0.01, 0.1, 1]

best_models = {}
best_params = {}

# Step 4: Iterate over applications and fit models
for application in unique_applications:
    # Filter data for the current application
    application_data = df[df['Tag_fo_application'] == application][['Date', 'direct_cost']]
    application_data = application_data.rename(columns={'Date': 'ds', 'direct_cost': 'y'})

    train_size = int(len(application_data) * 0.8)
    train_data = application_data[:train_size]
    valid_data = application_data[train_size:]

    best_mape = float('inf')
    best_model = None

    for seasonality_prior_scale in seasonality_prior_scales:
        for changepoint_prior_scale in changepoint_prior_scales:
            # Fit the Prophet model
            model = Prophet(seasonality_prior_scale=seasonality_prior_scale,
                            changepoint_prior_scale=changepoint_prior_scale,
                            weekly_seasonality=True)
            model.fit(train_data)

            # Make predictions on the validation set
            forecast = model.predict(valid_data[['ds']])
            predictions = forecast['yhat'].values

            # Calculate the MAPE score
            mape = mean_absolute_percentage_error(valid_data['y'], predictions)

            # Check if this model has the best MAPE score so far
            if mape < best_mape:
                best_mape = mape
                best_model = model
                best_params[application] = {
                    'seasonality_prior_scale': seasonality_prior_scale,
                    'changepoint_prior_scale': changepoint_prior_scale
                }

    # Save the best model for this application
    best_models[application] = best_model

# Step 5: Print the best models and their parameters
for application, model in best_models.items():
    print(f"Application: {application}")
    print("Best Model Parameters:")
    print(best_params[application])
    print()

# Step 6: Example of using the best models for forecasting
for application, model in best_models.items():
    future = model.make_future_dataframe(periods=365)  # Adjust the number of periods as needed
    forecast = model.predict(future)
    # Perform further operations or analysis on the forecast data
    # ...

# Additional steps as per your requirement


import streamlit as st
from prophet import Prophet

# Define the available application options and frequencies
application_options = ['App 1', 'App 2', 'App 3']
frequency_options = ['D', 'W', 'M']

# Function to train the Prophet model and generate the forecast
def train_prophet_model(application, period, frequency):
    # Perform the necessary data preprocessing and Prophet model training here
    # Replace this with your actual implementation
    # ...

    # Return the forecast results
    return forecast


# Streamlit app
def main():
    # Page title
    st.title("Prophet Forecasting App")

    # User inputs
    application = st.selectbox("Select Application", application_options)
    period = st.text_input("Enter Period")
    frequency = st.selectbox("Select Frequency", frequency_options)

    # Submit button
    if st.button("Submit"):
        if application and period and frequency:  # Check if all fields are selected
            # Call the function to train the Prophet model and generate the forecast
            forecast = train_prophet_model(application, period, frequency)

            # Display the forecast results
            st.write("Forecast Results:")
            st.write(forecast)
        else:
            st.warning("Please select all fields.")

# Run the Streamlit app
if __name__ == "__main__":
    main()


import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet

# Assuming you have already trained your Prophet model and obtained the forecast
# and the model instance is stored in 'model'

# Visualize forecast plot with Plotly
fig_forecast = model.plot(forecast)
fig_forecast.update_layout(
    title="Forecast Plot",
    xaxis_title="Date",
    yaxis_title="Value",
)

# Visualize components plot with Plotly
fig_components = model.plot_components(forecast)
fig_components.update_layout(
    title="Components Plot",
)

# Render the interactive Plotly charts using Streamlit
st.plotly_chart(fig_forecast)
st.plotly_chart(fig_components)

import streamlit as st
import plotly.figure_factory as ff
from prophet import Prophet

# Assuming you have already trained your Prophet model and obtained the forecast
# and the model instance is stored in 'model'

# Visualize forecast plot with Plotly
fig_forecast = model.plot(forecast)
fig_forecast = ff.create_2d_density(
    fig_forecast.data[0].x,
    fig_forecast.data[0].y,
    colorscale='Viridis',
    hist_color='rgba(0,100,200,0.7)'
)

# Visualize components plot with Plotly
fig_components = model.plot_components(forecast)
fig_components = ff.create_2d_density(
    fig_components.data[0].x,
    fig_components.data[0].y,
    colorscale='Viridis',
    hist_color='rgba(0,100,200,0.7)'
)

----------------------------------------------------------

import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Step 1: Load and preprocess the data
df = pd.read_csv('your_dataset.csv', parse_dates=['Date'])

# Step 2: Separate the data by application
unique_applications = df['Tag_fo_application'].unique()

# Step 3: Define the parameter grid for tuning
seasonality_prior_scales = [0.01, 0.1, 1, 10]
changepoint_prior_scales = [0.001, 0.01, 0.1, 1]
growth_types = ['linear', 'logistic']

best_models = {}
best_params = {}

# Step 4: Iterate over applications and fit models
for application in unique_applications:
    # Filter data for the current application
    application_data = df[df['Tag_fo_application'] == application][['Date', 'direct_cost']]
    application_data = application_data.rename(columns={'Date': 'ds', 'direct_cost': 'y'})

    train_size = int(len(application_data) * 0.8)
    train_data = application_data[:train_size]
    valid_data = application_data[train_size:]

    best_mape = float('inf')
    best_model = None

    for seasonality_prior_scale in seasonality_prior_scales:
        for changepoint_prior_scale in changepoint_prior_scales:
            for growth_type in growth_types:
                # Fit the Prophet model with specified parameters
                model = Prophet(seasonality_prior_scale=seasonality_prior_scale,
                                changepoint_prior_scale=changepoint_prior_scale,
                                weekly_seasonality=True,
                                growth=growth_type,
                                seasonality_mode='multiplicative')

                # Fit the model
                model.fit(train_data)

                # Make predictions on the validation set
                forecast = model.predict(valid_data[['ds']])
                predictions = forecast['yhat'].values

                # Calculate the MAPE score
                mape = mean_absolute_percentage_error(valid_data['y'], predictions)

                # Check if this model has the best MAPE score so far
                if mape < best_mape:
                    best_mape = mape
                    best_model = model
                    best_params[application] = {
                        'seasonality_prior_scale': seasonality_prior_scale,
                        'changepoint_prior_scale': changepoint_prior_scale,
                        'growth_type': growth_type
                    }

    # Save the best model for this application
    best_models[application] = best_model

# Step 5: Print the best models and their parameters
for application, model in best_models.items():
    print(f"Application: {application}")
    print("Best Model Parameters:")
    print(best_params[application])
    print()

# Step 6: Example of using the best models for forecasting
for application, model in best_models.items():
    future = model.make_future_dataframe(periods=365)  # Adjust the number of periods as needed
    forecast = model.predict(future)
    # Perform further operations or analysis on the forecast data
    # ...

# Additional steps as per your requirement

# Render the interactive Plotly charts using Streamlit
st.plotly_chart(fig_forecast)
st.plotly_chart(fig_components)
