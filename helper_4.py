import pandas as pd

def get_application_dates(df):
    application_dates = {}

    unique_applications = df['Tag_fo_application'].unique()

    for application in unique_applications:
        application_data = df[df['Tag_fo_application'] == application]
        start_date = application_data['Date'].min()
        end_date = application_data['Date'].max()
        application_dates[application] = (start_date, end_date)

    return application_dates

###################################

# Step 6: Function to get the actuals and forecasts table
def get_actuals_forecasts_table(application, year):
    model = best_models.get(application)
    if model is None:
        print(f"No model found for application: {application}")
        return None

    application_data = df[df['Tag_fo_application'] == application][['Date', 'direct_cost']]
    application_data = application_data.rename(columns={'Date': 'ds', 'direct_cost': 'y'})

    # Filter data for the specified year
    year_data = application_data[(application_data['ds'].dt.year == year) & (application_data['ds'].dt.month == 1)]

    # Generate monthly dates for aggregation
    monthly_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')

    # Initialize empty lists for actuals and forecasts
    actuals = []
    forecasts = []

    # Aggregate weekly predictions into monthly actuals and forecasts
    for month_start in monthly_dates:
        month_end = month_start + pd.offsets.MonthEnd()
        month_data = year_data[(year_data['ds'] >= month_start) & (year_data['ds'] <= month_end)]
        if len(month_data) > 0:
            # Use actuals if available for the month
            actuals.append(month_data['y'].sum())
            forecasts.append(None)
        else:
            # Use forecast for the month
            month_forecast = model.predict(pd.DataFrame({'ds': [month_start]}))
            forecasts.append(month_forecast['yhat'].values[0])
            actuals.append(None)

    # Create the table
    table = pd.DataFrame({'Month': monthly_dates, 'Actuals': actuals, 'Forecasts': forecasts})

    return table

# Example usage
application = 'Application A'
year = 2023

table = get_actuals_forecasts_table(application, year)

# Print the table
print(table)
