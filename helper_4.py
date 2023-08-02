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
            #actuals.append(None)
            actuals.append(None if month_forecast['yhat_lower'].values[0] > month_forecast['yhat_upper'].values[0] else month_forecast['yhat'].values[0])


    # Create the table
    table = pd.DataFrame({'Month': monthly_dates, 'Actuals': actuals, 'Forecasts': forecasts})

    return table

# Example usage
application = 'Application A'
year = 2023

table = get_actuals_forecasts_table(application, year)

# Print the table
print(table)


#######################
# Handling More Use cases:
import pandas as pd

# Function to get the actuals and forecasts table
def get_actuals_forecasts_table(application, year):
    model = best_models.get(application)
    if model is None:
        print(f"No model found for application: {application}")
        return None

    application_data = df[df['Tag_fo_application'] == application][['Date', 'direct_cost']]
    application_data = application_data.rename(columns={'Date': 'ds', 'direct_cost': 'y'})

    # Filter data for the specified year
    year_data = application_data[application_data['ds'].dt.year == year]

    # Generate monthly dates for aggregation
    monthly_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')

    # Initialize empty lists for actuals and forecasts
    actuals = []
    forecasts = []
    is_actual = []

    # Aggregate weekly predictions into monthly actuals and forecasts
    for month_start in monthly_dates:
        month_end = month_start + pd.offsets.MonthEnd()
        month_data = year_data[(year_data['ds'] >= month_start) & (year_data['ds'] <= month_end)]
        
        if len(month_data) > 0:
            # Use actuals if available for the month
            actuals.append(month_data['y'].sum())
            forecasts.append(None)  # Placeholder for forecasts since actuals are available
            is_actual.append(True)  # Mark as actual
        elif len(actuals) > 0:
            # Use forecast for the month
            month_forecast = model.predict(pd.DataFrame({'ds': [month_start]}))
            forecasts.append(month_forecast['yhat'].values[0])
            actuals.append(None)  # Placeholder for actuals since forecasts are used
            is_actual.append(False)  # Mark as forecast
        else:
            # No data available, fill with None for actuals and forecasts
            actuals.append(None)
            forecasts.append(None)
            is_actual.append(False)
    
    # Create the table
    table = pd.DataFrame({'Month': monthly_dates, 'Actuals': actuals, 'Forecasts': forecasts, 'IsActual': is_actual})
    
    return table

# Example usage
application = 'Application A'
year = 2023

table = get_actuals_forecasts_table(application, year)

# Print the table
print(table)

####################
import datetime

# User inputs
application = st.selectbox("Select Application", application_options)
yearly_prediction = st.radio("Do you want a yearly prediction for all months?", ("Yes", "No"))

if yearly_prediction == "Yes":
    current_year = datetime.date.today().year
    years = list(range(current_year - 10, current_year + 10))  # Adjust the range as needed
    year = st.selectbox("Select the year for the prediction", years)

period = st.text_input("Enter Period")
frequency = st.selectbox("Select Frequency", frequency_options)

# ------------------------------------------------------------
import pandas as pd
import redis
from snowflake.connector import connect

# Snowflake connection details
conn_params = {
    'user': '<your_username>',
    'password': '<your_password>',
    'account': '<your_account_url>',
    'warehouse': '<your_warehouse>',
    'database': '<your_database>',
    'schema': '<your_schema>',
}

# Function to establish Snowflake connection and fetch data
def fetch_data_from_snowflake():
    # Check if the data is already cached
    if redis_cache.exists('data_df'):
        # Retrieve the data from the cache
        data_df_bytes = redis_cache.get('data_df')
        data_df = pd.read_msgpack(data_df_bytes)
        return data_df

    conn = connect(**conn_params)
    # Execute the necessary SQL queries to fetch the data
    query = 'SELECT * FROM your_table'
    df = pd.read_sql(query, conn)
    conn.close()

    # Store the data in the cache
    data_df_bytes = df.to_msgpack()
    redis_cache.set('data_df', data_df_bytes)

    return df

# Initialize Redis cache connection
redis_cache = redis.Redis(host='redis-host', port=6379, db=0)

# Fetch data from Snowflake
data_df = fetch_data_from_snowflake()

# Perform analysis on the fetched data
perform_analysis(data_df)


import pickle

# Define a global variable to store the loaded models
loaded_models = None

def load_models():
    global loaded_models
    if loaded_models is None:
        # If the models are not already loaded, read them from the pickle file
        with open('model.pkl', 'rb') as file:
            loaded_models = pickle.load(file)
            
def your_function_that_uses_models():
    # Call the load_models() function before using the models
    load_models()

# Plotting the figures:
plt.figure(figsize=(10, 6))
plt.plot(forecast_application['ds'], forecast_application['y'], label='Custom Y-Axis Label')
plt.xlabel('Custom X-Axis Label')
plt.ylabel('Custom Y-Axis Label')
plt.title('Custom Forecast Plot')
plt.legend()

# Show the plot using Streamlit
st.pyplot(plt)

# Export Button:
import streamlit as st
import pandas as pd

# Sample data for the table (replace this with your actual data)
data = {
    'Column1': [1, 2, 3],
    'Column2': ['A', 'B', 'C']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Display the table in Streamlit
st.table(df)

# Add an export button to download the table as a CSV file
def download_csv():
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="table_data.csv">Download CSV File</a>'
    return href

st.markdown(download_csv(), unsafe_allow_html=True)


# Add custom CSS to style the button
st.markdown("""
<style>
.styled-button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Display the download button
st.markdown(download_csv(), unsafe_allow_html=True)
In this updated code, we use HTML and CSS to style the download button with a green background color and rounded corners. The button will be displayed below the table, and when users click on it, the CSV file will be downloaded.

You can further customize the button's appearance by modifying the CSS styles in the styled-button class. You can change the background color, font size, padding, border, and other properties to match your application's style.

Remember to replace the sample data (data) with your actual data to download the corresponding CSV file.






