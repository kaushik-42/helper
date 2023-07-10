import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import pickle
import plotly.figure_factory as ff
import altair as alt
from prophet import Prophet
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title = "Forecasting Budgets Demo", page_icon = "🔮")

st.markdown("# Forecasting Budgets 🔮")
st.sidebar.header("Forecasting Demo")
st.write(
    """
    This Forecasting Budgets page provides the Forecasting for the Future cloud spends (Budgets) in terms of the Direct Costs and Shared Costs.
    
    **TODO:  You have to select the desired application from the Application Drop down box, Provide the Period value in terms of a number, where each Period represents w.r.t Frequency.**
    
    Let's say if we select "W" frequency from the drop-down box, then each and every period represents a week. If the Frequency is "W" and the number of periods are 30, then we get a forecasting for the next 30 weeks.
    
    This Forecasting is currently being done on the "Application-level" and mainly the target variable represents the "Direct Costs". To view all the applications which are available for Forecasting: Check the drop-down box by selecting the application.
    """
)

st.markdown("### Forecasting Prediction for Snowflake Warehouse:")

with open('best_models.pkl', 'rb') as file:
	best_models = pickle.load(file)

def helper(application_model, periods, freq, df_temp):
    # Creating the Forecasting data i.e the Future data points based on the Periods and Frequency, which will provide the Forecasting
    future_application = application_model.make_future_dataframe(periods=int(periods), freq=freq)

    # Forecast data for an application:
    forecast_application = application_model.predict(future_application)

    if(df_temp.empty):
        pass
    else:
        st.write("**Analyzing Budgets for 12 Months from the Budgets Page:**")
        try:
            df_temp.set_index("Month ", inplace = True)
            print(df_temp.columns)
            st.line_chart(df_temp["Total Direct Costs"])
        except:
            pass

    st.write("Forecast Results:")
    # TODO:
    # Visualize forecast plot with Plotly
    st.pyplot(application_model.plot(forecast_application))
    st.write("""
     The Below Plots describes the Trend Component, Seasonality Component, Bias Component, Yearly/Monthly/Weekly/Daily trends. 

     Note that the above plots depends on the Application, Data Points present for that application etc.
     """)

    st.pyplot(application_model.plot_components(forecast_application))
    st.write("To Analyze more regarding the Future data points:")

    df = pd.DataFrame(forecast_application)
    #print(df)
    #print(df.columns)
    df_application = df[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
    df_application.columns = ['Date','Prediction Lower Limit', 'Prediction Upper Limit', 'Prediction Point']

    #df_application['Date'] = df_application['Date'].apply(lambda x: f'color: green')
    #styler = df.style

    #styled_df = styler.apply(lambda _: 'color: green', subset=['Date'])

    st.dataframe(df_application.style.background_gradient(subset=['Prediction Point'], cmap="BrBG"), width=800)

    # st.dataframe(df_application, width=800)
    # Plotting the Forecast:
    #st.pyplot(application_model.plot(forecast_application))

    #st.pyplot(application_model.plot_components(forecast_application))
    return ""

# Define the available application options and frequencies
application_options = ['340B ANALYTICS', 'ABC_DE_DSS', 'ADOBE', 'AI GOVERNANCE', 'AIR FLOW DATA SUPPORT', 'ANALYTIC PRODUCTS', 'ASSET PROTECTION', 'BI COE', 'BI ENGINEERING RETAIL DATA STRATEGY', 'CAREPASS ANALYTICS', 'CDP', 'CENTRAL FILL', 'CEX', 'CLINICAL DATA REPOSITORY', 'CLINICAL TRIALS', 'CLOUD COST REPORTING', 'CMX', 'COVID VACCINATION', 'COVID VACCINATION CUSTOMER LOYALTY', 'CS&G', 'CUSTOMER EXPERIENCE', 'DAILY DIGEST', 'DBA', 'DEMAND_FORECAST', 'DEWA', 'DEWA DATA MIGRATION TO SNOWFLAKE', 'DIGITAL - POC', 'DIGITAL MERCHANDISING', 'DIGITAL RETAIL ONSITE SEARCH PLATFORM', 'DME ANALYTICS', 'DUR', 'EDO', 'EDP', 'EDPCLOUDECBI', 'EHS', 'ENROLLED SERVICES', 'EPSO', 'EVENT DRIVEN OUTREACH', 'EXECUTIVE COMPLAINTS TEAM', 'EXECUTIVE DASHBOARD', 'FINANCIAL ATTRIBUTION', 'FL DL', 'FRONT STORE ASSORTMENT - BCG', 'FRONT STORE SUPPLY CHAIN', 'FS ANALYTICS', 'FS BI SOLUTIONS', 'FS DATA STRATEGY', 'FS_WORKFLOW', 'GIC HH-COMMON DATA MODEL', 'HR', 'HUMAN RESOURCES', 'IMMUNIZATION', 'INGEST - RETAIL RX', 'LEARNING HUB', 'LOYALTY & PERSONALIZATION', 'MARKETING AND CUSTOMER ANALYTICS', 'MC ANALYTICS', 'MEDIA DELIVERY DATA MART', 'MERCH BU', 'MICROSTRATEGY', 'MINUTE CLINIC', 'MPC THOUGHTSPOT', 'NCPDP', 'NETWORK OPERATIONS', 'NUTRITION', 'OMNIRX', 'OUTREACH PORTAL', 'PALANTIR', 'PANEL OUTREACH', 'PATIENT JOURNEY', 'PATIENT MERGE', 'PAY FOR PERFORMANCE', 'PERSONIZATION ENGINE - FRONT STORE RETAIL', 'PHARMACY OPERATIONS', 'PRODUCT DEVELOPMENT DATA LAB', 'PROFESSIONAL PRACTICE', 'PROMO', 'PROMO FORECAST', 'RCC ANALYTICS', 'REAL TIME INTEGRATION TESTING', 'RELATIONSHIP MARKETING- FINANCIAL ATTRIBUTION', 'RETAIL ANALYTICS', 'RETAIL CUSTOMER GROWTH ANALYTICS', 'RETAIL MERCHANDISING ANALYTICS', 'RETAIL RX', 'REVENUE CYCLE', 'RPHAI', 'RPHAI-DUR', 'RPHAI-REPORTING', 'RX ANALYTICS - B2B INSIGHTS', 'RX ANALYTICS - RONBA', 'RX IMAGING VIRTUAL VERIFICATION', 'RX OPERATIONS', 'RX OPERATIONS - COMMON', 'RX OPS-IMZ', 'RX PERSONALIZATION', 'RX PRACTICE INNOVATION', 'RX STORE OPS ANALYTICS - FDD', 'RX STORE OPS ANALYTICS - HR', 'RXDW', 'RXDW - IT', 'RXOPSTCT', 'RXPERSONALIZATION', 'SCRIPT PROFITABILITY', 'SNOWFLAKE', 'SNOWFLAKE DBA SUPPORT', 'SNOWFLAKE MARKETPLACE', 'SOCS COMPLIANCE', 'SOM', 'SPM', 'STARBURST BUSINESS', 'STORE DIGEST', 'STORE EXPERIENCE - IN STORE FULFILMENT', 'STORE EXPERIENCE-IN STORE FULFILMENT', 'STORE OPERATIONS', 'STRATEGIC PLANNING AND ANALYSIS', 'SUPPLY CHAIN', 'SUPPLY CHAIN ANALYTICS', 'SUPPLY CHAIN INNOVATION', 'TABLEAU', 'THIRD PARTY FINANCE', 'THOUGHTSPOT', 'THOUGHTSPOT FSGA']

#for application in application_options:
#    print(application)
#frequency_options = ['D', 'W', 'M']
frequency_options = ['Daily', 'Weekly', 'Monthly']
#print(best_models)

# Streamlit app
def main():

    # Importing inputs from other pages (Budget Page):
    try:
        df_temp = st.session_state["df"]
        print('--------------------')
        #st.pyplot(df_temp)
        #print(df_temp)
        #df_temp = pd.DataFrame(df_temp)
        #print(df_temp)
        #st.pyplot(df_temp)
    except:
        df_temp = pd.DataFrame()

    # User inputs
    application = st.selectbox("Select Application", application_options)
    period = st.text_input("Enter Period")
    frequency = st.selectbox("Select Frequency", frequency_options)

    # Accepting a date input:
    date = st.date_input(
        "Enter the Time frame limit (From Date)",
        datetime.date(2021, 7, 6))
    ending_date = st.date_input(
        "Enter the Time frame limit (Till Date)",
        datetime.date(2021, 9, 10))

    # Format: YYYY/MM/DD
    #st.write('Your requested date starting from:', date)
    #st.write('Yor requested date till:', ending_date)

    # Selecting that respective application model from the picke file we have downloaded.
    application_model = best_models[application]
    # Submit button
    if st.button("Submit"):
        if application and period and frequency:  # Check if all fields are selected
            # Display the forecast results
            if(frequency == 'Daily'):
                frequency_t = 'D'
            elif(frequency == 'Weekly'):
                frequency_t = 'W'
            else:
                frequency_t = 'M'
            # Call the function to train the Prophet model and generate the forecast
            forecast = helper(application_model, period, frequency_t, df_temp)

        else:
            st.warning("Please select all fields.")

# Run the Streamlit app:
if __name__ == "__main__":
    main()


