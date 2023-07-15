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
