import streamlit as st
import pandas as pd

# Define the credits per hour for each t-shirt size
tshirt_sizes = {
    "x-small": 1,
    "small": 2,
    "medium": 4,
    "large": 8,
    "x-large": 16,
    "xx-large": 32,
    "3xl": 64,
    "4xl": 128,
    "5xl": 256,
    "6xl": 512,
    "sp-medium": 6,
    "sp-large": 12,
    "sp-xl": 24,
    "sp-2xl": 48,
    "sp-3xl": 96,
    "sp-4xl": 192,
    "sp-5xl": 384,
    "sp-6xl": 768
}

# Create an empty dataframe to store the results
result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                  "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

st.title("Cloud Cost Estimation")
st.header("Use Cases")

# Create a table to store use case details
use_case_table = pd.DataFrame(columns=["Use Case", "T-Shirt Size", "Weekday Hours", "Weekend Hours",
                                       "Prod Storage Volume", "Non-Prod Storage Volume"])

# Add a default row to the use case table
use_case_table = use_case_table.append({"Use Case": "", "T-Shirt Size": "", "Weekday Hours": 8.0,
                                        "Weekend Hours": 0.0, "Prod Storage Volume": 1.0,
                                        "Non-Prod Storage Volume": 1.0}, ignore_index=True)

# Function to calculate costs for all use cases
def calculate_costs():
    # Iterate through the use case table
    for index, row in use_case_table.iterrows():
        use_case = row["Use Case"]
        tshirt_size = row["T-Shirt Size"]
        weekdays_hours_per_day = row["Weekday Hours"]
        weekends_hours_per_day = row["Weekend Hours"]
        Production_Storage_Volume = row["Prod Storage Volume"]
        NonProd_Storage_Volume = row["Non-Prod Storage Volume"]

        # Calculate the credits per hour based on the selected t-shirt size
        credits_per_hour = tshirt_sizes.get(tshirt_size)

        # Calculate the total credits per month
        total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))

        # Calculate Direct Costs for each month and store in the result dataframe
        monthly_costs = []
        for month in range(1, 13):
            # Calculate Compute Cost
            compute_cost = total_credits * 2.5

            # Calculate Storage Costs
            production_storage_cost_per_month = 20 * Production_Storage_Volume
            nonprod_storage_cost_per_month = 20 * NonProd_Storage_Volume

            # Add the monthly costs to the list
            monthly_costs.append({
                "Use Case": use_case,
                "Month": f"Month {month}",
                "Compute Cost": compute_cost,
                "Compute Volume": total_credits,
                "Storage Costs": production_storage_cost_per_month + nonprod_storage_cost_per_month,
                "Prod Storage Volume": Production_Storage_Volume,
                "Non-Prod Storage Volume": NonProd_Storage_Volume})
            
# Function to calculate costs for all use cases
def calculate_costs():
    result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                      "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

    for index, row in use_case_table.iterrows():
        use_case = row["Use Case"]
        tshirt_size = row["T-Shirt Size"]
        weekdays_hours_per_day = row["Weekdays Hours/Day"]
        weekends_hours_per_day = row["Weekends Hours/Day"]
        Production_Storage_Volume = row["Production Storage Volume"]
        Production_Growth = row["Production Growth"]
        NonProd_Storage_Volume = row["Non-Production Storage Volume"]
        NonProd_Growth = row["Non-Production Growth"]

        monthly_costs = []
        for month in range(1, 13):
            credits_per_hour = tshirt_sizes.get(tshirt_size)
            total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))
            compute_cost = total_credits * 2.5

            production_storage_cost_per_month = 20 * Production_Storage_Volume * (1 + Production_Growth / 100)
            nonprod_storage_cost_per_month = 20 * NonProd_Storage_Volume * (1 + NonProd_Growth / 100)

            monthly_costs.append({
                "Use Case": use_case,
                "Month": f"Month {month}",
                "Compute Cost": compute_cost,
                "Compute Volume": total_credits,
                "Storage Costs": production_storage_cost_per_month + nonprod_storage_cost_per_month,
                "Prod Storage Volume": Production_Storage_Volume,
                "Non-Prod Storage Volume": NonProd_Storage_Volume
            })

        result_df = result_df.append(monthly_costs, ignore_index=True)

    return result_df

# Main code
st.title("Cloud Cost Estimation")
st.header("Use Cases")

use_case_table = pd.DataFrame(columns=["Use Case", "T-Shirt Size", "Weekdays Hours/Day",
                                       "Weekends Hours/Day", "Production Storage Volume",
                                       "Production Growth", "Non-Production Storage Volume",
                                       "Non-Production Growth"])

add_row()
result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                  "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

# Display the use case table
st.subheader("Use Case Details")
st.dataframe(use_case_table)

# Add buttons to add or remove rows in the use case table
if st.button("Add Use Case"):
    add_row()

if len(use_case_table) > 1:
    remove_indices = []
    for index, row in use_case_table.iterrows():
        if st.button(f"Remove Use Case {index+1}"):
            remove_indices.append(index)
    for index in remove_indices:
        remove_row(index)

# Calculate the costs for all use cases
if st.button("Calculate"):
    result_df = calculate_costs()

# Display the result dataframe
st.subheader("Results")
st.dataframe(result_df.set_index('Month'))

# Check if the user wants to clear the result dataframe
if st.button("Clear Results"):
    result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                      "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

# Run the Streamlit app
st.write("Use case details and results have been successfully processed!")
