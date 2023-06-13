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

def calculate_costs():
    st.title("Cloud Cost Estimation")
    st.header("Use Cases")
    
    # Create an empty dataframe to store the results
    result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                      "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

    while True:
        st.subheader("Use Case Details")
        
        use_case = st.text_input("Enter the use case:", key=f"use_case_{len(result_df)+1}")
        tshirt_size = st.selectbox("Select the t-shirt size:", list(tshirt_sizes.keys()), key=f"tshirt_size_{len(result_df)+1}")
        weekdays_hours_per_day = st.number_input("Enter the number of hours per day for weekdays:", min_value=0.0, value=8.0, key=f"weekday_hours_{len(result_df)+1}")
        weekends_hours_per_day = st.number_input("Enter the number of hours per day for weekends:", min_value=0.0, value=0.0, key=f"weekend_hours_{len(result_df)+1}")
        Production_Storage_Volume = st.number_input("Enter the production storage volume (in TB):", min_value=0.0, value=1.0, key=f"prod_volume_{len(result_df)+1}")
        Production_Growth = st.number_input("Enter the growth for production storage volume (in %):", min_value=0.0, value=0.0, key=f"prod_growth_{len(result_df)+1}")
        NonProd_Storage_Volume = st.number_input("Enter the non-production storage volume (in TB):", min_value=0.0, value=1.0, key=f"nonprod_volume_{len(result_df)+1}")
        NonProd_Growth = st.number_input("Enter the growth for non-production storage volume (in %):", min_value=0.0, value=0.0, key=f"nonprod_growth_{len(result_df)+1}")
        # Calculate Direct Costs for each month
        monthly_costs = []
        for month in range(1, 13):
            # Calculate Compute Cost
            credits_per_hour = tshirt_sizes.get(tshirt_size)
            total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))
            compute_cost = total_credits * 2.5

            # Calculate Storage Costs
            production_storage_cost_per_month = 20 * Production_Storage_Volume * (1 + Production_Growth / 100)
            nonprod_storage_cost_per_month = 20 * NonProd_Storage_Volume * (1 + NonProd_Growth / 100)

            # Add the monthly costs to the list
            monthly_costs.append({
                "Use Case": use_case,
                "Month": f"Month {month}",
                "Compute Cost": compute
                "Compute Cost": compute_cost,
                "Compute Volume": total_credits,
                "Storage Costs": production_storage_cost_per_month + nonprod_storage_cost_per_month,
                "Prod Storage Volume": Production_Storage_Volume,
                "Non-Prod Storage Volume": NonProd_Storage_Volume
            })

        # Append the monthly costs to the result dataframe
        result_df = result_df.append(monthly_costs, ignore_index=True)

        # Check if the user wants to continue entering use cases
        continue_entering = st.button("Continue entering use cases")
        if not continue_entering:
            break

    # Display the result dataframe
    st.subheader("Results")
    st.dataframe(result_df.set_index('Month'))

    # Check if the user wants to clear the result dataframe
    if st.button("Clear Results"):
        result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                          "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

# Run the Streamlit app
calculate_costs()
