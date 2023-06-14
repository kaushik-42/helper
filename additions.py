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

# Create an empty DataFrame to store the use case details
use_case_table = pd.DataFrame(columns=["Use Case", "T-Shirt Size", "Weekdays Hours/Day", "Weekends Hours/Day",
                                       "Production Storage Volume", "Production Growth",
                                       "Non-Production Storage Volume", "Non-Production Growth"])

# Function to add a new row to the use case table
def add_row():
    new_row = {"Use Case": "", "T-Shirt Size": "", "Weekdays Hours/Day": 0.0, "Weekends Hours/Day": 0.0,
               "Production Storage Volume": 0.0, "Production Growth": 0.0,
               "Non-Production Storage Volume": 0.0, "Non-Production Growth": 0.0}
    use_case_table.loc[len(use_case_table)] = new_row

# Function to remove a row from the use case table
def remove_row(index):
    use_case_table.drop(index, inplace=True)
    use_case_table.reset_index(drop=True, inplace=True)

# Function to calculate costs based on the use case details
def calculate_costs():
    result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                      "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

    for index, row in use_case_table.iterrows():
        use_case = row["Use Case"]
        tshirt_size = row["T-Shirt Size"]
        weekdays_hours_per_day = row["Weekdays Hours/Day"]
        weekends_hours_per_day = row["Weekends Hours/Day"]
        production_storage_volume = row["Production Storage Volume"]
        production_growth = row["Production Growth"]
        non_production_storage_volume = row["Non-Production Storage Volume"]
        non_production_growth = row["Non-Production Growth"]

        # Calculate costs for each use case
        monthly_costs = []
        for month in range(1, 13):
            credits_per_hour = tshirt_sizes.get(tshirt_size)
            total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))
            compute_cost = total_credits * 2.5

            production_storage_cost_per_month = 20 * production_storage_volume * (1 + production_growth / 100)
            non_production_storage_cost_per_month = 20 * non_production_storage_volume
            non_production_storage_cost_per_month = 20 * non_production_storage_volume * (1 + non_production_growth / 100)

            total_direct_costs = compute_cost + production_storage_cost_per_month + non_production_storage_cost_per_month

            monthly_costs.append({
                "Use Case": use_case,
                "Month": f"Month {month}",
                "Compute Cost": compute_cost,
                "Compute Volume": total_credits,
                "Storage Costs": production_storage_cost_per_month + non_production_storage_cost_per_month,
                "Prod Storage Volume": production_storage_volume,
                "Non-Prod Storage Volume": non_production_storage_volume
            })

        result_df = result_df.append(monthly_costs, ignore_index=True)

    return result_df


# Streamlit app
def main():
    st.title("Cloud Cost Estimation")
    st.header("Use Cases")

    # Add initial row
    add_row()

    # Display the use case table
    st.table(use_case_table)

    # Add a new row button
    if st.button("Add Row"):
        add_row()

    # Remove a row button
    if st.button("Remove Row"):
        selected_rows = st.multiselect("Select Rows to Remove", list(use_case_table.index))
        for row in selected_rows:
            remove_row(row)

    # Calculate costs button
    if st.button("Calculate Costs"):
        result_df = calculate_costs()
        st.subheader("Results")
        st.dataframe(result_df.set_index('Month'))


if __name__ == "__main__":
    main()

