import streamlit as st
import pandas as pd

# Create an empty dataframe to store the use case details
use_case_table = pd.DataFrame(columns=["Use Case", "T-shirt Size", "Weekdays Hours", "Weekends Hours",
                                       "Prod Storage Volume", "Non-Prod Storage Volume"])

def add_row():
    global use_case_table
    new_row = {"Use Case": "", "T-shirt Size": "", "Weekdays Hours": 0, "Weekends Hours": 0,
               "Prod Storage Volume": 0, "Non-Prod Storage Volume": 0}
    use_case_table = use_case_table.append(new_row, ignore_index=True)

def remove_row(index):
    global use_case_table
    use_case_table = use_case_table.drop(index)

def calculate_costs():
    result_df = pd.DataFrame(columns=["Use Case", "Month", "Compute Cost", "Compute Volume",
                                      "Storage Costs", "Prod Storage Volume", "Non-Prod Storage Volume"])

    for _, row in use_case_table.iterrows():
        use_case = row["Use Case"]
        tshirt_size = row["T-shirt Size"]
        weekdays_hours_per_day = row["Weekdays Hours"]
        weekends_hours_per_day = row["Weekends Hours"]
        production_storage_volume = row["Prod Storage Volume"]
        non_production_storage_volume = row["Non-Prod Storage Volume"]

        # Calculate the credits per hour based on the selected t-shirt size
        credits_per_hour = tshirt_sizes.get(tshirt_size)

        # Calculate the total credits per month
        total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))

        # Calculate Compute Cost
        compute_cost = total_credits * 2.5

        # Calculate Storage Costs
        production_storage_cost_per_month = 20 * production_storage_volume
        non_production_storage_cost_per_month = 20 * non_production_storage_volume

        # Calculate Total Direct Costs for each month
        monthly_costs = []
        for month in range(1, 13):
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
    if len(use_case_table) == 0:
        add_row()

    # Display the use case table
    for i, row in use_case_table.iterrows():
        st.text_input(f"Use Case {i+1}", value=row["Use Case"], key=f"use_case_input_{i}")
        st.selectbox(f"T-shirt Size {i+1}", list(tshirt_sizes.keys()), index=list(tshirt_sizes.keys()).index(row["T-shirt Size"]), key=f"tshirt_size_input_{i}")
        st.number_input(f"Weekdays Hours {i+1}", min_value=0.0, value=row["Weekdays Hours"], key=f"weekdays_hours
        st.number_input(f"Weekends Hours {i+1}", min_value=0.0, value=row["Weekends Hours"], key=f"weekends_hours_input_{i}")
        st.number_input(f"Prod Storage Volume {i+1}", min_value=0.0, value=row["Prod Storage Volume"], key=f"prod_storage_input_{i}")
        st.number_input(f"Non-Prod Storage Volume {i+1}", min_value=0.0, value=row["Non-Prod Storage Volume"], key=f"nonprod_storage_input_{i}")
        st.button(f"Remove Row {i+1}", on_click=lambda i=i: remove_row(i))

    if st.button("Add Row"):
        add_row()

    if st.button("Calculate Costs"):
        result_df = calculate_costs()
        st.subheader("Results")
        st.dataframe(result_df)

# Run the Streamlit app
if __name__ == "__main__":
    main()
