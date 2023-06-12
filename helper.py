import streamlit as st

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

    num_use_cases = st.number_input("Enter the number of use cases:", min_value=1, max_value=5, value=1, step=1)

    for i in range(num_use_cases):
        st.header(f"Use Case {i+1}")
        use_case = st.text_input("Enter the use case:")
        tshirt_size = st.selectbox("Select the t-shirt size:", list(tshirt_sizes.keys()))
        weekdays_hours_per_day = st.number_input("Enter the number of hours per day for weekdays:", min_value=0.0, value=8.0)
        weekends_hours_per_day = st.number_input("Enter the number of hours per day for weekends:", min_value=0.0, value=0.0)

        # Get user inputs for storage volumes and growth percentages
        Production_Storage_Volume = st.number_input("Enter the production storage volume (in TB):", min_value=0.0, value=1.0)
        Production_Growth = st.number_input("Enter the growth for production storage volume (in %):", min_value=0.0, value=0.0)
        NonProd_Storage_Volume = st.number_input("Enter the non-production storage volume (in TB):", min_value=0.0, value=1.0)
        NonProd_Growth = st.number_input("Enter the growth for non-production storage volume (in %):", min_value=0.0, value=0.0)

        # Calculate Direct Costs for each month and generate yearly estimate
        monthly_costs = []
        yearly_estimate = 0

        for month in range(1, 13):
            # Calculate Compute Cost
            credits_per_hour = tshirt_sizes.get(tshirt_size)
            total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))
            compute_cost = total_credits * 2.5

            # Calculate Production Storage Cost
            production_storage_cost_per_month = 20 * Production_Storage_Volume * (1 + Production_Growth / 100)

            # Calculate Non-Production Storage Cost
            nonprod_storage_cost_per_month = 20 * NonProd_Storage_Volume * (1 + NonProd_Growth / 100)

            # Calculate Total Direct Costs for the month
            total_direct_costs = compute_cost + production_storage_cost_per_month + nonprod_storage_cost_per_month

            # Add the monthly cost to the list
            monthly_costs.append(total_direct_costs)

            # Accumulate monthly cost to get yearly estimate
            yearly_estimate += total_direct_costs

            # Display the monthly cost
            st.write(f"Month {month}: Total Direct Costs for {use_case} = {total_direct_costs}")

        # Store the monthly costs for the use case
        use_case_costs[use_case] = monthly_costs

        # Display the total credits per month for the use case
        st.write(f"The total credits per month for {use_case} using {tshirt_size} is {total_credits} credits.")

        # Display the yearly estimate for the use case
        st.write(f"Yearly Estimate for {use_case}: Total Direct Costs = {yearly_estimate}")

    # Calculate the total yearly estimate for all use cases
    total_yearly_estimate = sum(sum(costs) for costs in use_case_costs.values())

    # Display the total yearly estimate for all use cases
    st.write(f"Total Yearly Estimate for all use cases: Total Direct Costs = {total_yearly_estimate}")

# Run the Streamlit app
calculate_costs()
