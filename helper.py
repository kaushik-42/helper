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

# Get user inputs
use_case = input("Enter the use case: ")
tshirt_size = input("Enter the t-shirt size: ")
weekdays_hours_per_day = float(input("Enter the number of hours per day for weekdays: "))
weekends_hours_per_day = float(input("Enter the number of hours per day for weekends: "))

# Calculate the credits per hour based on the selected t-shirt size
credits_per_hour = tshirt_sizes.get(tshirt_size)

# Calculate the total credits per month
total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))

# Calculate Direct Costs for each month and generate yearly estimate
monthly_costs = []
yearly_estimate = 0

for month in range(1, 13):
    # Get user inputs for Compute and Storage growth percentages (if applicable)

    # Calculate Compute Cost
    compute_cost = total_credits * 2.5

    # Calculate Production Storage Cost
    production_storage_cost_per_month = Production_Storage_Cost * Production_Storage_Volume * (1 + Production_Growth)

    # Calculate Non-Production Storage Cost
    nonprod_storage_cost_per_month = NonProd_Storage_Cost * NonProd_Storage_Volume * (1 + NonProd_Growth)

    # Calculate Total Direct Costs for the month
    total_direct_costs = compute_cost + production_storage_cost_per_month + nonprod_storage_cost_per_month

    # Add the monthly cost to the list
    monthly_costs.append(total_direct_costs)

    # Accumulate monthly cost to get yearly estimate
    yearly_estimate += total_direct_costs

    # Print the monthly cost
    print(f"Month {month}: Total Direct Costs = {total_direct_costs}")

# Print the total credits per month
print(f"The total credits per month for {use_case} using {tshirt_size} is {total_credits} credits.")

# Print the yearly estimate
print(f"Yearly Estimate: Total Direct Costs = {yearly_estimate}")
