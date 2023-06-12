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

# Get user inputs for each use case
num_use_cases = int(input("Enter the number of use cases: "))

# Define dictionary to store costs for each use case
use_case_costs = {}

for i in range(num_use_cases):
    # Get user inputs for the current use case
    use_case = input(f"Enter the use case {i+1}: ")
    tshirt_size = input("Enter the t-shirt size: ")
    weekdays_hours_per_day = float(input("Enter the number of hours per day for weekdays: "))
    weekends_hours_per_day = float(input("Enter the number of hours per day for weekends: "))

    # Calculate the credits per hour based on the selected t-shirt size
    credits_per_hour = tshirt_sizes.get(tshirt_size)

    # Calculate the total credits per month
    total_credits = credits_per_hour * ((weekdays_hours_per_day * 21.7) + (weekends_hours_per_day * 8.7))

    # Get user inputs for storage volumes and growth percentages
    Production_Storage_Volume = float(input("Enter the production storage volume (in TB): "))
    Production_Growth = float(input("Enter the growth for production storage volume (in %): "))
    NonProd_Storage_Volume = float(input("Enter the non-production storage volume (in TB): "))
    NonProd_Growth = float(input("Enter the growth for non-production storage volume (in %): "))

    # Calculate Direct Costs for each month and generate yearly estimate
    monthly_costs = []
    yearly_estimate = 0

    for month in range(1, 13):
        # Calculate Compute Cost
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

        # Print the monthly cost
        print(f"Month {month}: Total Direct Costs for {use_case} = {total_direct_costs}")

    # Store the monthly costs for the use case
    use_case_costs[use_case] = monthly_costs

    # Print the total credits per month for the use case:
    print(f"The total credits per month for {use_case} using {tshirt_size} is {total_credits} credits.")

    # Print the yearly estimate for the use case
    print(f"Yearly Estimate for {use_case}: Total Direct Costs = {yearly_estimate}")

# Calculate the total yearly estimate for all use cases
total_yearly_estimate = sum(sum(costs) for costs in use_case_costs.values())

# Print the total yearly estimate for all use cases
print(f"Total Yearly Estimate for all use cases: Total Direct Costs = {total_yearly_estimate}")
