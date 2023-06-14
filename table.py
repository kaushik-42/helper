import streamlit as st
import pandas as pd

def main():
    st.title("Cloud Cost Estimation")
    st.header("Use Cases")

    # Create an empty dataframe to store the use cases
    use_case_df = pd.DataFrame(columns=["Use Case", "T-shirt Size", "Weekdays Hours", "Weekends Hours",
                                        "Prod Storage Volume", "Non-Prod Storage Volume"])

    # Function to add a new row to the dataframe
    def add_row():
        default_row = {
            "Use Case": "",
            "T-shirt Size": "",
            "Weekdays Hours": 8.0,
            "Weekends Hours": 0.0,
            "Prod Storage Volume": 1.0,
            "Non-Prod Storage Volume": 1.0
        }
        use_case_df.loc[len(use_case_df)] = default_row

    # Function to remove a row from the dataframe
    def remove_row(index):
        use_case_df.drop(index, inplace=True)

    # Display the use case table
    st.subheader("Use Case Details")
    for i, row in use_case_df.iterrows():
        st.text_input(f"Use Case {i+1}", value=row["Use Case"], key=f"use_case_input_{i}")
        st.text_input(f"T-shirt Size {i+1}", value=row["T-shirt Size"], key=f"tshirt_size_input_{i}")
        st.number_input(f"Weekdays Hours {i+1}", min_value=0.0, value=row["Weekdays Hours"], key=f"weekdays_hours_input_{i}")
        st.number_input(f"Weekends Hours {i+1}", min_value=0.0, value=row["Weekends Hours"], key=f"weekends_hours_input_{i}")
        st.number_input(f"Prod Storage Volume {i+1}", min_value=0.0, value=row["Prod Storage Volume"], key=f"prod_storage_input_{i}")
        st.number_input(f"Non-Prod Storage Volume {i+1}", min_value=0.0, value=row["Non-Prod Storage Volume"], key=f"nonprod_storage_input_{i}")
        st.button(f"Remove Row {i+1}", on_click=lambda i=i: remove_row(i))

    if st.button("Add Row"):
        add_row()

# Run the Streamlit app
if __name__ == "__main__":
    main()
