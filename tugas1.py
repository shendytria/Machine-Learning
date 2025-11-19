import pandas as pd

# Read the CSV file
df = pd.read_csv('teen_phone_addiction_dataset.csv')

# Display the first few rows of the data
print(df.head())

# Select specific columns
df_selected = df[['Name', 'Age', 'Location']]
print(df_selected.head())
