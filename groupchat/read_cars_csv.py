# filename: read_cars_csv.py
import pandas as pd

# Load the data from the CSV file (assuming the file is named car.csv)
data = pd.read_csv("car.csv")

# Display the first few rows of the dataset to understand its structure
print(data.head())

# You can then analyze or extract information based on the structure of the data.

# TERMINATE