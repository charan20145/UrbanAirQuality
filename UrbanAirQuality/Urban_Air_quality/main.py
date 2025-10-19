import pandas as pd

# Load dataset
df = pd.read_csv("urban_air_quality_dataset.csv")

# Show first 5 rows
print("Dataset loaded successfully!")
print(df.head())

# Show info
print("\nDataset Info:")
print(df.info())

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe())
