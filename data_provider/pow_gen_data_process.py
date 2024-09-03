import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "./beigang_data/因子数据.csv"
df = pd.read_csv(file_path)
print("Original columns:", df.columns)

# Check for missing values
print("Missing value statistics:\n", df.isnull().sum())

# Compute the threshold for dropping columns
threshold = len(df) * 1/6

# Drop columns with more than 1/6 missing values
df.dropna(thresh=threshold, axis=1, inplace=True)
print("Columns after dropping those with more than 1/6 missing values:", df.columns)

# Fill remaining missing values using forward fill
df.fillna(method='ffill', inplace=True)

# Define the columns to sum
columns_to_sum = ['可门火力发电', '邵武火力发电', '永安火力发电', '漳平火力发电', '厦门火力发电']
if all(col in df.columns for col in columns_to_sum):  # Ensure columns exist
    df['total_pow_generation'] = df[columns_to_sum].sum(axis=1)
    # Drop the original columns
    df.drop(columns=columns_to_sum, axis=1, inplace=True)
else:
    print("Some columns to sum are missing after dropping.")

print("Updated columns:", df.columns)

# Save the processed data
output_path = "beigang_data/pow_gen_data.csv"
df.to_csv(output_path, index=False)
print(f"File saved successfully to {output_path}")


plt.figure(figsize=(10, 6))
plt.plot(df.index, df["total_pow_generation"])

plt.title('Power Generation of Different Power Plants Over Time')
plt.xlabel('Time')
plt.ylabel('Power Generation')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
output_plot_path = "beigang_data/power_generation_plot.png"
plt.savefig(output_plot_path, format='png')
print(f"Plot saved as {output_plot_path}")

# Optionally, show the plot
# plt.show()
