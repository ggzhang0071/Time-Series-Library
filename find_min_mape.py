import pandas as pd

# File path
file_path = 'result_long_term_forecast_inverse.txt'

# Initialize an empty list to store the lines with their respective acc values
data = []

# Read the file
with open(file_path, 'r') as file:
    current_line = ""  # Initialize variable to store lines with 'beigang' and 'pl15'
    for line in file:
        if 'pl5' in line:
            current_line = line.strip()  # Save the line containing 'beigang' and 'pl15'
        elif 'Min_acc:' in line and 'Max_acc:' in line and current_line:
            min_acc_str = line.split('Min_acc:')[1].split(',')[0]
            max_acc_str = line.split('Max_acc:')[1].split(',')[0]
            min_acc_value = float(min_acc_str)
            max_acc_value = float(max_acc_str)
            acc_value = (min_acc_value + max_acc_value) / 2  # Calculate the average of Min_acc and Max_acc
            full_line = f"{current_line} {line.strip()}"  # Combine the two lines
            data.append((full_line, acc_value, min_acc_value))  # Store the full line, acc and min_acc values
            current_line = ""  # Reset for the next 'beigang'

# Convert the data to a DataFrame and sort first by acc (descending), then by min_acc (descending)
df = pd.DataFrame(data, columns=['full_line', 'acc', 'min_acc'])
df_sorted = df.sort_values(by=['acc', 'min_acc'], ascending=[False, False]).head(20)

# Print the top 10 rows with the full original lines
for index, row in df_sorted.iterrows():
    print(f"Line: {row['full_line']}")
