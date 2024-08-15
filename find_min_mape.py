import pandas as pd

# 文件路径
file_path = 'result_long_term_forecast_inverse.txt'

# 初始化一个空的列表来存储符合条件的行
data = []
# 读取文件
with open(file_path, 'r') as file:
    for line in file:
        if 'beigang' in line:
            current_line = line.strip()  # 保存含有 'beigang' 的行
        elif 'mape:' in line and current_line:
            mape_str = line.split('mape:')[1].split(',')[0]
            mape_value = float(mape_str)
            data.append((current_line, mape_value))  # 将行和 mape 存入列表
            current_line = ""  # 重置行以查找下一个 'beigang'

# 将数据转换为 DataFrame 并排序
df = pd.DataFrame(data, columns=['line', 'mape'])
df_sorted = df.sort_values(by='mape').head(10)

# 打印前10行
for index, row in df_sorted.iterrows():
    print(f"Line: {row['line']}, MAPE: {row['mape']}")