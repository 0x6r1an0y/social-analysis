import os
import pandas as pd

# 設定資料夾路徑（請自行修改）
folder_path = '../../data/csv'
# 合併後要儲存的檔案名稱
output_file = 'merged_output.csv'

# 收集所有 .csv 檔案的路徑
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 用來存放讀取的資料
dataframes = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# 合併所有 DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# 存成新的 CSV 檔案
merged_df.to_csv(output_file, index=False)

print(f"合併完成，已儲存為：{output_file}")
