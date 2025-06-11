import pandas as pd

# 讀取原始資料
df = pd.read_csv("merged_output.csv")
original_count = len(df)

print(f"原始總筆數：{original_count}")

# 清除所有欄位皆為空的列
empty_all_count = df.isnull().all(axis=1).sum()
df = df.dropna(how='all')

# 清除 pos_tid 為空的列
empty_pos_tid_count = df['pos_tid'].isnull().sum() + (df['pos_tid'].astype(str).str.strip() == '').sum()
df = df[df['pos_tid'].notna() & (df['pos_tid'].astype(str).str.strip() != '')]

# 清除 content 為空的列
empty_content_count = df['content'].isnull().sum() + (df['content'].astype(str).str.strip() == '').sum()
df = df[df['content'].notna() & (df['content'].astype(str).str.strip() != '')]

# 清除 pos_tid 重複的列（保留第一筆）
duplicate_count = df.duplicated(subset='pos_tid').sum()
df = df.drop_duplicates(subset='pos_tid', keep='first')

# 最後資料筆數
final_count = len(df)

# 儲存結果
df.to_csv("cleaned_output.csv", index=False)

# 印出統計資訊
print(f"🔍 清洗筆數統計：")
print(f"    ✂️ 全欄空值刪除：{empty_all_count} 筆")
print(f"    ✂️ 主鍵 pos_tid 為空刪除：{empty_pos_tid_count} 筆")
print(f"    ✂️ content 為空刪除：{empty_content_count} 筆")
print(f"    ✂️ 主鍵 pos_tid 重複刪除：{duplicate_count} 筆")
print(f"✅ 清洗後剩下：{final_count} 筆")
print(f"📁 已輸出至 cleaned_output.csv")
