import pandas as pd
from sqlalchemy import create_engine, text
import time
import psycopg2
import os
import sys

DB_TYPE = "postgresql"
# --- 1. 設定資料庫連接參數 ---
# PostgreSQL 範例
DB_USER = "postgres"        # 你的資料庫使用者名稱
DB_PASSWORD = "00000000"    # 你的資料庫密碼
DB_HOST = "localhost"            # 資料庫主機 (若是本機通常是 localhost)
DB_PORT = "5432"                 # 資料庫埠號 (PostgreSQL 預設 5432)
DB_NAME = "social_media_analysis" # 你創建的資料庫名稱
TABLE_NAME = "posts"             # 你要創建的資料表名稱

# --- 2. 設定 CSV 檔案路徑和分塊大小 ---
CSV_FILE_PATH = "merged_output.csv"
CHUNK_SIZE = 2000000  # 每次處理的行數，可根據你的記憶體大小調整 (例如 10,000 到 100,000)

# --- 3. 檢查並創建資料庫 ---
try:
    # 先連接到預設的 postgres 資料庫
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database="postgres"  # 連接到預設資料庫
    )
    conn.autocommit = True  # 自動提交，這樣創建資料庫的指令才會生效
    cursor = conn.cursor()
    
    # 檢查資料庫是否存在
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
    exists = cursor.fetchone()
    
    if not exists:
        print(f"資料庫 '{DB_NAME}' 不存在，正在創建...")
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"資料庫 '{DB_NAME}' 創建成功！")
    else:
        print(f"資料庫 '{DB_NAME}' 已存在。")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"檢查/創建資料庫時發生錯誤: {e}")
    print("\n可能的解決方案:")
    print("1. 確認 PostgreSQL 服務是否已安裝並運行")
    print("2. 確認資料庫使用者名稱和密碼是否正確")
    print("3. 確認 PostgreSQL 是否正在監聽 5432 埠")
    print("\n如需安裝 PostgreSQL，請前往: https://www.postgresql.org/download/windows/")
    sys.exit(1)

# --- 4. 創建 SQLAlchemy 引擎 ---
try:
    engine_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url)
    
    # 測試連接
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print("成功連接到 PostgreSQL 資料庫！")
except Exception as e:
    print(f"連接資料庫時發生錯誤: {e}")
    sys.exit(1)

# --- 5. 創建資料表結構 ---
# 如果資料表已存在，這段可以跳過或修改
# 注意：欄位類型需要根據你的 CSV 內容精確定義
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    pos_tid VARCHAR(255) PRIMARY KEY,
    post_type VARCHAR(255),
    page_category TEXT,
    page_name TEXT,
    page_id VARCHAR(255),
    content TEXT,
    created_time BIGINT,
    reaction_all INTEGER,
    comment_count INTEGER,
    share_count INTEGER,
    date DATE
);
"""
# 執行創建資料表的 SQL
try:
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))
        connection.commit() # 對 DDL 操作（如 CREATE TABLE）進行 commit
    print(f"資料表 '{TABLE_NAME}' 檢查/創建成功。")
except Exception as e:
    print(f"創建資料表時發生錯誤: {e}")
    print("請檢查資料庫連接設定或聯絡管理員。")
    sys.exit(1)


# --- 6. 分塊讀取 CSV 並寫入資料庫 ---
start_time = time.time()
total_rows_processed = 0

print(f"開始從 '{CSV_FILE_PATH}' 匯入資料到資料表 '{TABLE_NAME}'...")

try:
    # 檢查檔案是否存在
    if not os.path.exists(CSV_FILE_PATH):
        print(f"錯誤: 找不到 CSV 檔案 '{CSV_FILE_PATH}'。")
        sys.exit(1)

    # 獲取 CSV 檔案的表頭，以確保 to_sql 時欄位名稱正確
    # 如果CSV沒有表頭，或者你想手動指定，可以調整
    # header_df = pd.read_csv(CSV_FILE_PATH, nrows=0)
    # column_names = header_df.columns.tolist()
    # print(f"CSV 欄位: {column_names}")

    for i, chunk in enumerate(pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk_start_time = time.time()
        print(f"正在處理第 {i+1} 個區塊...")

        # (可選) 資料清理或轉換
        # 例如，如果 'date' 欄位是字串，需要轉換成 datetime 物件
        if 'date' in chunk.columns:
            try:
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            except Exception as e:
                print(f"轉換日期欄位時發生錯誤: {e}")
                print("繼續處理，但日期欄位可能無法正確轉換...")
        
        # 確保 created_time 是整數
        if 'created_time' in chunk.columns and not pd.api.types.is_numeric_dtype(chunk['created_time']):
            try:
                chunk['created_time'] = pd.to_numeric(chunk['created_time'], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                print(f"轉換 created_time 欄位時發生錯誤: {e}")
                print("繼續處理，但 created_time 欄位可能無法正確轉換...")

        # 將數據寫入 SQL 資料庫
        try:
            chunk.to_sql(TABLE_NAME, engine, if_exists='append', index=False, method='multi')
            total_rows_processed += len(chunk)
            chunk_time_taken = time.time() - chunk_start_time
            print(f"第 {i+1} 個區塊 ({len(chunk)} 筆資料) 已處理並插入，耗時 {chunk_time_taken:.2f} 秒。")
            print(f"目前已處理總資料筆數: {total_rows_processed}")

        except Exception as e:
            print(f"將第 {i+1} 個區塊插入資料庫時發生錯誤: {e}")
            print("有問題的區塊資料範例 (前 5 筆):")
            print(chunk.head())
            print("繼續處理下一個區塊...")
            continue # 跳過這個 chunk

    end_time = time.time()
    print(f"成功匯入 {total_rows_processed} 筆資料到 '{TABLE_NAME}'。")
    print(f"總耗時: {(end_time - start_time):.2f} 秒。")

except FileNotFoundError:
    print(f"錯誤: 找不到檔案 '{CSV_FILE_PATH}'。")
    sys.exit(1)
except Exception as e:
    print(f"發生未預期的錯誤: {e}")
    sys.exit(1)