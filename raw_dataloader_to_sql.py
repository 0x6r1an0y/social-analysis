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
CSV_FILE_PATH = "cleaned_output.csv"
CHUNK_SIZE = 100000  # 每次處理的行數，可根據你的記憶體大小調整 (例如 10,000 到 100,000)

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
    post_type VARCHAR(255) DEFAULT 'unknown',  -- 設定預設值
    page_category TEXT,
    page_name TEXT,
    page_id VARCHAR(255),
    content TEXT,
    created_time BIGINT,
    reaction_all BIGINT,  -- 改用 BIGINT 以支援更大的數值範圍
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
error_rows = []  # 用於記錄錯誤的資料
error_details = []  # 用於記錄詳細錯誤資訊

print(f"開始從 '{CSV_FILE_PATH}' 匯入資料到資料表 '{TABLE_NAME}'...")

def analyze_error_data(chunk, error_msg):
    """分析錯誤資料的詳細資訊"""
    error_info = {
        'error_type': type(error_msg).__name__,
        'error_message': str(error_msg),
        'sample_data': None,
        'data_types': None,
        'null_counts': None,
        'duplicate_pos_tid': None
    }
    
    # 檢查資料型別
    error_info['data_types'] = chunk.dtypes.to_dict()
    
    # 檢查空值數量
    error_info['null_counts'] = chunk.isnull().sum().to_dict()
    
    # 檢查 pos_tid 重複
    if 'pos_tid' in chunk.columns:
        duplicates = chunk[chunk.duplicated(subset=['pos_tid'], keep=False)]
        if not duplicates.empty:
            error_info['duplicate_pos_tid'] = duplicates['pos_tid'].tolist()
    
    # 取樣錯誤資料（最多5筆）
    error_info['sample_data'] = chunk.head().to_dict('records')
    
    return error_info

try:
    if not os.path.exists(CSV_FILE_PATH):
        print(f"錯誤: 找不到 CSV 檔案 '{CSV_FILE_PATH}'。")
        sys.exit(1)

    for i, chunk in enumerate(pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk_start_time = time.time()
        print(f"正在處理第 {i+1} 個區塊...")

        # 資料清理和轉換
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        
        if 'created_time' in chunk.columns:
            chunk['created_time'] = pd.to_numeric(chunk['created_time'], errors='coerce').fillna(0).astype(int)
            
        # 處理 post_type 的空值
        if 'post_type' in chunk.columns:
            chunk['post_type'] = chunk['post_type'].fillna('unknown')
            
        # 處理 reaction_all 的數值範圍問題
        if 'reaction_all' in chunk.columns:
            chunk['reaction_all'] = pd.to_numeric(chunk['reaction_all'], errors='coerce').fillna(0).astype('Int64')  # 使用可空整數類型

        # 使用原生 SQL 插入來提高效能
        try:
            # 將 DataFrame 轉換為值列表
            values = chunk.values.tolist()
            columns = chunk.columns.tolist()
            
            # 建立插入語句
            insert_stmt = f"""
                INSERT INTO {TABLE_NAME} ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(columns))})
            """

            '''
            insert_stmt = f"""
                INSERT INTO {TABLE_NAME} ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(columns))})
                ON CONFLICT (pos_tid) DO NOTHING
            """'''
            # 使用 psycopg2 直接插入
            with psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME
            ) as conn:
                with conn.cursor() as cur:
                    try:
                        cur.executemany(insert_stmt, values)
                        conn.commit()
                    except psycopg2.Error as db_error:
                        # 詳細分析資料庫錯誤
                        error_info = analyze_error_data(chunk, db_error)
                        error_details.append({
                            'chunk_number': i + 1,
                            'error_info': error_info
                        })
                        
                        print(f"\n🔍 區塊 {i+1} 插入失敗，錯誤分析：")
                        print(f"錯誤類型: {error_info['error_type']}")
                        print(f"錯誤訊息: {error_info['error_message']}")
                        
                        if error_info['duplicate_pos_tid']:
                            print(f"\n⚠️ 發現重複的 pos_tid: {len(error_info['duplicate_pos_tid'])} 筆")
                            print("前5筆重複值:", error_info['duplicate_pos_tid'][:5])
                        
                        if error_info['null_counts']:
                            print("\n📊 空值統計:")
                            for col, count in error_info['null_counts'].items():
                                if count > 0:
                                    print(f"  - {col}: {count} 筆空值")
                        
                        print("\n📝 資料型別檢查:")
                        for col, dtype in error_info['data_types'].items():
                            print(f"  - {col}: {dtype}")
                        
                        print("\n🔬 資料樣本（前5筆）:")
                        for idx, row in enumerate(error_info['sample_data'][:5]):
                            print(f"\n第 {idx+1} 筆資料:")
                            for key, value in row.items():
                                print(f"  {key}: {value}")
                        
                        raise  # 重新拋出錯誤以中斷當前區塊的處理
            
            total_rows_processed += len(chunk)
            chunk_time_taken = time.time() - chunk_start_time
            print(f"====第 {i+1} 個區塊 ({len(chunk)} 筆資料) 已處理並插入，耗時 {chunk_time_taken:.2f} 秒。====")
            #print(f"目前已處理總資料筆數: {total_rows_processed}")

        except Exception as e:
            print(f"⚠️ 第 {i+1} 個區塊插入失敗: {e}")
            # 記錄錯誤的區塊
            error_rows.extend(chunk.index.tolist())
            continue

    # 處理完成後輸出錯誤統計
    if error_details:
        print("\n📊 錯誤統計摘要：")
        print(f"總共有 {len(error_rows)} 筆資料插入失敗")
        print(f"發生錯誤的區塊數：{len(error_details)}")
        
        # 分析最常見的錯誤類型
        error_types = {}
        for detail in error_details:
            error_type = detail['error_info']['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("\n🔍 錯誤類型分布：")
        for error_type, count in error_types.items():
            print(f"  - {error_type}: {count} 個區塊")
        
        print("\n💡 建議解決方案：")
        if any('duplicate' in str(detail['error_info']['error_message']).lower() for detail in error_details):
            print("1. 檢查並移除重複的 pos_tid")
        if any('null' in str(detail['error_info']['error_message']).lower() for detail in error_details):
            print("2. 檢查必填欄位的空值")
        if any('type' in str(detail['error_info']['error_message']).lower() for detail in error_details):
            print("3. 檢查資料型別是否符合資料表定義")
    
    end_time = time.time()
    print(f"\n✅ 成功匯入 {total_rows_processed} 筆資料到 '{TABLE_NAME}'")
    print(f"⏱️ 總耗時: {(end_time - start_time):.2f} 秒")

except Exception as e:
    print(f"發生未預期的錯誤: {e}")
    sys.exit(1)