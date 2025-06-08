import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import gc  # 用於手動釋放記憶體

def check_database_exists(conn, db_name):
    """檢查資料庫是否存在"""
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
    exists = cur.fetchone() is not None
    cur.close()
    return exists

def copy_table_in_chunks(source_engine, target_engine, table_name, chunk_size=10000):
    """分批複製表格數據"""
    try:
        # 獲取表格結構
        inspector = inspect(source_engine)
        columns = inspector.get_columns(table_name)
        
        # 建立表格
        create_table_sql = f"CREATE TABLE {table_name} ("
        column_defs = []
        for column in columns:
            col_def = f"{column['name']} {column['type']}"
            if not column['nullable']:
                col_def += " NOT NULL"
            column_defs.append(col_def)
        create_table_sql += ", ".join(column_defs) + ")"
        
        with target_engine.connect() as conn:
            conn.execute(text(create_table_sql))
        
        # 獲取總行數
        with source_engine.connect() as conn:
            total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        
        print(f"表格 {table_name} 共有 {total_rows} 行數據")
        
        # 分批讀取和寫入
        offset = 0
        while True:
            # 使用 LIMIT 和 OFFSET 分批讀取
            query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
            chunk_df = pd.read_sql_query(query, source_engine)
            
            if chunk_df.empty:
                break
                
            # 寫入這批數據
            chunk_df.to_sql(table_name, target_engine, if_exists='append', index=False)
            
            # 更新進度
            offset += chunk_size
            progress = min(100, (offset / total_rows) * 100)
            print(f"表格 {table_name} 複製進度: {progress:.1f}%")
            
            # 清理記憶體
            del chunk_df
            gc.collect()
            
    except Exception as e:
        print(f"複製表格 {table_name} 時發生錯誤: {e}")
        raise

def copy_database():
    # 資料庫連線資訊
    source_db = "social_media_analysis"
    target_db = "social_media_analysis_hash"
    username = "postgres"
    password = "00000000"
    host = "localhost"
    port = "5432"

    # 建立連線字串
    source_conn_str = f"postgresql://{username}:{password}@{host}:{port}/{source_db}"
    target_conn_str = f"postgresql://{username}:{password}@{host}:{port}/{target_db}"
    
    try:
        # 連接到 postgres 資料庫
        conn = psycopg2.connect(
            dbname="postgres",
            user=username,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # 檢查目標資料庫是否存在
        if check_database_exists(conn, target_db):
            while True:
                response = input(f"資料庫 '{target_db}' 已存在。是否要清空並重新複製？(y/n): ").strip().lower()
                if response in ['y', 'yes', '是']:
                    # 關閉所有連接到目標資料庫的連線
                    cur = conn.cursor()
                    cur.execute(f"""
                        SELECT pg_terminate_backend(pg_stat_activity.pid)
                        FROM pg_stat_activity
                        WHERE pg_stat_activity.datname = '{target_db}'
                        AND pid <> pg_backend_pid()
                    """)
                    # 刪除現有資料庫
                    cur.execute(f"DROP DATABASE {target_db}")
                    print(f"已刪除現有的資料庫 {target_db}")
                    break
                elif response in ['n', 'no', '否']:
                    print("操作已取消")
                    return
                else:
                    print("請輸入 'y' 或 'n'")
        
        # 創建新資料庫
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE {target_db}")
        print(f"已創建新資料庫 {target_db}")
        
        cur.close()
        conn.close()

        # 建立源資料庫和目標資料庫的 SQLAlchemy 引擎
        source_engine = create_engine(source_conn_str)
        target_engine = create_engine(target_conn_str)

        # 獲取所有表格名稱
        inspector = inspect(source_engine)
        tables = inspector.get_table_names()
        
        print("開始複製資料...")
        # 複製每個表格
        for table in tables:
            print(f"\n開始複製表格: {table}")
            copy_table_in_chunks(source_engine, target_engine, table)
            print(f"表格 {table} 複製完成")
            
            # 在每個表格複製完成後清理記憶體
            gc.collect()

        print("\n資料庫複製完成！")
        print(f"源資料庫: {source_db}")
        print(f"目標資料庫: {target_db}")

    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    copy_database() 