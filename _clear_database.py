import psycopg2
import sys

# 資料庫連接參數
DB_USER = "postgres"        # 資料庫使用者名稱
DB_PASSWORD = "00000000"    # 資料庫密碼
DB_HOST = "localhost"       # 資料庫主機
DB_PORT = "5432"            # 資料庫埠號
DB_NAME = "labeling_db" # 資料庫名稱

def clear_database():
    try:
        # 連接到資料庫
        print(f"正在連接到資料庫 '{DB_NAME}'...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 獲取所有資料表名稱
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        
        if not tables:
            print("資料庫中沒有資料表。")
            return
        
        # 清空每個資料表
        for table in tables:
            table_name = table[0]
            print(f"正在清空資料表 '{table_name}'...")
            cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE")
            print(f"資料表 '{table_name}' 已清空。")
        
        print("\n所有資料表已清空。")
        
    except psycopg2.OperationalError as e:
        print(f"連接資料庫時發生錯誤: {e}")
        print("\n可能的解決方案:")
        print("1. 確認 PostgreSQL 服務是否正在運行")
        print("2. 確認資料庫使用者名稱和密碼是否正確")
        sys.exit(1)
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        sys.exit(1)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
            print("資料庫連接已關閉。")

def drop_database():
    try:
        # 連接到預設的 postgres 資料庫
        print("正在連接到 PostgreSQL 伺服器...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database="postgres"  # 連接到預設資料庫
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 檢查資料庫是否存在
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"資料庫 '{DB_NAME}' 不存在。")
            return
        
        # 關閉所有到目標資料庫的連接
        print(f"正在關閉所有到資料庫 '{DB_NAME}' 的連接...")
        cursor.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{DB_NAME}'
            AND pid <> pg_backend_pid()
        """)
        
        # 刪除資料庫
        print(f"正在刪除資料庫 '{DB_NAME}'...")
        cursor.execute(f"DROP DATABASE {DB_NAME}")
        print(f"資料庫 '{DB_NAME}' 已刪除。")
        
    except psycopg2.OperationalError as e:
        print(f"連接資料庫時發生錯誤: {e}")
        print("\n可能的解決方案:")
        print("1. 確認 PostgreSQL 服務是否正在運行")
        print("2. 確認資料庫使用者名稱和密碼是否正確")
        sys.exit(1)
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        sys.exit(1)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
            print("資料庫連接已關閉。")

if __name__ == "__main__":
    print("=== 資料庫清理工具 ===")
    print("1. 清空所有資料表（保留資料庫結構）")
    print("2. 完全刪除資料庫（包括所有資料和結構）")
    
    choice = input("\n請選擇操作 (1/2): ").strip()
    
    if choice == "1":
        clear_database()
    elif choice == "2":
        confirm = input(f"警告：此操作將完全刪除資料庫 '{DB_NAME}' 及其所有資料！確定要繼續嗎？(y/N): ").strip().lower()
        if confirm == 'y':
            drop_database()
        else:
            print("操作已取消。")
    else:
        print("無效的選擇。") 