from sqlalchemy import create_engine, inspect
import pandas as pd

# 設定資料庫連線
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
engine = create_engine(DB_URL)

def list_all_tables():
    """
    回傳該資料庫下的所有表名
    """
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        print("資料庫中的所有資料表：")
        for name in table_names:
            print(f" - {name}")
        return table_names
    except Exception as e:
        print(f"無法取得資料表列表: {e}")
        return []

def get_table_headers(table_name):
    """
    取得指定資料表的欄位名稱
    """
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        headers = [col["name"] for col in columns]
        print(f"{table_name} 的欄位名稱：{headers}")
        return headers
    except Exception as e:
        print(f"取得欄位名稱失敗: {e}")
        return []

# 測試執行
if __name__ == "__main__":
    tables = list_all_tables()
    for table in tables:
        get_table_headers(table)
