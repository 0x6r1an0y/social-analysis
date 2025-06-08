from sqlalchemy import create_engine, text, inspect
import pandas as pd

# 設定資料庫連線
#DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis"
engine = create_engine(DB_URL)

def get_table_columns(table_name):
    """
    取得指定資料表的所有欄位名稱
    """
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        return [col["name"] for col in columns]
    except Exception as e:
        print(f"取得欄位名稱失敗: {e}")
        return []

def get_data_by_pos_tid(pos_tid_value, table_name="your_table_name"):
    """
    根據 pos_tid 查詢整行資料
    
    Args:
        pos_tid_value: 要查詢的 pos_tid 值
        table_name: 資料表名稱
    
    Returns:
        查詢結果
    """
    try:
        # 獲取表格的所有欄位
        columns = get_table_columns(table_name)
        if not columns:
            print(f"無法獲取表格 {table_name} 的欄位資訊")
            return None
            
        # 動態建立 SELECT 語句
        columns_str = ", ".join(columns)
        query = f"SELECT {columns_str} FROM {table_name} WHERE pos_tid = :pos_tid_val"
        
        with engine.connect() as connection:
            result = connection.execute(
                text(query),
                {"pos_tid_val": pos_tid_value}
            )
            
            # 獲取所有匹配的行
            rows = result.fetchall()
            
            if rows:
                # 轉換為字典格式方便查看
                data = []
                for row in rows:
                    data.append(dict(zip(columns, row)))
                return data
            else:
                print(f"沒有找到 pos_tid = {pos_tid_value} 的資料")
                return None
                
    except Exception as e:
        print(f"查詢錯誤: {e}")
        return None

def get_data_by_pos_tid_pandas(pos_tid_value, table_name="your_table_name"):
    """
    使用 pandas 查詢（如果你喜歡 DataFrame 格式）
    """
    try:
        # 獲取表格的所有欄位
        columns = get_table_columns(table_name)
        if not columns:
            print(f"無法獲取表格 {table_name} 的欄位資訊")
            return None
            
        # 動態建立 SELECT 語句
        columns_str = ", ".join(columns)
        query = f"SELECT {columns_str} FROM {table_name} WHERE pos_tid = %(pos_tid)s"
        
        df = pd.read_sql_query(query, engine, params={"pos_tid": pos_tid_value})
        return df
    except Exception as e:
        print(f"查詢錯誤: {e}")
        return None

# 使用範例
if __name__ == "__main__":
    table_name = "posts"  # 請替換成實際的表名
    target_pos_tid = input("請輸入要查詢的 pos_tid: ").strip()
    
    # 方法1: 使用字典格式
    print("\n=== 使用字典格式查詢 ===")
    result = get_data_by_pos_tid(target_pos_tid, table_name)
    if result:
        for row in result:
            for column, value in row.items():
                print(f"{column}: {value}")
            print("-" * 50)
    
    # 方法2: 使用 pandas DataFrame
    print("\n=== 使用 pandas DataFrame 查詢 ===")
    df_result = get_data_by_pos_tid_pandas(target_pos_tid, table_name)
    if df_result is not None and not df_result.empty:
        print(df_result)
        
        # 如果只想要第一行資料
        if len(df_result) > 0:
            first_row = df_result.iloc[0]
            print(f"\n第一筆資料:")
            for column in df_result.columns:
                print(f"{column}: {first_row[column]}")

# 簡化版本 - 直接查詢
def simple_query(pos_tid_value, table_name="your_table_name"):
    """簡化版本的查詢函數"""
    with engine.connect() as connection:
        result = connection.execute(
            text(f"SELECT * FROM {table_name} WHERE pos_tid = :pos_tid"),
            {"pos_tid": pos_tid_value}
        )
        return result.fetchone()  # 返回第一筆匹配的資料

# 使用簡化版本
# row = simple_query("your_pos_tid_value", "your_table_name")
# if row:
#     print(f"查詢結果: {row}")