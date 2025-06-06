from sqlalchemy import create_engine, text
import pandas as pd

# 設定資料庫連線
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
engine = create_engine(DB_URL)

def get_data_by_pos_tid(pos_tid_value, table_name="your_table_name"):
    """
    根據 pos_tid 查詢整行資料
    
    Args:
        pos_tid_value: 要查詢的 pos_tid 值
        table_name: 資料表名稱（請替換成實際的表名）
    
    Returns:
        查詢結果
    """
    
    # 方法1: 使用 text() 和參數化查詢（推薦）
    query = text("""
        SELECT pos_tid, content, group_id, label, note 
        FROM :table_name 
        WHERE pos_tid = :pos_tid_val
    """)
    
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(f"SELECT pos_tid, content, group_id, label, note FROM {table_name} WHERE pos_tid = :pos_tid_val"),
                {"pos_tid_val": pos_tid_value}
            )
            
            # 獲取所有匹配的行
            rows = result.fetchall()
            
            if rows:
                # 轉換為字典格式方便查看
                columns = ['pos_tid', 'content', 'group_id', 'label', 'note']
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
    query = f"""
        SELECT pos_tid, content, group_id, label, note 
        FROM {table_name} 
        WHERE pos_tid = %(pos_tid)s
    """
    
    try:
        df = pd.read_sql_query(query, engine, params={"pos_tid": pos_tid_value})
        return df
    except Exception as e:
        print(f"查詢錯誤: {e}")
        return None

# 使用範例
if __name__ == "__main__":
    # 替換成你的實際表名和要查詢的 pos_tid
    table_name = "candidates"  # 請替換成實際的表名
    target_pos_tid = "763099540467049_3512500108860298"  # 請替換成要查詢的 pos_tid 值
    
    # 方法1: 使用字典格式
    print("=== 使用字典格式查詢 ===")
    result = get_data_by_pos_tid(target_pos_tid, table_name)
    if result:
        for row in result:
            print(f"pos_tid: {row['pos_tid']}")
            print(f"content: {row['content']}")
            print(f"group_id: {row['group_id']}")
            print(f"label: {row['label']}")
            print(f"note: {row['note']}")
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
            print(f"pos_tid: {first_row['pos_tid']}")
            print(f"content: {first_row['content']}")
            print(f"group_id: {first_row['group_id']}")
            print(f"label: {first_row['label']}")
            print(f"note: {first_row['note']}")

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