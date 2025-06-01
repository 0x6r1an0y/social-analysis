from sqlalchemy import create_engine, text

# 設定資料庫連線
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis"
engine = create_engine(DB_URL)
table_name = "posts"  # 替換成你的資料表名稱

# 查詢特定資料表的行數
def get_table_row_count(table_name):
    with engine.connect() as conn:
        query = text(f"SELECT COUNT(*) FROM {table_name}")
        result = conn.execute(query)
        row_count = result.scalar()
    return row_count

# 使用範例
row_count = get_table_row_count(table_name)
print(f"資料表 {table_name} 共有 {row_count} 行")