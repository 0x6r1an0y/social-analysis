from sqlalchemy import create_engine, text
import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# check 資料庫存在
def create_db_if_not_exists(dbname):
    conn = psycopg2.connect(dbname='postgres', user='postgres', password='00000000', host='localhost')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f'CREATE DATABASE {dbname}')
        print(f'✅ 資料庫 {dbname} 已建立')
    else:
        print(f'✅ 資料庫 {dbname} 已存在')
    cur.close()
    conn.close()

create_db_if_not_exists('labeling_db')

# 來源資料庫（原始貼文）
source_engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis")

# 目標資料庫（標記用 DB）
target_engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db")

# 1. 撈出 4000 筆疑似詐騙貼文
keywords = ['穩賺', '保證獲利', '老師帶你賺', '量化交易', '虛擬貨幣']
where_clause = " OR ".join([f"content ILIKE '%{kw}%'" for kw in keywords])
sql = f"SELECT pos_tid, content FROM posts WHERE {where_clause} LIMIT 4000"

df = pd.read_sql_query(text(sql), source_engine)

df["group_id"] = pd.Series(range(len(df))) % 5  # 輪流分配 group_id 為 0~4

# 2. 存到另一個 DB 的新表格中
df.to_sql("candidates", target_engine, if_exists="replace", index=False)

print("✅ 已複製 4000 筆資料到 labeling_db.candidates")





