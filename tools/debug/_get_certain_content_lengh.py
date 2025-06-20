from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash")
table = "posts_deduplicated"

with engine.connect() as conn:    
    # 查詢 content 長度小於 20 的資料筆數
    short_content_count = conn.execute(
        text(f"SELECT COUNT(*) FROM {table} WHERE LENGTH(content) < 20")
    ).scalar()

print(f"content 欄位長度小於 20 的資料筆數：{short_content_count}")