from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis")

with engine.connect() as conn:
    result = conn.execute(text("SELECT date FROM posts ORDER BY date DESC LIMIT 1"))
    last_date = result.scalar()

print(f"資料庫中最後一筆資料的日期是：{last_date}")