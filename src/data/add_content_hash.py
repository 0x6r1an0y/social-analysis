from sqlalchemy import create_engine, text
import time

def add_content_hash():
    # 資料庫連線資訊
    db_name = "social_media_analysis_hash"
    username = "postgres"
    password = "00000000"
    host = "localhost"
    port = "5432"

    # 建立連線字串
    conn_str = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
    
    try:
        # 建立資料庫引擎
        engine = create_engine(conn_str)
        
        print("開始處理 posts 表格的雜湊值...")
        start_time = time.time()
        
        with engine.connect() as conn:
            # 啟用 pgcrypto 擴充功能
            print("啟用 pgcrypto 擴充功能...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
            conn.commit()
            print("pgcrypto 擴充功能已啟用")
            
            # 檢查 content_hash 欄位是否已存在
            result = conn.execute(text("""
                SELECT column_name, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'posts' 
                AND column_name = 'content_hash'
            """))
            
            column_info = result.fetchone()
            
            if column_info is None:
                print("添加 content_hash 欄位...")
                # 添加 content_hash 欄位
                conn.execute(text("ALTER TABLE posts ADD COLUMN content_hash VARCHAR(64)"))
                print("content_hash 欄位添加完成")
            else:
                current_length = column_info[1]
                if current_length < 64:
                    print(f"修改 content_hash 欄位長度從 {current_length} 到 64...")
                    conn.execute(text("ALTER TABLE posts ALTER COLUMN content_hash TYPE VARCHAR(64)"))
                    print("content_hash 欄位長度修改完成")
                else:
                    print("content_hash 欄位已存在，將更新雜湊值...")
            
            # 獲取需要處理的總行數
            total_rows = conn.execute(text("""
                SELECT COUNT(*) FROM posts 
                WHERE content IS NOT NULL 
                AND (content_hash IS NULL OR content_hash = '')
            """)).scalar()
            print(f"總共有 {total_rows} 筆資料需要處理")
            
            if total_rows == 0:
                print("沒有需要處理的資料")
                return
            
            # 使用分批處理來更新雜湊值
            batch_size = 100000  # 每批處理 10 萬筆
            processed_count = 0
            
            while True:
                # 更新這批資料的雜湊值
                result = conn.execute(text(f"""
                    UPDATE posts 
                    SET content_hash = encode(
                        digest(
                            convert_to(content, 'UTF8'),
                            'sha256'::text
                        ),
                        'hex'
                    )
                    WHERE content IS NOT NULL
                    AND (content_hash IS NULL OR content_hash = '')
                    AND pos_tid IN (
                        SELECT pos_tid 
                        FROM posts 
                        WHERE content IS NOT NULL
                        AND (content_hash IS NULL OR content_hash = '')
                        ORDER BY pos_tid 
                        LIMIT {batch_size}
                    )
                """))
                
                # 獲取這批處理的實際行數
                affected_rows = result.rowcount
                
                if affected_rows == 0:
                    break
                    
                processed_count += affected_rows
                progress = min(100, (processed_count / total_rows) * 100)
                elapsed_time = time.time() - start_time
                print(f"處理進度: {progress:.1f}% (已處理 {processed_count}/{total_rows} 筆，耗時 {elapsed_time:.1f} 秒)")
                
                # 提交事務
                conn.commit()
        
        total_time = time.time() - start_time
        print(f"\n雜湊值計算完成！")
        print(f"總耗時: {total_time:.1f} 秒")
        if total_rows > 0:
            print(f"平均處理速度: {total_rows/total_time:.1f} 筆/秒")

    except Exception as e:
        print(f"發生錯誤: {e}")
        raise

if __name__ == "__main__":
    add_content_hash() 