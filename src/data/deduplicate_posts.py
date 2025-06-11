from sqlalchemy import create_engine, text
import time
import pandas as pd
from datetime import datetime

def deduplicate_posts():
    # 資料庫連線資訊
    db_name = "social_media_analysis_hash"
    username = "postgres"
    password = "00000000"
    host = "localhost"
    port = "5432"
    
    # 新表名稱
    new_table_name = "posts_deduplicated"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 建立連線字串
    conn_str = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
    
    try:
        # 建立資料庫引擎
        print("正在建立資料庫連線...")
        engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")
        
        print("開始處理重複貼文...")
        start_time = time.time()
        
        with engine.connect() as conn:
            # 先測試連線並檢查原始表
            print("檢查原始 posts 表...")
            try:
                posts_count = conn.execute(text("SELECT COUNT(*) FROM posts")).fetchone()[0]
                print(f"原始 posts 表中有 {posts_count:,} 筆資料")
                
                if posts_count == 0:
                    print("錯誤：posts 表為空，無法進行去重操作")
                    return
                    
                # 檢查有多少筆資料有 content_hash
                hash_count = conn.execute(text("""
                    SELECT COUNT(*) FROM posts 
                    WHERE content_hash IS NOT NULL AND content_hash != ''
                """)).fetchone()[0]
                print(f"其中有 content_hash 的資料: {hash_count:,} 筆")
                
                if hash_count == 0:
                    print("錯誤：沒有任何資料有 content_hash，無法進行去重")
                    return
                    
            except Exception as e:
                print(f"檢查原始表時發生錯誤: {e}")
                print("可能原因：posts 表不存在或無權限存取")
                return
            
            # 1. 首先檢查新表是否已存在
            print(f"檢查目標表 {new_table_name} 是否存在...")
            check_table = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{new_table_name}'
                );
            """)).fetchone()[0]
            
            if check_table:
                print(f"警告：表 {new_table_name} 已存在！")
                drop_choice = input("是否要刪除現有的表並重新創建？(y/n): ").strip().lower()
                if drop_choice not in ['y', 'yes', '是']:
                    print("操作已取消")
                    return
                print(f"刪除現有表 {new_table_name}...")
                conn.execute(text(f"DROP TABLE {new_table_name}"))
                print("現有表已刪除")
            
            # 2. 創建新表（使用與原始表相同的結構）
            print("正在創建新表...")
            try:
                create_table_sql = f"""
                    CREATE TABLE {new_table_name} AS
                    WITH deduplicated AS (
                        SELECT DISTINCT ON (content_hash) *
                        FROM posts
                        WHERE content_hash IS NOT NULL 
                        AND content_hash != ''
                        ORDER BY content_hash, created_time DESC
                    )
                    SELECT * FROM deduplicated;
                """
                print("執行的 SQL:")
                print(create_table_sql)
                
                result = conn.execute(text(create_table_sql))
                print("表創建成功！")
                
                # 立即檢查表是否真的被創建了
                verify_table = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{new_table_name}'
                    );
                """)).fetchone()[0]
                
                if verify_table:
                    print(f"✅ 確認：表 {new_table_name} 已成功創建")
                else:
                    print(f"❌ 錯誤：表 {new_table_name} 創建失敗")
                    return
                    
            except Exception as e:
                print(f"創建表時發生錯誤: {e}")
                print("可能原因：SQL 語法錯誤或權限不足")
                return
            
            # 3. 添加索引以提升查詢效能
            print("建立索引...")
            try:
                conn.execute(text(f"""
                    CREATE INDEX idx_{new_table_name}_content_hash 
                    ON {new_table_name} (content_hash);
                """))
                print("content_hash 索引創建完成")
                
                conn.execute(text(f"""
                    CREATE INDEX idx_{new_table_name}_created_time 
                    ON {new_table_name} (created_time);
                """))
                print("created_time 索引創建完成")
                
            except Exception as e:
                print(f"創建索引時發生錯誤: {e}")
                print("警告：索引創建失敗，但不影響主要功能")
            
            # 4. 統計資訊
            print("計算統計資訊...")
            try:
                stats = conn.execute(text(f"""
                    SELECT 
                        (SELECT COUNT(*) FROM posts) as original_count,
                        (SELECT COUNT(*) FROM {new_table_name}) as deduplicated_count
                """)).fetchone()
                
                original_count = stats[0]
                deduplicated_count = stats[1]
                removed_count = original_count - deduplicated_count
                
                print("\n=== 去重結果統計 ===")
                print(f"原始貼文數量: {original_count:,}")
                print(f"去重後貼文數量: {deduplicated_count:,}")
                print(f"移除的重複貼文: {removed_count:,}")
                print(f"重複率: {(removed_count/original_count)*100:.2f}%")
                
            except Exception as e:
                print(f"計算統計資訊時發生錯誤: {e}")
            
            # 5. 匯出詳細報告
            print("生成詳細報告...")
            try:
                report_query = f"""
                SELECT 
                    p.content_hash,
                    COUNT(*) as original_count,
                    COUNT(DISTINCT p.content_hash) as unique_count,
                    MIN(p.created_time) as first_post_time,
                    MAX(p.created_time) as last_post_time,
                    STRING_AGG(DISTINCT p.page_name, '; ') as pages,
                    LEFT(MIN(p.content), 200) as content_sample
                FROM posts p
                WHERE p.content_hash IN (
                    SELECT content_hash 
                    FROM posts 
                    WHERE content_hash IS NOT NULL 
                    AND content_hash != ''
                    GROUP BY content_hash 
                    HAVING COUNT(*) > 1
                )
                GROUP BY p.content_hash
                ORDER BY original_count DESC
                """
                
                df = pd.read_sql_query(report_query, engine)
                report_filename = f"deduplication_report_{timestamp}.csv"
                df.to_csv(report_filename, index=False, encoding='utf-8-sig')
                
                print(f"詳細報告已匯出到: {report_filename}")
                
            except Exception as e:
                print(f"生成報告時發生錯誤: {e}")
                print("警告：報告生成失敗，但主要去重操作已完成")
            
            # 最終確認
            print(f"\n最終確認表 {new_table_name} 狀態...")
            final_check = conn.execute(text(f"""
                SELECT 
                    table_name,
                    (SELECT COUNT(*) FROM {new_table_name}) as row_count
                FROM information_schema.tables 
                WHERE table_name = '{new_table_name}'
            """)).fetchone()
            
            if final_check:
                print(f"✅ 表 {new_table_name} 存在，包含 {final_check[1]:,} 筆資料")
            else:
                print(f"❌ 表 {new_table_name} 不存在！")
            
        total_time = time.time() - start_time
        print(f"\n處理完成！總耗時: {total_time:.2f} 秒")
        
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        print("詳細錯誤資訊:")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    deduplicate_posts() 