from sqlalchemy import create_engine, text
import time
import pandas as pd

def check_duplicate_posts():
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
        
        print("開始檢查重複的貼文內容...")
        start_time = time.time()
        
        with engine.connect() as conn:
            # 檢查 content_hash 欄位是否存在且有資料
            print("檢查 content_hash 欄位狀態...")
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_posts,
                    COUNT(content_hash) as posts_with_hash,
                    COUNT(CASE WHEN content_hash IS NOT NULL AND content_hash != '' THEN 1 END) as valid_hashes
                FROM posts
            """)).fetchone()
            
            total_posts = result[0]
            posts_with_hash = result[1]
            valid_hashes = result[2]
            
            print(f"總貼文數量: {total_posts:,}")
            print(f"有雜湊值的貼文: {posts_with_hash:,}")
            print(f"有效雜湊值的貼文: {valid_hashes:,}")
            
            if valid_hashes == 0:
                print("錯誤：沒有找到有效的雜湊值！請先執行 add_content_hash.py")
                return
            
            # 檢查重複的雜湊值
            print("\n正在分析重複的貼文...")
            duplicate_analysis = conn.execute(text("""
                SELECT 
                    content_hash,
                    COUNT(*) as duplicate_count
                FROM posts 
                WHERE content_hash IS NOT NULL 
                AND content_hash != ''
                GROUP BY content_hash
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
            """)).fetchall()
            
            if not duplicate_analysis:
                print("沒有發現重複的貼文內容！")
                return
            
            # 計算統計資訊
            unique_duplicate_groups = len(duplicate_analysis)  # 有多少組重複的貼文
            total_duplicate_posts = sum(row[1] for row in duplicate_analysis)  # 總共有多少篇重複的貼文
            
            print(f"\n=== 重複貼文統計結果 ===")
            print(f"重複的貼文組數: {unique_duplicate_groups:,} 組")
            print(f"總重複貼文數量: {total_duplicate_posts:,} 篇")
            print(f"重複率: {(total_duplicate_posts/valid_hashes)*100:.2f}%")
            
            # 顯示前10組重複最多的內容
            print(f"\n=== 重複最多的前10組貼文 ===")
            for i, (content_hash, count) in enumerate(duplicate_analysis[:10], 1):
                print(f"{i:2d}. 雜湊值: {content_hash[:16]}... 重複次數: {count:,}")
            
            # 統計重複次數的分布
            print(f"\n=== 重複次數分布 ===")
            duplicate_counts = [row[1] for row in duplicate_analysis]
            
            # 計算各種重複次數的統計
            distribution = {}
            for count in duplicate_counts:
                distribution[count] = distribution.get(count, 0) + 1
            
            # 按重複次數排序顯示
            for dup_count in sorted(distribution.keys()):
                group_count = distribution[dup_count]
                total_posts_in_groups = dup_count * group_count
                print(f"重複 {dup_count:2d} 次: {group_count:,} 組 (共 {total_posts_in_groups:,} 篇貼文)")
            
            # 詳細分析選項
            show_details = input("\n是否要查看重複貼文的詳細內容？(y/n): ").strip().lower()
            if show_details in ['y', 'yes', '是']:
                show_detail_analysis(conn, duplicate_analysis[:5])  # 只顯示前5組
        
        total_time = time.time() - start_time
        print(f"\n分析完成！總耗時: {total_time:.2f} 秒")

    except Exception as e:
        print(f"發生錯誤: {e}")
        raise

def show_detail_analysis(conn, top_duplicates):
    """顯示重複貼文的詳細內容"""
    print(f"\n=== 重複貼文詳細內容 (前5組) ===")
    
    for i, (content_hash, count) in enumerate(top_duplicates, 1):
        print(f"\n--- 第 {i} 組 (重複 {count} 次) ---")
        
        # 獲取這組重複貼文的詳細資訊
        details = conn.execute(text("""
            SELECT 
                pos_tid, 
                page_name, 
                created_time,
                LEFT(content, 100) as content_preview
            FROM posts 
            WHERE content_hash = :hash 
            ORDER BY created_time
            LIMIT 3
        """), {"hash": content_hash}).fetchall()
        
        for j, (pos_tid, page_name, created_time, content_preview) in enumerate(details, 1):
            print(f"  {j}. ID: {pos_tid}")
            print(f"     頁面: {page_name}")
            print(f"     時間: {created_time}")
            print(f"     內容預覽: {content_preview}...")
            print()
        
        if count > 3:
            print(f"  ... 還有 {count - 3} 篇相同內容的貼文")

def export_duplicate_report():
    """匯出重複貼文報告到CSV檔案"""
    db_name = "social_media_analysis_hash"
    username = "postgres"
    password = "00000000"
    host = "localhost"
    port = "5432"
    
    conn_str = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
    
    try:
        engine = create_engine(conn_str)
        
        print("正在匯出重複貼文報告...")
        
        # 查詢重複的貼文
        query = """
        SELECT 
            p1.content_hash,
            COUNT(*) as duplicate_count,
            MIN(p1.created_time) as first_post_time,
            MAX(p1.created_time) as last_post_time,
            STRING_AGG(DISTINCT p1.page_name, '; ') as pages,
            LEFT(MIN(p1.content), 200) as content_sample
        FROM posts p1
        WHERE p1.content_hash IN (
            SELECT content_hash 
            FROM posts 
            WHERE content_hash IS NOT NULL AND content_hash != ''
            GROUP BY content_hash 
            HAVING COUNT(*) > 1
        )
        GROUP BY p1.content_hash
        ORDER BY duplicate_count DESC
        """
        
        df = pd.read_sql_query(query, engine)
        
        # 匯出到CSV
        filename = f"duplicate_posts_report_{int(time.time())}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"報告已匯出到: {filename}")
        print(f"共匯出 {len(df)} 組重複貼文的詳細資訊")
        
    except Exception as e:
        print(f"匯出報告時發生錯誤: {e}")

if __name__ == "__main__":
    check_duplicate_posts()
    
    # 詢問是否要匯出詳細報告
    export_choice = input("\n是否要匯出重複貼文的詳細報告到CSV檔案？(y/n): ").strip().lower()
    if export_choice in ['y', 'yes', '是']:
        export_duplicate_report() 