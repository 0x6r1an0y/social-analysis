from sqlalchemy import create_engine, text

def preview_short_content_posts(max_length=30):
    """
    預覽內容少於指定字數的貼文
    
    Args:
        max_length: 最大內容長度（預設30個字）
    """
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print(f"預覽內容少於 {max_length} 個字的貼文...")
        
        with engine.connect() as conn:
            # 查詢要刪除的記錄統計
            query = text("""
                SELECT page_name, COUNT(*) as post_count
                FROM posts_deduplicated 
                WHERE LENGTH(content) < :max_length
                AND content IS NOT NULL
                GROUP BY page_name 
                ORDER BY post_count DESC
            """)
            
            result = conn.execute(query, {"max_length": max_length})
            page_stats = [(row[0], row[1]) for row in result]
            
            if not page_stats:
                print(f"沒有找到內容少於 {max_length} 個字的貼文")
                return False
            
            # 查詢總數
            total_query = text("""
                SELECT COUNT(*) as total_count
                FROM posts_deduplicated 
                WHERE LENGTH(content) < :max_length
                AND content IS NOT NULL
            """)
            
            total_result = conn.execute(total_query, {"max_length": max_length})
            total_count = total_result.fetchone()[0]
            
            print(f"找到 {total_count} 筆內容少於 {max_length} 個字的貼文")
            print(f"\n各 page_name 的貼文數量：")
            print("-" * 60)
            
            for i, (page_name, count) in enumerate(page_stats, 1):
                print(f"{i:3d}. {page_name:<30} | {count:>8} 篇貼文")
            
            print("-" * 60)
            print(f"總計：{len(page_stats)} 個不同的 page_name，{total_count} 篇貼文")
            
            # 顯示一些樣本內容
            sample_query = text("""
                SELECT pos_tid, page_name, content, LENGTH(content) as content_length
                FROM posts_deduplicated 
                WHERE LENGTH(content) < :max_length
                AND content IS NOT NULL
                ORDER BY LENGTH(content) ASC
                LIMIT 10
            """)
            
            sample_result = conn.execute(sample_query, {"max_length": max_length})
            samples = sample_result.fetchall()
            
            print(f"\n內容樣本（前10筆）：")
            print("-" * 80)
            for i, (pos_tid, page_name, content, length) in enumerate(samples, 1):
                print(f"{i:2d}. ID: {pos_tid}")
                print(f"    頁面: {page_name}")
                print(f"    長度: {length} 字")
                print(f"    內容: {content}")
                print()
            
            return True
            
    except Exception as e:
        print(f"預覽過程中發生錯誤：{str(e)}")
        return False

def delete_short_content_posts(max_length=30):
    """
    刪除內容少於指定字數的貼文
    
    Args:
        max_length: 最大內容長度（預設30個字）
    """
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print(f"正在刪除內容少於 {max_length} 個字的貼文...")
        
        with engine.connect() as conn:
            # 1. 先查詢要刪除的 pos_tid
            query = text("""
                SELECT pos_tid, page_name, content, LENGTH(content) as content_length
                FROM posts_deduplicated 
                WHERE LENGTH(content) < :max_length
                AND content IS NOT NULL
            """)
            
            result = conn.execute(query, {"max_length": max_length})
            to_delete = [(row[0], row[1], row[2], row[3]) for row in result]
            
            if not to_delete:
                print(f"沒有找到內容少於 {max_length} 個字的貼文")
                return
                
            print(f"找到 {len(to_delete)} 筆內容少於 {max_length} 個字的貼文將被刪除")
            
            # 顯示要刪除的 page_name 統計
            page_stats = {}
            for _, page_name, _, _ in to_delete:
                if page_name not in page_stats:
                    page_stats[page_name] = 0
                page_stats[page_name] += 1
            
            print(f"\n要刪除的 page_name 統計：")
            print("-" * 60)
            for page_name, count in sorted(page_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"{page_name:<30} | {count:>8} 篇貼文")
            print("-" * 60)
            
            # 顯示內容長度分布
            length_stats = {}
            for _, _, _, length in to_delete:
                if length not in length_stats:
                    length_stats[length] = 0
                length_stats[length] += 1
            
            print(f"\n內容長度分布：")
            print("-" * 40)
            for length in sorted(length_stats.keys()):
                print(f"{length:2d} 字: {length_stats[length]:>6} 篇")
            print("-" * 40)
            
            # 2. 刪除資料庫記錄
            print("刪除資料庫記錄...")
            
            # 使用 begin() 確保交易正確提交
            with engine.begin() as transaction:
                placeholders = ','.join([f"'{tid}'" for tid in [item[0] for item in to_delete]])
                delete_query = text(f"""
                    DELETE FROM posts_deduplicated 
                    WHERE pos_tid IN ({placeholders})
                """)
                
                result = transaction.execute(delete_query)
                deleted_count = result.rowcount
                
                print(f"成功刪除 {deleted_count} 筆資料庫記錄")
                
                # 交易會自動提交
            
            print("\n" + "="*60)
            print("刪除操作完成！")
            print("="*60)
            print(f"刪除統計：")
            print(f"   - 刪除條件: 內容少於 {max_length} 個字")
            print(f"   - 刪除的記錄數: {deleted_count}")
            print(f"   - 刪除的 page_name 數: {len(page_stats)}")
            
    except Exception as e:
        print(f"刪除過程中發生錯誤：{str(e)}")
        raise

def show_content_length_distribution():
    """顯示內容長度分布統計"""
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print("內容長度分布統計：")
        print("-" * 60)
        
        with engine.connect() as conn:
            # 查詢內容長度分布
            query = text("""
                WITH length_stats AS (
                    SELECT 
                        CASE 
                            WHEN LENGTH(content) < 5 THEN '0-4字'
                            WHEN LENGTH(content) < 10 THEN '5-9字'
                            WHEN LENGTH(content) < 20 THEN '10-19字'
                            WHEN LENGTH(content) < 30 THEN '20-29字'
                            WHEN LENGTH(content) < 50 THEN '30-49字'
                            WHEN LENGTH(content) < 100 THEN '50-99字'
                            WHEN LENGTH(content) < 200 THEN '100-199字'
                            ELSE '200字以上'
                        END as length_range,
                        CASE 
                            WHEN LENGTH(content) < 5 THEN 1
                            WHEN LENGTH(content) < 10 THEN 2
                            WHEN LENGTH(content) < 20 THEN 3
                            WHEN LENGTH(content) < 30 THEN 4
                            WHEN LENGTH(content) < 50 THEN 5
                            WHEN LENGTH(content) < 100 THEN 6
                            WHEN LENGTH(content) < 200 THEN 7
                            ELSE 8
                        END as sort_order
                    FROM posts_deduplicated 
                    WHERE content IS NOT NULL
                )
                SELECT 
                    length_range,
                    COUNT(*) as post_count
                FROM length_stats
                GROUP BY length_range, sort_order
                ORDER BY sort_order
            """)
            
            result = conn.execute(query)
            distribution = [(row[0], row[1]) for row in result]
            
            total_posts = sum(count for _, count in distribution)
            
            for length_range, count in distribution:
                percentage = (count / total_posts) * 100
                print(f"{length_range:<10} | {count:>8} 篇 | {percentage:>6.2f}%")
            
            print("-" * 60)
            print(f"總計：{total_posts:,} 篇貼文")
            
    except Exception as e:
        print(f"查詢過程中發生錯誤：{str(e)}")

def interactive_delete():
    """互動式刪除功能"""
    print("\n" + "="*60)
    print("短內容貼文刪除工具")
    print("="*60)
    
    # 顯示內容長度分布
    show_content_length_distribution()
    
    print(f"\n請輸入要刪除的最大內容長度：")
    try:
        max_length = int(input("輸入最大內容長度 (預設30): ").strip() or "30")
    except ValueError:
        print("輸入無效，使用預設值30")
        max_length = 30
    
    if max_length <= 0:
        print("長度必須大於0，操作取消")
        return
    
    # 先預覽要刪除的記錄
    has_records = preview_short_content_posts(max_length)
    
    if not has_records:
        print("沒有符合條件的記錄，操作取消")
        return
    
    print(f"\n請確認是否要刪除內容少於 {max_length} 個字的貼文：")
    print("1. 確認刪除")
    print("2. 取消操作")
    
    choice = input("\n請選擇 (1-2): ").strip()
    
    if choice == "1":
        confirm = input("輸入 'YES' 確認刪除，或按 Enter 取消: ").strip()
        
        if confirm.upper() == 'YES':
            delete_short_content_posts(max_length)
        else:
            print("操作已取消")
    else:
        print("操作已取消")

if __name__ == "__main__":
    print("短內容貼文刪除工具")
    print("="*60)
    
    interactive_delete() 