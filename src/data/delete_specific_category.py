from sqlalchemy import create_engine, text

def delete_posts_by_category(target_category):
    """
    刪除指定 PAGE_CATEGORY 的所有記錄
    
    Args:
        target_category: 要刪除的 PAGE_CATEGORY
    """
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print(f"正在刪除 PAGE_CATEGORY 為 '{target_category}' 的記錄...")
        
        with engine.connect() as conn:
            # 1. 先查詢要刪除的 pos_tid
            query = text("""
                SELECT pos_tid, page_name, page_category 
                FROM posts_deduplicated 
                WHERE page_category = :category
            """)
            
            result = conn.execute(query, {"category": target_category})
            to_delete = [(row[0], row[1], row[2]) for row in result]
            
            if not to_delete:
                print(f"沒有找到 PAGE_CATEGORY 為 '{target_category}' 的記錄")
                return
                
            print(f"找到 {len(to_delete)} 筆 '{target_category}' 記錄將被刪除")
            
            # 顯示要刪除的 page_name 統計
            page_stats = {}
            for _, page_name, _ in to_delete:
                if page_name not in page_stats:
                    page_stats[page_name] = 0
                page_stats[page_name] += 1
            
            print(f"\n要刪除的 page_name 統計：")
            print("-" * 60)
            for page_name, count in sorted(page_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"{page_name:<30} | {count:>8} 篇貼文")
            print("-" * 60)
            
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
            print(f"   - 刪除的 PAGE_CATEGORY: {target_category}")
            print(f"   - 刪除的記錄數: {deleted_count}")
            print(f"   - 刪除的 page_name 數: {len(page_stats)}")
            
    except Exception as e:
        print(f"刪除過程中發生錯誤：{str(e)}")
        raise

def preview_posts_by_category(target_category):
    """預覽指定 PAGE_CATEGORY 的記錄"""
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print(f"預覽 PAGE_CATEGORY 為 '{target_category}' 的記錄...")
        
        with engine.connect() as conn:
            # 查詢要刪除的記錄統計
            query = text("""
                SELECT page_name, COUNT(*) as post_count
                FROM posts_deduplicated 
                WHERE page_category = :category
                GROUP BY page_name 
                ORDER BY post_count DESC
            """)
            
            result = conn.execute(query, {"category": target_category})
            page_stats = [(row[0], row[1]) for row in result]
            
            if not page_stats:
                print(f"沒有找到 PAGE_CATEGORY 為 '{target_category}' 的記錄")
                return
            
            # 查詢總數
            total_query = text("""
                SELECT COUNT(*) as total_count
                FROM posts_deduplicated 
                WHERE page_category = :category
            """)
            
            total_result = conn.execute(total_query, {"category": target_category})
            total_count = total_result.fetchone()[0]
            
            print(f"找到 {total_count} 筆 '{target_category}' 記錄")
            print(f"\n各 page_name 的貼文數量：")
            print("-" * 60)
            
            for i, (page_name, count) in enumerate(page_stats, 1):
                print(f"{i:3d}. {page_name:<30} | {count:>8} 篇貼文")
            
            print("-" * 60)
            print(f"總計：{len(page_stats)} 個不同的 page_name，{total_count} 篇貼文")
            
    except Exception as e:
        print(f"預覽過程中發生錯誤：{str(e)}")

def show_all_categories():
    """顯示所有可用的 PAGE_CATEGORY"""
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print("所有可用的 PAGE_CATEGORY：")
        print("-" * 60)
        
        with engine.connect() as conn:
            query = text("""
                SELECT page_category, COUNT(*) as post_count
                FROM posts_deduplicated 
                WHERE page_category IS NOT NULL
                GROUP BY page_category 
                ORDER BY post_count DESC
            """)
            
            result = conn.execute(query)
            categories = [(row[0], row[1]) for row in result]
            
            for i, (category, count) in enumerate(categories, 1):
                print(f"{i:3d}. {category:<40} | {count:>8} 篇貼文")
            
            print("-" * 60)
            print(f"總計：{len(categories)} 個不同的 PAGE_CATEGORY")
            
    except Exception as e:
        print(f"查詢過程中發生錯誤：{str(e)}")

def interactive_delete():
    """互動式刪除功能"""
    print("\n" + "="*60)
    print("PAGE_CATEGORY 記錄刪除工具")
    print("="*60)
    
    # 顯示所有可用的 PAGE_CATEGORY
    show_all_categories()
    
    print(f"\n請輸入要刪除的 PAGE_CATEGORY：")
    target_category = input("輸入 PAGE_CATEGORY: ").strip()
    
    if not target_category:
        print("未輸入任何內容，操作取消")
        return
    
    # 先預覽要刪除的記錄
    preview_posts_by_category(target_category)
    
    print(f"\n請確認是否要刪除 PAGE_CATEGORY 為 '{target_category}' 的記錄：")
    print("1. 確認刪除")
    print("2. 取消操作")
    
    choice = input("\n請選擇 (1-2): ").strip()
    
    if choice == "1":
        confirm = input("輸入 'YES' 確認刪除，或按 Enter 取消: ").strip()
        
        if confirm.upper() == 'YES':
            delete_posts_by_category(target_category)
        else:
            print("操作已取消")
    else:
        print("操作已取消")

if __name__ == "__main__":
    print("PAGE_CATEGORY 記錄刪除工具")
    print("="*60)
    
    interactive_delete() 