from sqlalchemy import create_engine, text
import os

def check_database_fields():
    """連接資料庫並查詢 page_name、page_category 和 post_type 欄位的內容"""
    
    # 資料庫連線設定
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        # 使用 SQLAlchemy 建立連線
        engine = create_engine(DB_URL)
        
        print("🔗 正在連接資料庫...")
        
        # 測試連線
        with engine.connect() as conn:
            print(f"✅ 成功連接資料庫！")
            
            # 統計每個 page_name 發的文章有哪些 page_category
            print("\n" + "="*60)
            print("PAGE_NAME 與 PAGE_CATEGORY 關聯分析")
            print("="*60)
            
            # 查詢每個 page_name 的 page_category 分布
            category_distribution_query = text("""
                SELECT 
                    page_name,
                    page_category,
                    COUNT(*) as post_count
                FROM posts_deduplicated 
                WHERE page_name IS NOT NULL 
                    AND page_category IS NOT NULL
                GROUP BY page_name, page_category 
                ORDER BY page_name, post_count DESC
            """)
            
            result = conn.execute(category_distribution_query)
            category_data = [(row[0], row[1], row[2]) for row in result]
            
            # 整理資料結構
            page_category_stats = {}
            for page_name, page_category, post_count in category_data:
                if page_name not in page_category_stats:
                    page_category_stats[page_name] = []
                page_category_stats[page_name].append((page_category, post_count))
            
            # 計算總貼文數並排序
            page_totals = []
            for page_name, categories in page_category_stats.items():
                total_posts = sum(count for _, count in categories)
                page_totals.append((page_name, total_posts, categories))
            
            # 按總貼文數排序（降序）
            page_totals.sort(key=lambda x: x[1], reverse=True)
            
            print(f"找到 {len(page_category_stats)} 個不同的 page_name")
            print("\n每個 page_name 的 page_category 分布：")
            print("-" * 80)
            
            for i, (page_name, total_posts, categories) in enumerate(page_totals, 1):
                # 將分類資訊格式化為一行
                category_info = []
                for category, count in categories:
                    category_info.append(f"{category}: {count}篇")
                
                categories_str = " | ".join(category_info)
                print(f"{i:3d}. {page_name:<30} | {total_posts:>8} 篇貼文 | {categories_str}")
            
            print("-" * 80)
            print(f"總計：{len(page_category_stats)} 個不同的 page_name")
            
            # 統計各欄位的數量分布
            print("\n" + "="*60)
            print("📈 各欄位的數量統計")
            print("="*60)
            
            # page_name 統計
            print("\n📊 PAGE_NAME 數量統計：")
            print("-" * 60)
            count_query = text("""
                SELECT page_name, COUNT(*) as post_count
                FROM posts_deduplicated 
                WHERE page_name IS NOT NULL 
                GROUP BY page_name 
                ORDER BY post_count DESC
                LIMIT 20
            """)
            
            count_result = conn.execute(count_query)
            for row in count_result:
                print(f"{row[0]:<30} | {row[1]:>8} 篇貼文")
            
            # page_category 統計
            print("\n📊 PAGE_CATEGORY 數量統計：")
            print("-" * 60)
            count_query = text("""
                SELECT page_category, COUNT(*) as post_count
                FROM posts_deduplicated 
                WHERE page_category IS NOT NULL 
                GROUP BY page_category 
                ORDER BY post_count DESC
            """)
            
            count_result = conn.execute(count_query)
            for row in count_result:
                print(f"{row[0]:<30} | {row[1]:>8} 篇貼文")
            
            print("\n" + "="*60)
            print("✅ 分析完成！")
            print("="*60)
            
    except Exception as e:
        print(f"❌ 連接資料庫時發生錯誤：{str(e)}")
        print("請檢查：")
        print("1. PostgreSQL 服務是否正在運行")
        print("2. 資料庫連線設定是否正確")
        print("3. 資料庫名稱是否存在")
        print("4. 使用者名稱和密碼是否正確")

def delete_pages_by_name(page_names_to_delete):
    """
    刪除特定 page_name 的記錄
    
    Args:
        page_names_to_delete: 要刪除的 page_name 列表
    """
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    
    try:
        engine = create_engine(DB_URL)
        
        print(f"準備刪除以下 page_name 的記錄：{page_names_to_delete}")
        
        with engine.connect() as conn:
            # 1. 先查詢要刪除的 pos_tid
            placeholders = ','.join([f"'{name}'" for name in page_names_to_delete])
            query = text(f"""
                SELECT pos_tid, page_name 
                FROM posts_deduplicated 
                WHERE page_name IN ({placeholders})
            """)
            
            result = conn.execute(query)
            to_delete = [(row[0], row[1]) for row in result]
            
            if not to_delete:
                print("沒有找到符合條件的記錄")
                return
                
            print(f"找到 {len(to_delete)} 筆記錄將被刪除")
            
            # 2. 刪除資料庫記錄
            print("刪除資料庫記錄...")
            
            # 使用 begin() 確保交易正確提交
            with engine.begin() as transaction:
                pos_tids_to_delete = [item[0] for item in to_delete]
                placeholders = ','.join([f"'{tid}'" for tid in pos_tids_to_delete])
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
            print(f"   - 刪除的 page_name: {page_names_to_delete}")
            print(f"   - 刪除的記錄數: {deleted_count}")
            
    except Exception as e:
        print(f"刪除過程中發生錯誤：{str(e)}")
        raise

def interactive_delete():
    """互動式刪除功能"""
    print("\n" + "="*60)
    print("互動式刪除功能")
    print("="*60)
    
    # 先顯示所有 page_name
    DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"
    engine = create_engine(DB_URL)
    
    with engine.connect() as conn:
        # 獲取所有 page_name 及其數量
        query = text("""
            SELECT page_name, COUNT(*) as count
            FROM posts_deduplicated 
            WHERE page_name IS NOT NULL 
            GROUP BY page_name 
            ORDER BY count DESC
        """)
        
        result = conn.execute(query)
        page_stats = [(row[0], row[1]) for row in result]
        
        #print("所有 page_name 及其貼文數量：")
        #print("-" * 60)
        #for i, (page_name, count) in enumerate(page_stats, 1):
        #    print(f"{i:3d}. {page_name:<30} | {count:>8} 篇貼文")
        
        print("\n請輸入要刪除的 page_name（用逗號分隔多個）：")
        user_input = input("輸入 page_name: ").strip()
        
        if not user_input:
            print("未輸入任何內容，操作取消")
            return
        
        # 解析輸入
        page_names_to_delete = [name.strip() for name in user_input.split(',') if name.strip()]
        
        # 確認刪除
        print(f"\n確認要刪除以下 page_name 的所有記錄嗎？")
        for name in page_names_to_delete:
            count = next((count for pname, count in page_stats if pname == name), 0)
            print(f"   - {name}: {count} 篇貼文")
        
        confirm = input("\n輸入 'YES' 確認刪除，或按 Enter 取消: ").strip()
        
        if confirm.upper() == 'YES':
            delete_pages_by_name(page_names_to_delete)
        else:
            print("操作已取消")

if __name__ == "__main__":
    # 執行基本分析
    check_database_fields()
    
    # 詢問是否要執行刪除功能
    print("\n" + "="*60)
    print("額外功能")
    print("="*60)
    print("1. 執行互動式刪除功能")
    print("2. 退出程式")
    
    choice = input("\n請選擇 (1-2): ").strip()
    
    if choice == "1":
        interactive_delete()
    else:
        print("程式結束")