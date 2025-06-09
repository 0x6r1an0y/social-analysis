#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料庫診斷工具 - 檢查貼文數量問題
"""

from sqlalchemy import create_engine, text
import pandas as pd
import json
import os
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_stats(db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"):
    """檢查資料庫統計資訊"""
    try:
        engine = create_engine(db_url)
        
        # 1. 總貼文數量 (posts 表)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM posts"))
            total_posts = result.scalar()
            
        # 2. posts_deduplicated 表數量
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM posts_deduplicated"))
            total_deduplicated = result.scalar()
            
        # 3. 有內容的貼文數量 (posts_deduplicated 表)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM posts_deduplicated 
                WHERE content IS NOT NULL AND content != ''
            """))
            posts_with_content = result.scalar()
            
        # 4. 有 embeddings 的貼文數量
        embeddings_dir = "embeddings_data"
        index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                pos_tid_to_index = json.load(f)
            posts_with_embeddings = len(pos_tid_to_index)
        else:
            posts_with_embeddings = 0
            
        # 5. 檢查重複的 pos_tid (posts_deduplicated 表)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT pos_tid, COUNT(*) as count
                FROM posts_deduplicated 
                WHERE content IS NOT NULL AND content != ''
                GROUP BY pos_tid
                HAVING COUNT(*) > 1
                LIMIT 10
            """))
            duplicate_pos_tids = result.fetchall()
            
        # 6. 檢查內容長度分布 (posts_deduplicated 表)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN LENGTH(content) < 10 THEN 1 END) as short_content,
                    COUNT(CASE WHEN LENGTH(content) >= 10 AND LENGTH(content) < 100 THEN 1 END) as medium_content,
                    COUNT(CASE WHEN LENGTH(content) >= 100 THEN 1 END) as long_content
                FROM posts_deduplicated 
                WHERE content IS NOT NULL AND content != ''
            """))
            content_stats = result.fetchone()
            
        logger.info("=== 資料庫統計資訊 ===")
        logger.info(f"posts 表總數量: {total_posts:,}")
        logger.info(f"posts_deduplicated 表數量: {total_deduplicated:,}")
        logger.info(f"去重率: {(1 - total_deduplicated/total_posts)*100:.2f}%")
        logger.info(f"有內容的貼文數量: {posts_with_content:,}")
        logger.info(f"有 embeddings 的貼文數量: {posts_with_embeddings:,}")
        logger.info(f"內容覆蓋率: {posts_with_content/total_deduplicated*100:.2f}%")
        logger.info(f"Embeddings 覆蓋率: {posts_with_embeddings/posts_with_content*100:.2f}%")
        
        logger.info("\n=== 內容長度分布 ===")
        logger.info(f"短內容 (<10 字): {content_stats.short_content:,}")
        logger.info(f"中等內容 (10-100 字): {content_stats.medium_content:,}")
        logger.info(f"長內容 (>=100 字): {content_stats.long_content:,}")
        
        if duplicate_pos_tids:
            logger.info(f"\n=== 發現 {len(duplicate_pos_tids)} 個重複的 pos_tid ===")
            for pos_tid, count in duplicate_pos_tids:
                logger.info(f"pos_tid: {pos_tid}, 重複次數: {count}")
        else:
            logger.info("\n✅ 沒有發現重複的 pos_tid")
            
        return {
            'total_posts': total_posts,
            'total_deduplicated': total_deduplicated,
            'posts_with_content': posts_with_content,
            'posts_with_embeddings': posts_with_embeddings,
            'deduplication_rate': (1 - total_deduplicated/total_posts)*100,
            'content_coverage': posts_with_content/total_deduplicated*100,
            'embeddings_coverage': posts_with_embeddings/posts_with_content*100 if posts_with_content > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"檢查資料庫統計時發生錯誤: {str(e)}")
        raise

def test_sample_query(db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash", limit: int = 100):
    """測試樣本查詢"""
    try:
        engine = create_engine(db_url)
        
        # 載入 embeddings 索引
        embeddings_dir = "embeddings_data"
        index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        
        if not os.path.exists(index_file):
            logger.error("embeddings 索引檔案不存在")
            return
            
        with open(index_file, 'r', encoding='utf-8') as f:
            pos_tid_to_index = json.load(f)
            
        valid_pos_tids = list(pos_tid_to_index.keys())
        logger.info(f"有 embeddings 的貼文數量: {len(valid_pos_tids):,}")
        
        # 測試不同的查詢方法
        import random
        
        # 方法 1: 隨機選取 pos_tid
        selected_pos_tids = random.sample(valid_pos_tids, min(limit, len(valid_pos_tids)))
        placeholders = ','.join([f"'{pid}'" for pid in selected_pos_tids])
        
        sql1 = f"""
            SELECT pos_tid, content, page_name, created_time
            FROM posts 
            WHERE pos_tid IN ({placeholders})
            AND content IS NOT NULL 
            AND content != ''
            ORDER BY pos_tid
        """
        
        with engine.connect() as conn:
            df1 = pd.read_sql_query(text(sql1), conn)
            
        logger.info(f"方法 1 (隨機選取 pos_tid): 獲取到 {len(df1)} 筆")
        
        # 方法 2: 直接隨機查詢
        sql2 = f"""
            SELECT pos_tid, content, page_name, created_time
            FROM posts_deduplicated 
            WHERE content IS NOT NULL 
            AND content != ''
            ORDER BY RANDOM()
            LIMIT {limit}
        """
        
        with engine.connect() as conn:
            df2 = pd.read_sql_query(text(sql2), conn)
            
        logger.info(f"方法 2 (直接隨機查詢): 獲取到 {len(df2)} 筆")
        
        # 檢查兩種方法的交集
        pos_tids_1 = set(df1['pos_tid'].tolist())
        pos_tids_2 = set(df2['pos_tid'].tolist())
        intersection = pos_tids_1.intersection(pos_tids_2)
        
        logger.info(f"兩種方法的交集: {len(intersection)} 筆")
        
        # 檢查方法 2 中有多少有 embeddings
        method2_with_embeddings = pos_tids_2.intersection(set(valid_pos_tids))
        logger.info(f"方法 2 中有 embeddings 的貼文: {len(method2_with_embeddings)} 筆")
        
        return {
            'method1_count': len(df1),
            'method2_count': len(df2),
            'intersection_count': len(intersection),
            'method2_with_embeddings': len(method2_with_embeddings)
        }
        
    except Exception as e:
        logger.error(f"測試樣本查詢時發生錯誤: {str(e)}")
        raise

def main():
    """主要診斷函數"""
    logger.info("開始資料庫診斷...")
    
    # 1. 檢查資料庫統計
    logger.info("\n=== 資料庫統計檢查 ===")
    stats = check_database_stats()
    
    # 2. 測試樣本查詢
    logger.info("\n=== 樣本查詢測試 ===")
    test_result = test_sample_query(limit=100)
    
    # 3. 分析結果
    logger.info("\n=== 分析結果 ===")
    if stats['embeddings_coverage'] < 50:
        logger.warning(f"⚠️  Embeddings 覆蓋率過低: {stats['embeddings_coverage']:.2f}%")
        logger.warning("這可能是導致樣本數量不足的原因")
    else:
        logger.info(f"✅ Embeddings 覆蓋率正常: {stats['embeddings_coverage']:.2f}%")
        
    if test_result['method2_with_embeddings'] < test_result['method2_count'] * 0.8:
        logger.warning("⚠️  隨機查詢的貼文中大部分沒有 embeddings")
        logger.info("建議使用輕量級版本的修復方法")
    else:
        logger.info("✅ 隨機查詢的貼文大部分都有 embeddings")
        
    logger.info("診斷完成！")

if __name__ == "__main__":
    main() 