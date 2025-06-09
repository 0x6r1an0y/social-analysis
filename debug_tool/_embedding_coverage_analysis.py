#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embeddings 覆蓋率分析工具
"""

from sqlalchemy import create_engine, text
import pandas as pd
import json
import os
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_embedding_coverage(db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"):
    """分析 Embeddings 覆蓋率"""
    try:
        engine = create_engine(db_url)
        
        # 1. 檢查 posts_deduplicated 表總數
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM posts_deduplicated
                WHERE content IS NOT NULL AND content != ''
            """))
            total_posts = result.scalar()
            
        # 2. 檢查 embedding_processed 表數量
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM embedding_processed"))
            processed_posts = result.scalar()
            
        # 3. 檢查 embeddings 索引檔案
        embeddings_dir = "embeddings_data"
        index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                pos_tid_to_index = json.load(f)
            index_count = len(pos_tid_to_index)
        else:
            index_count = 0
            
        # 4. 檢查沒有 embeddings 的貼文
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM posts_deduplicated p
                LEFT JOIN embedding_processed ep ON p.pos_tid = ep.pos_tid
                WHERE p.content IS NOT NULL 
                AND p.content != ''
                AND ep.pos_tid IS NULL
            """))
            missing_embeddings = result.scalar()
            
        # 5. 檢查重複的 pos_tid
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT pos_tid, COUNT(*) as count
                FROM embedding_processed
                GROUP BY pos_tid
                HAVING COUNT(*) > 1
                LIMIT 10
            """))
            duplicate_processed = result.fetchall()
            
        # 6. 檢查索引檔案中但不在 embedding_processed 表中的記錄
        if index_count > 0:
            # 隨機選取一些索引中的 pos_tid 來檢查
            sample_pos_tids = list(pos_tid_to_index.keys())[:100]
            placeholders = ','.join([f"'{pid}'" for pid in sample_pos_tids])
            
            with engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM embedding_processed 
                    WHERE pos_tid IN ({placeholders})
                """))
                sample_in_db = result.scalar()
        else:
            sample_in_db = 0
            
        # 7. 檢查內容長度分布
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN LENGTH(content) < 5 THEN 1 END) as very_short,
                    COUNT(CASE WHEN LENGTH(content) >= 5 AND LENGTH(content) < 10 THEN 1 END) as short,
                    COUNT(CASE WHEN LENGTH(content) >= 10 AND LENGTH(content) < 50 THEN 1 END) as medium,
                    COUNT(CASE WHEN LENGTH(content) >= 50 THEN 1 END) as long
                FROM posts_deduplicated 
                WHERE content IS NOT NULL AND content != ''
            """))
            content_stats = result.fetchone()
            
        # 8. 檢查沒有 embeddings 的貼文內容樣本
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT p.pos_tid, p.content, LENGTH(p.content) as content_length
                FROM posts_deduplicated p
                LEFT JOIN embedding_processed ep ON p.pos_tid = ep.pos_tid
                WHERE p.content IS NOT NULL 
                AND p.content != ''
                AND ep.pos_tid IS NULL
                ORDER BY LENGTH(p.content) DESC
                LIMIT 10
            """))
            missing_samples = result.fetchall()
            
        logger.info("=== Embeddings 覆蓋率分析 ===")
        logger.info(f"posts_deduplicated 表總數: {total_posts:,}")
        logger.info(f"embedding_processed 表數量: {processed_posts:,}")
        logger.info(f"embeddings 索引檔案數量: {index_count:,}")
        logger.info(f"缺少 embeddings 的貼文: {missing_embeddings:,}")
        
        if total_posts > 0:
            coverage_rate = (processed_posts / total_posts) * 100
            logger.info(f"覆蓋率: {coverage_rate:.2f}%")
            
            if missing_embeddings > 0:
                missing_rate = (missing_embeddings / total_posts) * 100
                logger.info(f"缺失率: {missing_rate:.2f}%")
                
        logger.info("\n=== 內容長度分布 ===")
        logger.info(f"極短內容 (<5 字): {content_stats.very_short:,}")
        logger.info(f"短內容 (5-10 字): {content_stats.short:,}")
        logger.info(f"中等內容 (10-50 字): {content_stats.medium:,}")
        logger.info(f"長內容 (>=50 字): {content_stats.long:,}")
        
        if duplicate_processed:
            logger.info(f"\n=== 發現 {len(duplicate_processed)} 個重複的處理記錄 ===")
            for pos_tid, count in duplicate_processed:
                logger.info(f"pos_tid: {pos_tid}, 重複次數: {count}")
        else:
            logger.info("\n✅ 沒有發現重複的處理記錄")
            
        if index_count > 0:
            sample_coverage = (sample_in_db / 100) * 100
            logger.info(f"\n索引檔案與資料庫同步率 (樣本): {sample_coverage:.1f}%")
            
        if missing_samples:
            logger.info(f"\n=== 缺少 Embeddings 的貼文樣本 (前 10 筆) ===")
            for pos_tid, content, length in missing_samples:
                logger.info(f"pos_tid: {pos_tid}, 長度: {length}, 內容: {content[:50]}...")
                
        # 分析可能的原因
        logger.info("\n=== 可能的原因分析 ===")
        
        if missing_embeddings > 0:
            if missing_embeddings < 1000:
                logger.info("🔍 少量貼文缺少 embeddings，可能是處理過程中的小錯誤")
            elif missing_embeddings < total_posts * 0.1:
                logger.info("🔍 約 10% 的貼文缺少 embeddings，可能是批次處理中的問題")
            else:
                logger.info("🔍 大量貼文缺少 embeddings，可能是處理過程中的嚴重問題")
                
            # 檢查是否有特定模式的貼文沒有 embeddings
            if content_stats.very_short > 0:
                logger.info("🔍 發現極短內容的貼文，可能是過濾條件過於嚴格")
                
        if processed_posts != index_count:
            logger.info("🔍 embedding_processed 表與索引檔案數量不一致")
            logger.info("   這可能表示處理狀態表與索引檔案不同步")
            
        return {
            'total_posts': total_posts,
            'processed_posts': processed_posts,
            'index_count': index_count,
            'missing_embeddings': missing_embeddings,
            'coverage_rate': (processed_posts / total_posts) * 100 if total_posts > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"分析 Embeddings 覆蓋率時發生錯誤: {str(e)}")
        raise

def check_processing_status():
    """檢查處理狀態"""
    try:
        engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash")
        
        # 檢查最近的處理記錄
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    DATE(processed_at) as process_date,
                    COUNT(*) as count
                FROM embedding_processed
                GROUP BY DATE(processed_at)
                ORDER BY process_date DESC
                LIMIT 10
            """))
            daily_stats = result.fetchall()
            
        logger.info("=== 處理狀態檢查 ===")
        logger.info("最近 10 天的處理統計:")
        for date, count in daily_stats:
            logger.info(f"  {date}: {count:,} 筆")
            
        # 檢查是否有處理中的記錄
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM embedding_processed 
                WHERE processed_at > NOW() - INTERVAL '1 hour'
            """))
            recent_count = result.scalar()
            
        logger.info(f"最近 1 小時處理的記錄: {recent_count:,} 筆")
        
    except Exception as e:
        logger.error(f"檢查處理狀態時發生錯誤: {str(e)}")
        raise

def main():
    """主要分析函數"""
    logger.info("開始 Embeddings 覆蓋率分析...")
    
    # 1. 分析覆蓋率
    logger.info("\n=== 覆蓋率分析 ===")
    coverage_stats = analyze_embedding_coverage()
    
    # 2. 檢查處理狀態
    logger.info("\n=== 處理狀態檢查 ===")
    check_processing_status()
    
    # 3. 建議
    logger.info("\n=== 建議 ===")
    if coverage_stats['missing_embeddings'] > 0:
        logger.info("🔧 建議重新運行 embedding 生成器:")
        logger.info("   python embedding_generator_memmap.py")
        
        if coverage_stats['missing_embeddings'] < 10000:
            logger.info("🔧 或者手動處理剩餘的貼文")
        else:
            logger.info("🔧 建議檢查處理過程中的錯誤日誌")
    else:
        logger.info("✅ Embeddings 覆蓋率已達 100%")
        
    logger.info("分析完成！")

if __name__ == "__main__":
    main() 