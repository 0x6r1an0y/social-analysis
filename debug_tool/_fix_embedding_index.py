#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修復 Embeddings 索引檔案同步問題
"""

from sqlalchemy import create_engine, text
import pandas as pd
import json
import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_index_sync_issue():
    """分析索引同步問題"""
    try:
        engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash")
        
        # 載入現有索引
        embeddings_dir = "embeddings_data"
        index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                pos_tid_to_index = json.load(f)
            index_count = len(pos_tid_to_index)
        else:
            pos_tid_to_index = {}
            index_count = 0
            
        # 獲取資料庫中已處理的 pos_tid
        with engine.connect() as conn:
            result = conn.execute(text("SELECT pos_tid FROM embedding_processed ORDER BY pos_tid"))
            db_pos_tids = set([row[0] for row in result.fetchall()])
            
        # 獲取索引檔案中的 pos_tid
        index_pos_tids = set(pos_tid_to_index.keys())
        
        # 分析差異
        missing_in_index = db_pos_tids - index_pos_tids
        extra_in_index = index_pos_tids - db_pos_tids
        
        logger.info("=== 索引同步問題分析 ===")
        logger.info(f"資料庫中已處理的 pos_tid: {len(db_pos_tids):,}")
        logger.info(f"索引檔案中的 pos_tid: {len(index_pos_tids):,}")
        logger.info(f"索引檔案缺少的 pos_tid: {len(missing_in_index):,}")
        logger.info(f"索引檔案多餘的 pos_tid: {len(extra_in_index):,}")
        
        if missing_in_index:
            logger.info(f"\n=== 索引檔案缺少的 pos_tid 樣本 (前 10 筆) ===")
            for pos_tid in list(missing_in_index)[:10]:
                logger.info(f"  {pos_tid}")
                
        if extra_in_index:
            logger.info(f"\n=== 索引檔案多餘的 pos_tid 樣本 (前 10 筆) ===")
            for pos_tid in list(extra_in_index)[:10]:
                logger.info(f"  {pos_tid}")
                
        return {
            'db_pos_tids': db_pos_tids,
            'index_pos_tids': index_pos_tids,
            'missing_in_index': missing_in_index,
            'extra_in_index': extra_in_index
        }
        
    except Exception as e:
        logger.error(f"分析索引同步問題時發生錯誤: {str(e)}")
        raise

def get_missing_posts_content(missing_pos_tids, limit=1000):
    """獲取缺少 embeddings 的貼文內容"""
    try:
        engine = create_engine("postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash")
        
        # 分批獲取內容
        pos_tid_list = list(missing_pos_tids)[:limit]
        placeholders = ','.join([f"'{pid}'" for pid in pos_tid_list])
        
        with engine.connect() as conn:
            sql = f"""
                SELECT pos_tid, content
                FROM posts_deduplicated
                WHERE pos_tid IN ({placeholders})
                ORDER BY pos_tid
            """
            result = conn.execute(text(sql))
            posts_data = result.fetchall()
            
        return {pos_tid: content for pos_tid, content in posts_data}
        
    except Exception as e:
        logger.error(f"獲取缺少的貼文內容時發生錯誤: {str(e)}")
        raise

def regenerate_missing_embeddings(missing_pos_tids, batch_size=1000):
    """重新生成缺少的 embeddings"""
    try:
        # 載入模型
        logger.info("正在載入模型...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embedding_dim = model.get_sentence_embedding_dimension()
        
        # 載入現有索引和 metadata
        embeddings_dir = "embeddings_data"
        index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        metadata_file = os.path.join(embeddings_dir, "metadata.json")
        embeddings_file = os.path.join(embeddings_dir, "embeddings.dat")
        
        # 載入現有索引
        with open(index_file, 'r', encoding='utf-8') as f:
            pos_tid_to_index = json.load(f)
            
        # 載入 metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 計算需要的總大小
        total_needed = len(pos_tid_to_index) + len(missing_pos_tids)
        logger.info(f"現有 embeddings: {len(pos_tid_to_index):,}")
        logger.info(f"需要添加: {len(missing_pos_tids):,}")
        logger.info(f"總共需要: {total_needed:,}")
        
        # 創建新的 memmap 檔案（如果大小不足）
        if os.path.exists(embeddings_file):
            current_size = metadata['total_embeddings']
            if current_size < total_needed:
                logger.info(f"現有 memmap 檔案大小不足 ({current_size:,} < {total_needed:,})，需要重新創建...")
                
                # 備份現有檔案
                backup_file = embeddings_file + ".backup"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(embeddings_file, backup_file)
                
                # 創建新的 memmap 檔案
                new_embeddings_array = np.memmap(
                    embeddings_file,
                    dtype=np.float32,
                    mode='w+',
                    shape=(total_needed, embedding_dim)
                )
                
                # 從備份檔案複製現有資料
                logger.info("正在複製現有 embeddings...")
                old_embeddings_array = np.memmap(
                    backup_file,
                    dtype=np.float32,
                    mode='r',
                    shape=(current_size, embedding_dim)
                )
                
                new_embeddings_array[:current_size] = old_embeddings_array[:]
                new_embeddings_array.flush()
                
                # 清理備份檔案
                del old_embeddings_array
                os.remove(backup_file)
                
                logger.info("✅ 現有 embeddings 已複製到新檔案")
            else:
                # 使用現有檔案
                new_embeddings_array = np.memmap(
                    embeddings_file,
                    dtype=np.float32,
                    mode='r+',
                    shape=(total_needed, embedding_dim)
                )
        else:
            # 創建新檔案
            new_embeddings_array = np.memmap(
                embeddings_file,
                dtype=np.float32,
                mode='w+',
                shape=(total_needed, embedding_dim)
            )
        
        # 獲取缺少的貼文內容
        missing_posts = get_missing_posts_content(missing_pos_tids, len(missing_pos_tids))
        
        logger.info(f"開始重新生成 {len(missing_posts)} 筆 embeddings...")
        
        # 分批處理
        pos_tid_list = list(missing_posts.keys())
        next_index = len(pos_tid_to_index)
        
        for i in range(0, len(pos_tid_list), batch_size):
            batch_pos_tids = pos_tid_list[i:i + batch_size]
            batch_contents = [missing_posts[pid] for pid in batch_pos_tids]
            
            # 生成 embeddings
            logger.info(f"正在處理第 {i + 1} 到 {min(i + batch_size, len(pos_tid_list))} 筆...")
            embeddings = model.encode(batch_contents, convert_to_tensor=False, show_progress_bar=False)
            
            # 儲存 embeddings
            for j, pos_tid in enumerate(batch_pos_tids):
                if pos_tid not in pos_tid_to_index:
                    pos_tid_to_index[pos_tid] = next_index
                    new_embeddings_array[next_index] = embeddings[j]
                    next_index += 1
                    
            # 強制寫入
            new_embeddings_array.flush()
            
            logger.info(f"已處理 {min(i + batch_size, len(pos_tid_list))}/{len(pos_tid_list)} 筆")
            
        # 更新 metadata
        metadata['total_embeddings'] = len(pos_tid_to_index)
        metadata['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 儲存更新後的索引和 metadata
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(pos_tid_to_index, f, ensure_ascii=False, indent=2)
            
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        logger.info("✅ 索引檔案修復完成！")
        
    except Exception as e:
        logger.error(f"重新生成 embeddings 時發生錯誤: {str(e)}")
        raise

def main():
    """主要修復函數"""
    logger.info("開始修復 Embeddings 索引檔案...")
    
    # 1. 分析問題
    logger.info("\n=== 分析索引同步問題 ===")
    sync_analysis = analyze_index_sync_issue()
    
    # 2. 檢查是否需要修復
    missing_count = len(sync_analysis['missing_in_index'])
    extra_count = len(sync_analysis['extra_in_index'])
    
    if missing_count == 0 and extra_count == 0:
        logger.info("✅ 索引檔案與資料庫已同步，無需修復")
        return
        
    logger.info(f"\n需要修復的項目:")
    logger.info(f"  - 索引檔案缺少 {missing_count:,} 筆")
    logger.info(f"  - 索引檔案多餘 {extra_count:,} 筆")
    
    # 3. 修復缺少的 embeddings
    if missing_count > 0:
        logger.info(f"\n=== 開始修復缺少的 {missing_count:,} 筆 embeddings ===")
        
        # 詢問是否繼續
        response = input(f"是否要重新生成缺少的 {missing_count:,} 筆 embeddings？(y/N): ")
        if response.lower() == 'y':
            regenerate_missing_embeddings(sync_analysis['missing_in_index'])
        else:
            logger.info("取消修復操作")
            
    # 4. 清理多餘的索引
    if extra_count > 0:
        logger.info(f"\n=== 清理多餘的 {extra_count:,} 筆索引 ===")
        
        # 載入索引
        embeddings_dir = "embeddings_data"
        index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            pos_tid_to_index = json.load(f)
            
        # 移除多餘的索引
        for pos_tid in sync_analysis['extra_in_index']:
            if pos_tid in pos_tid_to_index:
                del pos_tid_to_index[pos_tid]
                
        # 儲存清理後的索引
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(pos_tid_to_index, f, ensure_ascii=False, indent=2)
            
        logger.info("✅ 多餘的索引已清理")
        
    # 5. 最終驗證
    logger.info("\n=== 最終驗證 ===")
    final_analysis = analyze_index_sync_issue()
    
    if len(final_analysis['missing_in_index']) == 0 and len(final_analysis['extra_in_index']) == 0:
        logger.info("✅ 索引檔案修復完成，已與資料庫同步")
    else:
        logger.warning("⚠️ 仍有同步問題，請檢查")
        
    logger.info("修復完成！")

if __name__ == "__main__":
    main() 