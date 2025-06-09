#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
輕量級詐騙檢測器 - 解決記憶體不足問題
"""

from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import logging
import os
import json
import argparse
from typing import List, Dict, Optional
import psutil
import gc
import random

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """獲取當前記憶體使用情況"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

class ScamDetectorLightweight:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 20,  # 非常小的批次大小
                 embeddings_dir: str = "embeddings_data"):
        """
        輕量級詐騙檢測器
        
        Args:
            db_url: 資料庫連接字串
            model_name: 使用的 sentence-transformers 模型
            batch_size: 批次處理大小（建議 10-50）
            embeddings_dir: embeddings 存儲目錄
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.engine = None
        self.model = None
        
        # 預設詐騙提示詞
        self.default_scam_phrases = [
            "加入LINE", "加入Telegram", "快速賺錢", "被動收入", 
            "投資包你賺", "私訊我", "老師帶單", "穩賺不賠",
            "輕鬆賺錢", "一天賺萬元", "保證獲利", "高報酬低風險",
            "加群組", "跟單", "操盤手", "財富自由",
            "月收入", "兼職賺錢", "在家賺錢", "網路賺錢",
            "投資理財", "虛擬貨幣", "比特幣", "挖礦",
            "借貸", "小額貸款", "急用錢", "免抵押",
            "代辦信貸", "信用卡代償", "債務整合"
        ]
        
        # 檔案路徑
        self.embeddings_file = os.path.join(embeddings_dir, "embeddings.dat")
        self.index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        self.metadata_file = os.path.join(embeddings_dir, "metadata.json")
        
        # 初始化
        self._init_db_connection()
        self._load_model(model_name)
        self._load_embeddings_metadata()
        
    def _init_db_connection(self):
        """初始化資料庫連接"""
        try:
            self.engine = create_engine(self.db_url)
            logger.info("資料庫連接成功")
        except Exception as e:
            logger.error(f"資料庫連接失敗: {str(e)}")
            raise
            
    def _load_model(self, model_name: str):
        """載入模型"""
        logger.info(f"正在載入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("模型載入完成")
        
    def _load_embeddings_metadata(self):
        """載入 embeddings metadata"""
        try:
            # 載入索引
            if not os.path.exists(self.index_file):
                raise FileNotFoundError(f"索引檔案不存在: {self.index_file}")
                
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.pos_tid_to_index = json.load(f)
                
            # 載入 metadata
            if not os.path.exists(self.metadata_file):
                raise FileNotFoundError(f"Metadata 檔案不存在: {self.metadata_file}")
                
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            self.embedding_dim = self.metadata['embedding_dim']
            self.total_embeddings = self.metadata['total_embeddings']
            
            logger.info(f"載入 embeddings metadata：")
            logger.info(f"  - 總記錄數: {self.total_embeddings}")
            logger.info(f"  - Embedding 維度: {self.embedding_dim}")
            logger.info(f"  - 最後更新: {self.metadata.get('last_updated', 'Unknown')}")
            
            # 檢查 embeddings 檔案
            if not os.path.exists(self.embeddings_file):
                raise FileNotFoundError(f"Embeddings 檔案不存在: {self.embeddings_file}")
                
        except Exception as e:
            logger.error(f"載入 embeddings metadata 失敗: {str(e)}")
            raise
            
    def _get_single_embedding(self, pos_tid: str) -> Optional[np.ndarray]:
        """獲取單一貼文的 embedding"""
        try:
            if pos_tid not in self.pos_tid_to_index:
                return None
                
            index = self.pos_tid_to_index[pos_tid]
            
            # 使用 memmap 只讀取單一 embedding
            embeddings_array = np.memmap(
                self.embeddings_file,
                dtype=np.float32,
                mode='r',
                shape=(self.total_embeddings, self.embedding_dim)
            )
            
            embedding = embeddings_array[index].copy()
            del embeddings_array  # 立即釋放記憶體
            return embedding
            
        except Exception as e:
            logger.error(f"獲取 embedding 失敗: {str(e)}")
            return None
            
    def _get_posts_sample(self, limit: int = 100) -> pd.DataFrame:
        """獲取貼文樣本（避免載入全部資料）"""
        try:
            # 從有 embeddings 的貼文中隨機選取樣本
            valid_pos_tids = list(self.pos_tid_to_index.keys())
            
            if not valid_pos_tids:
                logger.warning("沒有找到有 embeddings 的貼文")
                return pd.DataFrame()
            
            # 為了確保能獲取到足夠的貼文，我們選取更多的 pos_tid
            # 考慮到有些貼文可能在資料庫中不存在或內容為空
            target_sample_size = int(limit * 1.2)  # 增加 20% 的樣本
            selected_pos_tids = random.sample(valid_pos_tids, min(target_sample_size, len(valid_pos_tids)))
            
            logger.info(f"從 {len(valid_pos_tids):,} 筆有 embeddings 的貼文中隨機選取 {len(selected_pos_tids)} 筆")
            
            # 構建 SQL 查詢 - 改為從 posts_deduplicated 表查詢
            placeholders = ','.join([f"'{pid}'" for pid in selected_pos_tids])
            sql = f"""
                SELECT pos_tid, content, page_name, created_time
                FROM posts_deduplicated 
                WHERE pos_tid IN ({placeholders})
                AND content IS NOT NULL 
                AND content != ''
                ORDER BY pos_tid
            """
            
            df = pd.read_sql_query(text(sql), self.engine)
            logger.info(f"成功獲取 {len(df)} 筆貼文資料")
            
            # 如果獲取到的貼文數量不足，記錄警告
            if len(df) < limit:
                logger.warning(f"只獲取到 {len(df)} 筆貼文，少於要求的 {limit} 筆")
                logger.warning("這可能是因為部分貼文在 posts_deduplicated 表中不存在或內容為空")
            elif len(df) > limit:
                # 如果獲取到太多，隨機選取指定數量
                df = df.sample(n=limit, random_state=42).reset_index(drop=True)
                logger.info(f"隨機選取 {limit} 筆貼文")
            
            return df
            
        except Exception as e:
            logger.error(f"獲取貼文樣本時發生錯誤: {str(e)}")
            raise
            
    def calculate_scam_score_single(self, content: str, scam_phrases: Optional[List[str]] = None) -> Dict:
        """計算單一貼文的詐騙分數"""
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        try:
            # 生成內容 embedding
            content_embedding = self.model.encode(content, convert_to_tensor=True)
            
            # 生成詐騙提示詞的 embeddings
            phrase_embeddings = self.model.encode(scam_phrases, convert_to_tensor=True)
            
            # 計算相似度
            similarities = util.cos_sim(content_embedding, phrase_embeddings).squeeze().cpu().numpy()
            scam_score = float(np.max(similarities))
            
            # 找出最相似的詐騙短語
            top_matches = []
            for i, phrase in enumerate(scam_phrases):
                top_matches.append({
                    'phrase': phrase,
                    'similarity': float(similarities[i])
                })
                
            top_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'scam_score': scam_score,
                'top_matches': top_matches[:5],
                'is_potential_scam': scam_score >= 0.6
            }
            
        except Exception as e:
            logger.error(f"計算詐騙分數時發生錯誤: {str(e)}")
            raise
            
    def detect_scam_sample(self, 
                          sample_size: int = 100,
                          scam_phrases: Optional[List[str]] = None,
                          threshold: float = 0.6,
                          output_file: Optional[str] = None) -> pd.DataFrame:
        """
        檢測貼文樣本
        
        Args:
            sample_size: 樣本大小
            scam_phrases: 詐騙提示詞
            threshold: 詐騙風險閾值
            output_file: 輸出檔案路徑
            
        Returns:
            包含詐騙分數的 DataFrame
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        logger.info(f"開始檢測 {sample_size} 筆貼文樣本")
        logger.info(f"詐騙風險閾值: {threshold}")
        
        try:
            # 顯示初始記憶體使用情況
            initial_memory = get_memory_usage()
            logger.info(f"初始記憶體使用: {initial_memory['rss_mb']:.1f} MB ({initial_memory['percent']:.1f}%)")
            
            # 獲取貼文樣本
            df = self._get_posts_sample(sample_size)
            
            if df.empty:
                logger.warning("沒有找到符合條件的貼文")
                return pd.DataFrame()
                
            logger.info(f"獲取到 {len(df)} 筆貼文樣本")
            
            results = []
            
            for idx, row in df.iterrows():
                try:
                    # 計算詐騙分數
                    scam_result = self.calculate_scam_score_single(row['content'], scam_phrases)
                    
                    # 只保留高風險貼文
                    if scam_result['scam_score'] >= threshold:
                        result_row = row.copy()
                        result_row['scam_score'] = scam_result['scam_score']
                        result_row['is_potential_scam'] = scam_result['is_potential_scam']
                        result_row['top_matches'] = scam_result['top_matches']
                        results.append(result_row)
                        
                    # 每處理 10 筆檢查一次記憶體
                    if (idx + 1) % 10 == 0:
                        current_memory = get_memory_usage()
                        logger.info(f"已處理 {idx + 1}/{len(df)} 筆，記憶體使用: {current_memory['rss_mb']:.1f} MB")
                        
                        # 強制垃圾回收
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"處理貼文 {row['pos_tid']} 時發生錯誤: {str(e)}")
                    continue
                    
            if results:
                final_results = pd.DataFrame(results)
                final_results = final_results.sort_values('scam_score', ascending=False)
                
                logger.info(f"🎯 檢測完成！在 {len(df)} 筆樣本中發現 {len(final_results)} 筆可疑詐騙貼文")
                
                # 輸出結果
                if output_file:
                    # 處理 top_matches 欄位
                    output_df = final_results.copy()
                    output_df['top_matches'] = output_df['top_matches'].apply(
                        lambda x: '; '.join([f"{item['phrase']}({item['similarity']:.3f})" for item in x])
                    )
                    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    logger.info(f"結果已儲存到: {output_file}")
                    
                return final_results
            else:
                logger.info("沒有發現可疑的詐騙貼文")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"樣本檢測時發生錯誤: {str(e)}")
            raise
            
    def get_statistics(self):
        """獲取統計資訊"""
        stats = {
            'total_embeddings': len(self.pos_tid_to_index),
            'embedding_dimension': self.embedding_dim,
            'embeddings_file_size_mb': os.path.getsize(self.embeddings_file) / 1024 / 1024 if os.path.exists(self.embeddings_file) else 0,
            'model_name': self.metadata.get('model_name', 'Unknown'),
            'last_updated': self.metadata.get('last_updated', 'Unknown')
        }
        return stats

def main():
    """主要執行函數"""
    parser = argparse.ArgumentParser(description='輕量級詐騙貼文檢測工具')
    parser.add_argument('--sample-size', type=int, default=100, help='樣本大小')
    parser.add_argument('--threshold', type=float, default=0.6, help='詐騙風險閾值')
    parser.add_argument('--output', type=str, help='輸出檔案路徑')
    parser.add_argument('--scam-phrases', nargs='+', help='自定義詐騙提示詞')
    parser.add_argument('--embeddings-dir', type=str, default='embeddings_data', help='Embeddings 存儲目錄')
    
    args = parser.parse_args()
    
    try:
        # 創建檢測器
        detector = ScamDetectorLightweight(embeddings_dir=args.embeddings_dir)
        
        # 顯示統計資訊
        stats = detector.get_statistics()
        logger.info(f"系統統計: {stats}")
        
        # 樣本檢測
        results = detector.detect_scam_sample(
            sample_size=args.sample_size,
            scam_phrases=args.scam_phrases,
            threshold=args.threshold,
            output_file=args.output
        )
        
        if not results.empty:
            print(f"\n🎯 發現 {len(results)} 筆可疑詐騙貼文:")
            print("\n前 10 筆結果:")
            for idx, row in results.head(10).iterrows():
                print(f"\n--- 貼文 {idx + 1} ---")
                print(f"ID: {row['pos_tid']}")
                print(f"頁面: {row['page_name']}")
                print(f"詐騙分數: {row['scam_score']:.3f}")
                print(f"內容: {row['content'][:100]}...")
                
                print("最相似的詐騙短語:")
                for i, match in enumerate(row['top_matches'][:3], 1):
                    print(f"  {i}. {match['phrase']}: {match['similarity']:.3f}")
        else:
            print("沒有發現可疑的詐騙貼文")
            
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 