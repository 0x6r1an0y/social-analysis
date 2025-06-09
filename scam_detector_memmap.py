#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
詐騙貼文檢測工具 (Memmap 版本)

參數調整指南：
1. batch_size: 批次處理大小
   - 100-200: 記憶體有限環境
   - 500-1000: 一般使用（推薦）
   - 2000-5000: 記憶體充足環境
   - 10000+: 高性能環境

2. 調整方式：
   - 命令列: --batch-size 500
   - 程式碼: batch_size=500
   - 預設值: 1024

3. 記憶體監控：
   - 程式會自動監控記憶體使用
   - 超過 85% 時會強制垃圾回收
   - 建議先測試較小批次大小
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
import time
import psutil  # 添加記憶體監控
import gc  # 添加垃圾回收

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """獲取當前記憶體使用情況"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # RSS in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # VMS in MB
        'percent': process.memory_percent()
    }

class ScamDetectorMemmap:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 1024,  # 修改為合理的預設值
                 embeddings_dir: str = "embeddings_data",
                 memory_optimized: bool = True):  # 新增記憶體優化選項
        """
        初始化詐騙檢測器 (使用 memmap 存儲)
        
        Args:
            db_url: 資料庫連接字串
            model_name: 使用的 sentence-transformers 模型
            batch_size: 批次處理大小（建議 200-1000）
            embeddings_dir: embeddings 存儲目錄
            memory_optimized: 是否啟用記憶體優化模式
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.memory_optimized = memory_optimized
        self.engine = None
        self.model = None
        self.embeddings_array = None  # 全局 memmap 物件
        
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
        self._init_embeddings_memmap()  # 添加全局 memmap 初始化
        
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
            
    def _init_embeddings_memmap(self):
        """初始化 embeddings memmap"""
        try:
            # 直接使用 metadata 中的總記錄數，避免重複查詢資料庫
            total_records = self.total_embeddings
            
            # 使用 memmap 載入 embeddings 檔案
            self.embeddings_array = np.memmap(
                self.embeddings_file,
                dtype=np.float32,
                mode='r',
                shape=(total_records, self.embedding_dim)
            )
            
        except Exception as e:
            logger.error(f"初始化 embeddings memmap 失敗: {str(e)}")
            raise
            
    def _get_embeddings_array(self) -> np.ndarray:
        """獲取 embeddings memmap 陣列"""
        try:
            # 直接使用 metadata 中的總記錄數，避免重複查詢資料庫
            total_records = self.total_embeddings
            
            # 使用 memmap 載入 embeddings 檔案
            embeddings_array = np.memmap(
                self.embeddings_file,
                dtype=np.float32,
                mode='r',
                shape=(total_records, self.embedding_dim)
            )
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"載入 embeddings 陣列失敗: {str(e)}")
            raise
            
    def _get_posts_batch(self, offset: int = 0, limit: Optional[int] = None) -> pd.DataFrame:
        """獲取批次貼文資料（包含有 embeddings 的貼文）"""
        try:
            if limit is None:
                limit = self.batch_size  # 使用實際的 batch_size，移除 50 筆限制
                
            # 只獲取有 embeddings 的貼文
            valid_pos_tids = list(self.pos_tid_to_index.keys())
            
            if not valid_pos_tids:
                return pd.DataFrame()
                
            # 分批獲取
            start_idx = offset
            end_idx = min(offset + limit, len(valid_pos_tids))
            batch_pos_tids = valid_pos_tids[start_idx:end_idx]
            
            if not batch_pos_tids:
                return pd.DataFrame()
                
            # 構建 SQL 查詢 - 改為從 posts_deduplicated 表查詢
            placeholders = ','.join([f"'{pid}'" for pid in batch_pos_tids])
            sql = f"""
                SELECT pos_tid, content, page_name, created_time
                FROM posts_deduplicated 
                WHERE pos_tid IN ({placeholders})
                ORDER BY pos_tid
            """
            
            df = pd.read_sql_query(text(sql), self.engine)
            return df
            
        except Exception as e:
            logger.error(f"獲取貼文資料時發生錯誤: {str(e)}")
            raise
            
    def _get_embeddings_for_pos_tids_optimized(self, pos_tids: List[str], batch_size: int = 100) -> Dict[str, np.ndarray]:
        """優化版本：分批獲取指定 pos_tids 的 embeddings（使用全局 memmap）"""
        try:
            # 檢查記憶體使用情況
            current_memory = get_memory_usage()
            if current_memory['percent'] > 85:
                logger.warning(f"記憶體使用過高: {current_memory['percent']:.1f}%，強制垃圾回收")
                gc.collect()
            
            result = {}
            
            # 使用全局 memmap 物件
            for pos_tid in pos_tids:
                if pos_tid in self.pos_tid_to_index:
                    index = self.pos_tid_to_index[pos_tid]
                    result[pos_tid] = self.embeddings_array[index].copy()
                    
            return result
            
        except Exception as e:
            logger.error(f"獲取 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def calculate_scam_scores_with_top_k(self, 
                                        pos_tids: List[str],
                                        content_embeddings: Dict[str, np.ndarray],
                                        scam_phrases: Optional[List[str]] = None,
                                        top_k: int = 5) -> Dict[str, Dict]:
        """
        計算詐騙分數並返回 top_k 相似度
        
        Args:
            pos_tids: 貼文 ID 列表
            content_embeddings: 貼文 embeddings 字典
            scam_phrases: 詐騙提示詞列表
            top_k: 返回前 k 個最相似的短語
            
        Returns:
            包含詐騙分數和 top_k 相似度的字典
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        try:
            # 生成詐騙提示詞的 embeddings
            phrase_embeddings = self.model.encode(scam_phrases, convert_to_tensor=True)
            
            # 獲取模型所在的 device
            device = phrase_embeddings.device
            
            results = {}
            
            for pos_tid in pos_tids:
                if pos_tid not in content_embeddings:
                    results[pos_tid] = {
                        'scam_score': 0.0,
                        'top_k_similarities': []
                    }
                    continue
                    
                content_emb = content_embeddings[pos_tid]
                
                # 直接使用預計算的 embedding 來計算相似度，指定正確的 device
                from torch import tensor
                content_tensor = tensor(content_emb, device=device).unsqueeze(0)
                similarities = util.cos_sim(content_tensor, phrase_embeddings).squeeze().cpu().numpy()
                
                # 獲取 top_k 相似度
                top_indices = np.argsort(similarities)[::-1][:top_k]
                top_similarities = []
                
                for idx in top_indices:
                    top_similarities.append({
                        'phrase': scam_phrases[idx],
                        'similarity': float(similarities[idx])
                    })
                
                # 取最大相似度作為詐騙分數
                score = float(np.max(similarities))
                
                results[pos_tid] = {
                    'scam_score': score,
                    'top_k_similarities': top_similarities
                }
                
            return results
            
        except Exception as e:
            logger.error(f"計算詐騙分數時發生錯誤: {str(e)}")
            raise

    def calculate_scam_scores(self, 
                            pos_tids: List[str],
                            content_embeddings: Dict[str, np.ndarray],
                            scam_phrases: Optional[List[str]] = None) -> Dict[str, float]:
        """
        計算詐騙分數
        
        Args:
            pos_tids: 貼文 ID 列表
            content_embeddings: 貼文 embeddings 字典
            scam_phrases: 詐騙提示詞列表
            
        Returns:
            詐騙風險分數字典
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        try:
            # 生成詐騙提示詞的 embeddings
            phrase_embeddings = self.model.encode(scam_phrases, convert_to_tensor=True)
            
            # 獲取模型所在的 device
            device = phrase_embeddings.device
            
            scam_scores = {}
            
            for pos_tid in pos_tids:
                if pos_tid not in content_embeddings:
                    scam_scores[pos_tid] = 0.0
                    continue
                    
                content_emb = content_embeddings[pos_tid]
                
                # 直接使用預計算的 embedding 來計算相似度，指定正確的 device
                from torch import tensor
                content_tensor = tensor(content_emb, device=device).unsqueeze(0)
                similarities = util.cos_sim(content_tensor, phrase_embeddings).squeeze().cpu().numpy()
                
                # 取最大相似度作為詐騙分數
                score = float(np.max(similarities))
                scam_scores[pos_tid] = score
                
            return scam_scores
            
        except Exception as e:
            logger.error(f"計算詐騙分數時發生錯誤: {str(e)}")
            raise
            
    def detect_scam_in_batch(self, 
                           scam_phrases: Optional[List[str]] = None,
                           threshold: float = 0.6,
                           output_file: Optional[str] = None,
                           max_results: Optional[int] = None,
                           top_k: int = 5,
                           return_top_k: bool = False) -> pd.DataFrame:
        """
        批次檢測詐騙貼文
        
        Args:
            scam_phrases: 自定義詐騙提示詞
            threshold: 詐騙風險閾值
            output_file: 輸出檔案路徑
            max_results: 最大結果數量
            top_k: 返回前 k 個最相似的短語
            return_top_k: 是否返回 top_k 相似度
            
        Returns:
            包含詐騙分數的 DataFrame
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        logger.info(f"使用的詐騙提示詞: {scam_phrases}")
        logger.info(f"詐騙風險閾值: {threshold}")
        if return_top_k:
            logger.info(f"將返回前 {top_k} 個最相似的短語")
        
        try:
            total_posts = len(self.pos_tid_to_index)
            logger.info(f"待檢測的貼文總數: {total_posts}")
            
            # 顯示初始記憶體使用情況
            initial_memory = get_memory_usage()
            logger.info(f"初始記憶體使用: {initial_memory['rss_mb']:.1f} MB ({initial_memory['percent']:.1f}%)")
            
            all_results = []
            processed = 0
            offset = 0
            
            while processed < total_posts:
                start_time = time.time()
                
                # 獲取批次資料
                df = self._get_posts_batch(offset)
                
                if df.empty:
                    break
                    
                logger.info(f"正在處理第 {processed + 1} 到 {processed + len(df)} 筆資料...")
                
                # 獲取這批次的 embeddings
                pos_tids = df['pos_tid'].tolist()
                embeddings_dict = self._get_embeddings_for_pos_tids_optimized(pos_tids)
                
                # 計算詐騙分數
                if return_top_k:
                    scam_scores_dict = self.calculate_scam_scores_with_top_k(
                        pos_tids,
                        embeddings_dict,
                        scam_phrases,
                        top_k
                    )
                    
                    # 添加分數和 top_k 相似度到 DataFrame
                    df['scam_score'] = df['pos_tid'].map(lambda x: scam_scores_dict.get(x, {}).get('scam_score', 0.0))
                    df['top_k_similarities'] = df['pos_tid'].map(lambda x: scam_scores_dict.get(x, {}).get('top_k_similarities', []))
                else:
                    scam_scores_dict = self.calculate_scam_scores(
                        pos_tids,
                        embeddings_dict,
                        scam_phrases
                    )
                    df['scam_score'] = df['pos_tid'].map(scam_scores_dict).fillna(0.0)
                
                df['is_potential_scam'] = df['scam_score'] >= threshold
                
                # 只保留可能的詐騙貼文
                high_risk_posts = df[df['is_potential_scam']].copy()
                
                if not high_risk_posts.empty:
                    all_results.append(high_risk_posts)
                    
                processed += len(df)
                offset += len(df)
                
                # 如果達到最大結果數量，提前結束
                if max_results and sum(len(r) for r in all_results) >= max_results:
                    break
                    
                elapsed_time = time.time() - start_time
                logger.info(f"已處理 {processed}/{total_posts} 筆，發現 {len(high_risk_posts)} 筆可疑貼文 - 耗時: {elapsed_time:.2f}秒")
                
                # 每處理 10 個批次檢查一次記憶體
                if processed % (self.batch_size * 10) == 0:
                    current_memory = get_memory_usage()
                    logger.info(f"已處理 {processed}/{total_posts} 筆，記憶體使用: {current_memory['rss_mb']:.1f} MB")
                    
                    # 強制垃圾回收
                    gc.collect()
                
            # 合併所有結果
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                final_results = final_results.sort_values('scam_score', ascending=False)
                
                if max_results:
                    final_results = final_results.head(max_results)
                    
                logger.info(f"🎯 檢測完成！共發現 {len(final_results)} 筆可疑詐騙貼文")
                
                # 輸出結果
                if output_file:
                    # 如果包含 top_k 相似度，需要特殊處理輸出
                    if return_top_k and 'top_k_similarities' in final_results.columns:
                        # 創建一個新的 DataFrame 來處理 top_k 相似度
                        output_df = final_results.copy()
                        # 將 top_k_similarities 轉換為可讀的格式
                        output_df['top_k_similarities'] = output_df['top_k_similarities'].apply(
                            lambda x: '; '.join([f"{item['phrase']}({item['similarity']:.3f})" for item in x])
                        )
                        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    else:
                        final_results.to_csv(output_file, index=False, encoding='utf-8-sig')
                    logger.info(f"結果已儲存到: {output_file}")
                    
                return final_results
            else:
                logger.info("沒有發現可疑的詐騙貼文")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"批次檢測時發生錯誤: {str(e)}")
            raise
            
    def detect_single_post(self, content: str, scam_phrases: Optional[List[str]] = None, top_k: int = 5) -> Dict:
        """
        檢測單一貼文
        
        Args:
            content: 貼文內容
            scam_phrases: 詐騙提示詞
            top_k: 返回前 k 個最相似的短語
            
        Returns:
            包含詐騙分數和詳細資訊的字典
        """
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
            
            result = {
                'content': content,
                'scam_score': scam_score,
                'is_potential_scam': scam_score >= 0.6,
                'top_matching_phrases': top_matches[:top_k],
                'risk_level': self._get_risk_level(scam_score)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"檢測單一貼文時發生錯誤: {str(e)}")
            raise
            
    def _get_risk_level(self, score: float) -> str:
        """根據分數獲取風險等級"""
        if score >= 0.8:
            return "極高風險"
        elif score >= 0.6:
            return "高風險"
        elif score >= 0.4:
            return "中等風險"
        elif score >= 0.2:
            return "低風險"
        else:
            return "無風險"
            
    def search_by_keywords(self, 
                          keywords: List[str], 
                          limit: int = 100,
                          threshold: float = 0.5) -> pd.DataFrame:
        """
        使用關鍵字進行語意搜尋
        
        Args:
            keywords: 搜尋關鍵字
            limit: 返回結果數量
            threshold: 相似度閾值
            
        Returns:
            搜尋結果 DataFrame
        """
        try:
            # 生成關鍵字 embeddings
            keyword_embeddings = self.model.encode(keywords, convert_to_tensor=True)
            
            # 獲取模型所在的 device
            device = keyword_embeddings.device
            
            results = []
            processed = 0
            offset = 0
            
            while len(results) < limit and processed < len(self.pos_tid_to_index):
                # 獲取批次資料
                df = self._get_posts_batch(offset, self.batch_size)
                
                if df.empty:
                    break
                    
                # 獲取 embeddings
                pos_tids = df['pos_tid'].tolist()
                embeddings_dict = self._get_embeddings_for_pos_tids_optimized(pos_tids)
                
                for _, row in df.iterrows():
                    pos_tid = row['pos_tid']
                    if pos_tid not in embeddings_dict:
                        continue
                        
                    # 計算與關鍵字的相似度，指定正確的 device
                    content_emb = embeddings_dict[pos_tid]
                    from torch import tensor
                    content_tensor = tensor(content_emb, device=device).unsqueeze(0)
                    similarities = util.cos_sim(content_tensor, keyword_embeddings).squeeze().cpu().numpy()
                    max_similarity = float(np.max(similarities))
                    
                    if max_similarity >= threshold:
                        result_row = row.copy()
                        result_row['similarity_score'] = max_similarity
                        result_row['matching_keyword'] = keywords[np.argmax(similarities)]
                        results.append(result_row)
                        
                        if len(results) >= limit:
                            break
                            
                processed += len(df)
                offset += len(df)
                
                logger.info(f"已處理 {processed} 筆，找到 {len(results)} 筆符合的結果")
                
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('similarity_score', ascending=False)
                return results_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"關鍵字搜尋時發生錯誤: {str(e)}")
            raise
            
    def get_statistics(self):
        """獲取統計資訊"""
        stats = {
            'total_embeddings': len(self.pos_tid_to_index),
            'embedding_dimension': self.embedding_dim,
            'embeddings_file_size_mb': os.path.getsize(self.embeddings_file) / 1024 / 1024 if os.path.exists(self.embeddings_file) else 0,
            'model_name': self.metadata.get('model_name', 'Unknown'),
            'last_updated': self.metadata.get('last_updated', 'Unknown'),
            'batch_size': self.batch_size
        }
        return stats
        
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'embeddings_array') and self.embeddings_array is not None:
            del self.embeddings_array
            logger.info("已清理 memmap 資源")

def main():
    """主要執行函數"""
    parser = argparse.ArgumentParser(description='詐騙貼文檢測工具 (Memmap 版本)')
    parser.add_argument('--mode', choices=['batch', 'single', 'search'], default='batch',
                       help='執行模式: batch=批次檢測, single=單一貼文檢測, search=關鍵字搜尋')
    parser.add_argument('--content', type=str, help='單一貼文內容 (single 模式使用)')
    parser.add_argument('--keywords', nargs='+', help='搜尋關鍵字 (search 模式使用)')
    parser.add_argument('--threshold', type=float, default=0.6, help='詐騙風險閾值')
    parser.add_argument('--output', type=str, help='輸出檔案路徑')
    parser.add_argument('--limit', type=int, default=500, help='最大結果數量')
    parser.add_argument('--scam-phrases', nargs='+', help='自定義詐騙提示詞')
    parser.add_argument('--embeddings-dir', type=str, default='embeddings_data', help='Embeddings 存儲目錄')
    parser.add_argument('--batch-size', type=int, default=131072, help='批次大小（建議 200-1000）')
    parser.add_argument('--top-k', type=int, default=5, help='返回前 k 個最相似的短語')
    parser.add_argument('--return-top-k', action='store_true', help='是否返回 top_k 相似度 (batch 模式)')
    
    args = parser.parse_args()
    
    try:
        # 創建檢測器
        detector = ScamDetectorMemmap(
            embeddings_dir=args.embeddings_dir,
            batch_size=args.batch_size
        )
        
        # 顯示統計資訊
        stats = detector.get_statistics()
        logger.info(f"系統統計: {stats}")
        
        if args.mode == 'batch':
            # 批次檢測
            results = detector.detect_scam_in_batch(
                scam_phrases=args.scam_phrases,
                threshold=args.threshold,
                output_file=args.output,
                max_results=args.limit,
                top_k=args.top_k,
                return_top_k=args.return_top_k
            )
            
            if not results.empty:
                print(f"\n🎯 發現 {len(results)} 筆可疑詐騙貼文:")
                if args.return_top_k and 'top_k_similarities' in results.columns:
                    # 顯示包含 top_k 相似度的結果
                    display_columns = ['pos_tid', 'page_name', 'scam_score', 'top_k_similarities', 'content']
                    print(results[display_columns].head(10).to_string())
                else:
                    print(results[['pos_tid', 'page_name', 'scam_score', 'content']].head(10).to_string())
                
        elif args.mode == 'single':
            if not args.content:
                print("❌ 請提供貼文內容 (--content)")
                return
                
            # 單一貼文檢測
            result = detector.detect_single_post(args.content, args.scam_phrases, args.top_k)
            
            print(f"\n📝 貼文內容: {result['content']}")
            print(f"🎯 詐騙分數: {result['scam_score']:.3f}")
            print(f"⚠️  風險等級: {result['risk_level']}")
            print(f"🚨 是否可疑: {'是' if result['is_potential_scam'] else '否'}")
            print(f"\n🔍 前 {args.top_k} 個最相似的詐騙短語:")
            for match in result['top_matching_phrases']:
                print(f"  - {match['phrase']}: {match['similarity']:.3f}")
                
        elif args.mode == 'search':
            if not args.keywords:
                print("❌ 請提供搜尋關鍵字 (--keywords)")
                return
                
            # 關鍵字搜尋
            results = detector.search_by_keywords(
                keywords=args.keywords,
                limit=args.limit,
                threshold=args.threshold
            )
            
            if not results.empty:
                print(f"\n🔍 找到 {len(results)} 筆相關貼文:")
                print(results[['pos_tid', 'page_name', 'similarity_score', 'matching_keyword', 'content']].head(10).to_string())
                
                if args.output:
                    results.to_csv(args.output, index=False, encoding='utf-8-sig')
                    print(f"結果已儲存到: {args.output}")
            else:
                print("沒有找到相關貼文")
                
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 