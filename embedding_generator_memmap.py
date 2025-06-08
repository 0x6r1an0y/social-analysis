from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import logging
import time
import os
import json
from typing import Optional, Tuple
import pickle

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGeneratorMemmap:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 1000,
                 embeddings_dir: str = "embeddings_data"):
        """
        初始化 Embedding 生成器 (使用 memmap 存儲)
        
        Args:
            db_url: 資料庫連接字串
            model_name: 使用的 sentence-transformers 模型
            batch_size: 批次處理大小
            embeddings_dir: embeddings 存儲目錄
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.engine = None
        self.model = None
        
        # 創建存儲目錄
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # 定義檔案路徑
        self.embeddings_file = os.path.join(embeddings_dir, "embeddings.dat")
        self.index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        self.metadata_file = os.path.join(embeddings_dir, "metadata.json")
        
        # 初始化資料庫連接
        self._init_db_connection()
        
        # 載入模型
        logger.info(f"正在載入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"模型載入完成，embedding 維度: {self.embedding_dim}")
        
        # 載入或初始化索引
        self.pos_tid_to_index = self._load_index()
        self.next_index = len(self.pos_tid_to_index)
        
    def _init_db_connection(self):
        """初始化資料庫連接"""
        try:
            self.engine = create_engine(self.db_url)
            logger.info("資料庫連接成功")
        except Exception as e:
            logger.error(f"資料庫連接失敗: {str(e)}")
            raise
            
    def _load_index(self) -> dict:
        """載入 pos_tid 到 index 的映射"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                logger.info(f"載入現有索引，共 {len(index_data)} 筆記錄")
                return index_data
        else:
            logger.info("創建新的索引檔案")
            return {}
            
    def _save_index(self):
        """儲存索引檔案"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.pos_tid_to_index, f, ensure_ascii=False, indent=2)
            
    def _save_metadata(self):
        """儲存 metadata"""
        metadata = {
            'total_embeddings': len(self.pos_tid_to_index),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model.model_name,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
    def _get_total_posts_count(self) -> int:
        """獲取需要處理的貼文總數"""
        try:
            # 獲取所有還沒處理的 pos_tid
            processed_ids = set(self.pos_tid_to_index.keys())
            
            with self.engine.connect() as conn:
                if processed_ids:
                    # 構建 NOT IN 子句
                    placeholders = ','.join([f"'{pid}'" for pid in processed_ids])
                    sql = f"""
                        SELECT COUNT(*) 
                        FROM posts 
                        WHERE content IS NOT NULL 
                        AND content != ''
                        AND pos_tid NOT IN ({placeholders})
                    """
                else:
                    sql = """
                        SELECT COUNT(*) 
                        FROM posts 
                        WHERE content IS NOT NULL 
                        AND content != ''
                    """
                
                result = conn.execute(text(sql))
                return result.scalar()
                
        except Exception as e:
            logger.error(f"獲取貼文總數時發生錯誤: {str(e)}")
            raise
            
    def _get_posts_without_embeddings(self, offset: int = 0) -> pd.DataFrame:
        """獲取還沒有 embedding 的貼文"""
        try:
            processed_ids = set(self.pos_tid_to_index.keys())
            
            if processed_ids:
                # 構建 NOT IN 子句 
                placeholders = ','.join([f"'{pid}'" for pid in processed_ids])
                sql = f"""
                    SELECT pos_tid, content 
                    FROM posts 
                    WHERE content IS NOT NULL 
                    AND content != ''
                    AND pos_tid NOT IN ({placeholders})
                    ORDER BY pos_tid
                    LIMIT {self.batch_size} OFFSET {offset}
                """
            else:
                sql = f"""
                    SELECT pos_tid, content 
                    FROM posts 
                    WHERE content IS NOT NULL 
                    AND content != ''
                    ORDER BY pos_tid
                    LIMIT {self.batch_size} OFFSET {offset}
                """
            
            df = pd.read_sql_query(text(sql), self.engine)
            return df
            
        except Exception as e:
            logger.error(f"獲取貼文資料時發生錯誤: {str(e)}")
            raise
            
    def _generate_embeddings(self, contents: list) -> np.ndarray:
        """生成文本的 embeddings"""
        try:
            # 過濾空白內容
            valid_contents = [content if content and content.strip() else " " for content in contents]
            embeddings = self.model.encode(valid_contents, convert_to_tensor=False, show_progress_bar=False)
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"生成 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def _get_memmap_array(self, total_size: int) -> np.ndarray:
        """獲取或創建 memmap 陣列"""
        if os.path.exists(self.embeddings_file):
            # 載入現有的 memmap
            embeddings_array = np.memmap(
                self.embeddings_file, 
                dtype=np.float32, 
                mode='r+',
                shape=(total_size, self.embedding_dim)
            )
        else:
            # 創建新的 memmap
            embeddings_array = np.memmap(
                self.embeddings_file, 
                dtype=np.float32, 
                mode='w+',
                shape=(total_size, self.embedding_dim)
            )
            
        return embeddings_array
        
    def _estimate_total_records(self) -> int:
        """估算總記錄數"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM posts 
                    WHERE content IS NOT NULL 
                    AND content != ''
                """))
                return result.scalar()
        except Exception as e:
            logger.error(f"估算總記錄數時發生錯誤: {str(e)}")
            raise
            
    def _save_embeddings_batch(self, pos_tids: list, embeddings: np.ndarray, embeddings_array: np.ndarray):
        """批次儲存 embeddings 到 memmap"""
        try:
            indices = []
            for pos_tid in pos_tids:
                if pos_tid not in self.pos_tid_to_index:
                    self.pos_tid_to_index[pos_tid] = self.next_index
                    indices.append(self.next_index)
                    self.next_index += 1
                else:
                    indices.append(self.pos_tid_to_index[pos_tid])
                    
            # 寫入 embeddings
            for i, (idx, embedding) in enumerate(zip(indices, embeddings)):
                embeddings_array[idx] = embedding
                
            # 強制寫入磁碟
            embeddings_array.flush()
            
        except Exception as e:
            logger.error(f"儲存 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def process_all_posts(self):
        """處理所有貼文，生成並儲存 embeddings"""
        try:
            # 估算總記錄數以創建 memmap
            estimated_total = self._estimate_total_records()
            logger.info(f"估算總記錄數: {estimated_total}")
            
            # 創建 memmap 陣列
            embeddings_array = self._get_memmap_array(estimated_total)
            
            # 獲取需要處理的數量
            remaining_posts = self._get_total_posts_count()
            logger.info(f"需要處理的貼文數量: {remaining_posts}")
            
            if remaining_posts == 0:
                logger.info("所有貼文都已經有 embeddings 了")
                return
                
            processed = 0
            offset = 0
            
            while processed < remaining_posts:
                start_time = time.time()
                
                # 獲取批次資料
                df = self._get_posts_without_embeddings(offset)
                
                if df.empty:
                    logger.info("沒有更多資料需要處理")
                    break
                    
                logger.info(f"正在處理第 {processed + 1} 到 {processed + len(df)} 筆資料...")
                
                # 生成 embeddings
                embeddings = self._generate_embeddings(df['content'].tolist())
                
                # 儲存到 memmap
                self._save_embeddings_batch(df['pos_tid'].tolist(), embeddings, embeddings_array)
                
                processed += len(df)
                
                # 計算處理時間和進度
                elapsed_time = time.time() - start_time
                progress = (processed / remaining_posts) * 100
                
                logger.info(f"✅ 已完成 {processed}/{remaining_posts} ({progress:.2f}%) - 本批次耗時: {elapsed_time:.2f}秒")
                
                # 定期保存索引
                if processed % (self.batch_size * 10) == 0:
                    self._save_index()
                    self._save_metadata()
                    logger.info("📝 已保存索引和 metadata")
                
                # 避免過度負載
                time.sleep(0.1)
                
            # 最終保存
            self._save_index()
            self._save_metadata()
            
            logger.info("🎉 所有貼文的 embeddings 處理完成！")
            logger.info(f"📁 Embeddings 檔案: {self.embeddings_file}")
            logger.info(f"📋 索引檔案: {self.index_file}")
            logger.info(f"📊 Metadata 檔案: {self.metadata_file}")
            
        except Exception as e:
            logger.error(f"處理過程中發生錯誤: {str(e)}")
            # 即使出錯也要保存進度
            self._save_index()
            self._save_metadata()
            raise
            
    def get_embedding(self, pos_tid: str) -> Optional[np.ndarray]:
        """獲取指定 pos_tid 的 embedding"""
        if pos_tid not in self.pos_tid_to_index:
            return None
            
        try:
            index = self.pos_tid_to_index[pos_tid]
            estimated_total = self._estimate_total_records()
            
            embeddings_array = np.memmap(
                self.embeddings_file, 
                dtype=np.float32, 
                mode='r',
                shape=(estimated_total, self.embedding_dim)
            )
            
            return embeddings_array[index].copy()
            
        except Exception as e:
            logger.error(f"獲取 embedding 時發生錯誤: {str(e)}")
            return None
            
    def get_embeddings_batch(self, pos_tids: list) -> dict:
        """批次獲取 embeddings"""
        try:
            estimated_total = self._estimate_total_records()
            embeddings_array = np.memmap(
                self.embeddings_file, 
                dtype=np.float32, 
                mode='r',
                shape=(estimated_total, self.embedding_dim)
            )
            
            result = {}
            for pos_tid in pos_tids:
                if pos_tid in self.pos_tid_to_index:
                    index = self.pos_tid_to_index[pos_tid]
                    result[pos_tid] = embeddings_array[index].copy()
                    
            return result
            
        except Exception as e:
            logger.error(f"批次獲取 embeddings 時發生錯誤: {str(e)}")
            return {}
            
    def test_connection(self):
        """測試資料庫連接和模型載入"""
        try:
            # 測試資料庫
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM posts"))
                total_posts = result.scalar()
                logger.info(f"資料庫連接正常，posts 表格共有 {total_posts} 筆資料")
                
            # 測試模型
            test_text = "這是一個測試文本"
            test_embedding = self.model.encode(test_text)
            logger.info(f"模型測試正常，embedding 維度: {test_embedding.shape}")
            
            # 測試檔案系統
            logger.info(f"Embeddings 目錄: {self.embeddings_dir}")
            logger.info(f"已處理的記錄數: {len(self.pos_tid_to_index)}")
            
            return True
            
        except Exception as e:
            logger.error(f"測試失敗: {str(e)}")
            return False
            
    def get_statistics(self):
        """獲取統計資訊"""
        stats = {
            'total_processed': len(self.pos_tid_to_index),
            'embedding_dimension': self.embedding_dim,
            'embeddings_file_size': os.path.getsize(self.embeddings_file) if os.path.exists(self.embeddings_file) else 0,
            'index_file_size': os.path.getsize(self.index_file) if os.path.exists(self.index_file) else 0
        }
        
        if stats['embeddings_file_size'] > 0:
            stats['embeddings_file_size_mb'] = stats['embeddings_file_size'] / 1024 / 1024
            
        return stats

def main():
    """主要執行函數"""
    try:
        # 創建 embedding 生成器
        generator = EmbeddingGeneratorMemmap(
            batch_size=1000,  # 可以設置更大的批次大小
            embeddings_dir="embeddings_data"
        )
        
        # 測試連接
        if not generator.test_connection():
            logger.error("連接測試失敗，程式結束")
            return
            
        # 顯示統計資訊
        stats = generator.get_statistics()
        logger.info(f"當前統計: {stats}")
        
        # 開始處理
        logger.info("開始生成和儲存 embeddings...")
        generator.process_all_posts()
        
        # 最終統計
        final_stats = generator.get_statistics()
        logger.info(f"處理完成統計: {final_stats}")
        
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 