from sqlalchemy import create_engine, text, Column, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Optional
import time

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 1000,
                 source_table: str = "posts_deduplicated"):
        """
        初始化 Embedding 生成器
        
        Args:
            db_url: 資料庫連接字串
            model_name: 使用的 sentence-transformers 模型
            batch_size: 批次處理大小
            source_table: 來源資料表名稱
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.source_table = source_table
        self.engine = None
        self.model = None
        
        # 初始化資料庫連接
        self._init_db_connection()
        
        # 載入模型
        logger.info(f"正在載入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("模型載入完成")
        
    def _init_db_connection(self):
        """初始化資料庫連接"""
        try:
            self.engine = create_engine(self.db_url)
            logger.info("資料庫連接成功")
        except Exception as e:
            logger.error(f"資料庫連接失敗: {str(e)}")
            raise
            
    def _check_and_add_embedding_column(self):
        """檢查並新增 content_emb 欄位"""
        try:
            with self.engine.connect() as conn:
                # 檢查是否存在 content_emb 欄位
                result = conn.execute(text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{self.source_table}' AND column_name = 'content_emb'
                """))
                
                if not result.fetchone():
                    # 新增 content_emb 欄位
                    conn.execute(text(f"ALTER TABLE {self.source_table} ADD COLUMN content_emb BYTEA"))
                    conn.commit()
                    logger.info(f"✅ 已新增 {self.source_table} 的 content_emb 欄位")
                else:
                    logger.info(f"✅ {self.source_table} 的 content_emb 欄位已存在")
                    
        except Exception as e:
            logger.error(f"檢查/新增欄位時發生錯誤: {str(e)}")
            raise
            
    def _get_posts_without_embeddings(self, offset: int = 0) -> pd.DataFrame:
        """獲取還沒有 embedding 的貼文"""
        try:
            sql = f"""
                SELECT pos_tid, content 
                FROM {self.source_table}
                WHERE content_emb IS NULL 
                AND content IS NOT NULL 
                AND content != ''
                ORDER BY pos_tid
                LIMIT %s OFFSET %s
            """
            
            df = pd.read_sql_query(text(sql), self.engine, params=(self.batch_size, offset))
            return df
            
        except Exception as e:
            logger.error(f"獲取貼文資料時發生錯誤: {str(e)}")
            raise
            
    def _generate_embeddings(self, contents: list) -> np.ndarray:
        """生成文本的 embeddings"""
        try:
            # 過濾空白內容
            valid_contents = [content if content and content.strip() else " " for content in contents]
            embeddings = self.model.encode(valid_contents, convert_to_tensor=False)
            return embeddings
            
        except Exception as e:
            logger.error(f"生成 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def _save_embeddings(self, pos_tids: list, embeddings: np.ndarray):
        """儲存 embeddings 到資料庫"""
        try:
            with self.engine.connect() as conn:
                for pos_tid, embedding in zip(pos_tids, embeddings):
                    # 序列化 embedding
                    embedding_bytes = pickle.dumps(embedding.astype(np.float32))
                    
                    # 更新資料庫
                    conn.execute(text(f"""
                        UPDATE {self.source_table}
                        SET content_emb = :embedding_bytes 
                        WHERE pos_tid = :pos_tid
                    """), {
                        'embedding_bytes': embedding_bytes,
                        'pos_tid': pos_tid
                    })
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"儲存 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def _get_total_posts_count(self) -> int:
        """獲取需要處理的貼文總數"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM {self.source_table}
                    WHERE content_emb IS NULL 
                    AND content IS NOT NULL 
                    AND content != ''
                """))
                return result.scalar()
                
        except Exception as e:
            logger.error(f"獲取貼文總數時發生錯誤: {str(e)}")
            raise
            
    def process_all_posts(self):
        """處理所有貼文，生成並儲存 embeddings"""
        try:
            # 檢查並新增欄位
            self._check_and_add_embedding_column()
            
            # 獲取總數
            total_posts = self._get_total_posts_count()
            logger.info(f"需要處理的貼文總數: {total_posts}")
            
            if total_posts == 0:
                logger.info("所有貼文都已經有 embeddings 了")
                return
                
            processed = 0
            offset = 0
            
            while processed < total_posts:
                start_time = time.time()
                
                # 獲取批次資料
                df = self._get_posts_without_embeddings(offset)
                
                if df.empty:
                    logger.info("沒有更多資料需要處理")
                    break
                    
                logger.info(f"正在處理第 {processed + 1} 到 {processed + len(df)} 筆資料...")
                
                # 生成 embeddings
                embeddings = self._generate_embeddings(df['content'].tolist())
                
                # 儲存到資料庫
                self._save_embeddings(df['pos_tid'].tolist(), embeddings)
                
                processed += len(df)
                offset += self.batch_size
                
                # 計算處理時間和進度
                elapsed_time = time.time() - start_time
                progress = (processed / total_posts) * 100
                
                logger.info(f"✅ 已完成 {processed}/{total_posts} ({progress:.2f}%) - 本批次耗時: {elapsed_time:.2f}秒")
                
                # 避免過度負載，稍作休息
                time.sleep(0.1)
                
            logger.info("🎉 所有貼文的 embeddings 處理完成！")
            
        except Exception as e:
            logger.error(f"處理過程中發生錯誤: {str(e)}")
            raise
            
    def test_connection(self):
        """測試資料庫連接和模型載入"""
        try:
            # 測試資料庫
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self.source_table}"))
                total_posts = result.scalar()
                logger.info(f"資料庫連接正常，{self.source_table} 表格共有 {total_posts} 筆資料")
                
            # 測試模型
            test_text = "這是一個測試文本"
            test_embedding = self.model.encode(test_text)
            logger.info(f"模型測試正常，embedding 維度: {test_embedding.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"測試失敗: {str(e)}")
            return False

def main():
    """主要執行函數"""
    try:
        # 創建 embedding 生成器
        generator = EmbeddingGenerator(
            batch_size=500,  # 可以根據記憶體情況調整
            source_table="posts_deduplicated"  # 指定來源表
        )
        
        # 測試連接
        if not generator.test_connection():
            logger.error("連接測試失敗，程式結束")
            return
            
        # 開始處理
        logger.info("開始生成和儲存 embeddings...")
        generator.process_all_posts()
        
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 