from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import logging
import time
import os
import json
from typing import Optional
import torch

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGeneratorMemmap:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 1000,
                 embeddings_dir: str = "embeddings_data",
                 source_table: str = "posts_deduplicated",
                 device: str = "auto"):
        """
        初始化 Embedding 生成器 (使用 memmap 存儲)
        
        Args:
            db_url: 資料庫連接字串
            model_name: 使用的 sentence-transformers 模型
            batch_size: 批次處理大小
            embeddings_dir: embeddings 存儲目錄
            source_table: 來源資料表名稱
            device: 計算裝置 ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.source_table = source_table
        self.engine = None
        self.model = None
        self.model_name = model_name  # 保存模型名稱
        
        # 設定計算裝置
        self.device = self._setup_device(device)
        logger.info(f"將使用裝置: {self.device}")
        
        # 創建存儲目錄
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # 定義檔案路徑
        self.embeddings_file = os.path.join(embeddings_dir, "embeddings.dat")
        self.index_file = os.path.join(embeddings_dir, "pos_tid_index.json")
        self.metadata_file = os.path.join(embeddings_dir, "metadata.json")
        
        # 初始化資料庫連接
        self._init_db_connection()
        
        # 創建處理狀態表
        self._create_processed_table()
        
        # 載入模型
        logger.info(f"正在載入模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"模型載入完成，embedding 維度: {self.embedding_dim}")
        
        # 如果使用 GPU，顯示 GPU 資訊
        if self.device.startswith('cuda'):
            self._log_gpu_info()
        
        # 載入或初始化索引
        self.pos_tid_to_index = self._load_index()
        self.next_index = len(self.pos_tid_to_index)
        
        # 同步處理狀態表
        self._sync_processed_table()
        
    def _setup_device(self, device: str) -> str:
        """設定計算裝置"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("✅ 檢測到 CUDA，將使用 GPU 加速")
            else:
                device = "cpu"
                logger.info("⚠️ 未檢測到 CUDA，將使用 CPU")
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("⚠️ 指定使用 CUDA 但系統不支援，改用 CPU")
                device = "cpu"
            else:
                logger.info(f"✅ 將使用指定的 GPU: {device}")
        else:
            logger.info(f"✅ 將使用指定的裝置: {device}")
            
        return device
        
    def _log_gpu_info(self):
        """記錄 GPU 資訊"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                
                logger.info(f"🎮 GPU 資訊:")
                logger.info(f"   - 可用 GPU 數量: {gpu_count}")
                logger.info(f"   - 當前使用 GPU: {current_device}")
                logger.info(f"   - GPU 名稱: {gpu_name}")
                logger.info(f"   - GPU 記憶體: {gpu_memory:.1f} GB")
                
        except Exception as e:
            logger.warning(f"無法獲取 GPU 資訊: {str(e)}")

    def _init_db_connection(self):
        """初始化資料庫連接"""
        try:
            self.engine = create_engine(self.db_url)
            logger.info("資料庫連接成功")
        except Exception as e:
            logger.error(f"資料庫連接失敗: {str(e)}")
            raise
            
    def _create_processed_table(self):
        """創建處理狀態表"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS embedding_processed (
                        pos_tid TEXT PRIMARY KEY,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # 創建索引以提高查詢效能
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_embedding_processed_pos_tid 
                    ON embedding_processed(pos_tid)
                """))
                
        except Exception as e:
            logger.error(f"創建處理狀態表時發生錯誤: {str(e)}")
            raise
            
    def _sync_processed_table(self):
        """同步處理狀態表與索引檔案"""
        try:
            processed_ids = list(self.pos_tid_to_index.keys())
            
            if processed_ids:
                with self.engine.begin() as conn:
                    # 批次插入新的已處理記錄
                    batch_size = 1000
                    for i in range(0, len(processed_ids), batch_size):
                        batch = processed_ids[i:i + batch_size]
                        values = ','.join([f"('{pid}')" for pid in batch])
                        conn.execute(text(f"""
                            INSERT INTO embedding_processed (pos_tid) 
                            VALUES {values}
                            ON CONFLICT (pos_tid) DO NOTHING
                        """))
        except Exception as e:
            logger.error(f"同步處理狀態表時發生錯誤: {str(e)}")
            
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
            'model_name': self.model_name,
            'device': self.device,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果使用 GPU，添加 GPU 資訊
        if self.device.startswith('cuda') and torch.cuda.is_available():
            try:
                metadata['gpu_info'] = {
                    'gpu_name': torch.cuda.get_device_name(),
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    'cuda_version': torch.version.cuda
                }
            except:
                pass
                
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
    def _get_total_posts_count(self) -> int:
        """獲取需要處理的貼文總數"""
        try:
            with self.engine.connect() as conn:
                sql = f"""
                    SELECT COUNT(*) 
                    FROM {self.source_table} p
                    LEFT JOIN embedding_processed ep ON p.pos_tid = ep.pos_tid
                    WHERE p.content IS NOT NULL 
                    AND p.content != ''
                    AND ep.pos_tid IS NULL
                """
                
                result = conn.execute(text(sql))
                return result.scalar()
                
        except Exception as e:
            logger.error(f"獲取貼文總數時發生錯誤: {str(e)}")
            raise
            
    def _get_posts_without_embeddings(self, offset: int = 0) -> pd.DataFrame:
        """獲取還沒有 embedding 的貼文 (使用處理狀態表)"""
        try:
            with self.engine.connect() as conn:
                sql = f"""
                    SELECT p.pos_tid, p.content 
                    FROM {self.source_table} p
                    LEFT JOIN embedding_processed ep ON p.pos_tid = ep.pos_tid
                    WHERE p.content IS NOT NULL 
                    AND p.content != ''
                    AND ep.pos_tid IS NULL
                    ORDER BY p.pos_tid
                    LIMIT {self.batch_size} OFFSET {offset}
                """
                
                df = pd.read_sql_query(text(sql), conn)
                return df
            
        except Exception as e:
            logger.error(f"獲取貼文資料時發生錯誤: {str(e)}")
            raise
            
    def _generate_embeddings(self, contents: list) -> np.ndarray:
        """生成文本的 embeddings (支援 GPU 加速)"""
        try:
            # 過濾空白內容
            valid_contents = [content if content and content.strip() else " " for content in contents]
            
            # 如果使用 GPU，記錄處理前的記憶體狀態
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理未使用的記憶體
                memory_before = torch.cuda.memory_allocated() / 1024**2
                
            # 使用 GPU 加速生成 embeddings - 調高batch size以更好利用資源
            embeddings = self.model.encode(
                valid_contents, 
                convert_to_tensor=False, 
                show_progress_bar=False,  
                batch_size=min(len(valid_contents), 1024 if self.device.startswith('cuda') else 256),  # 調高到1024充分利用GPU
                device=self.device
            )
            
            # 如果使用 GPU，記錄處理後的記憶體狀態
            if self.device.startswith('cuda') and torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                memory_peak = torch.cuda.max_memory_allocated() / 1024**2
                
                # 每隔一段時間記錄一次詳細的GPU狀態
                if hasattr(self, '_last_gpu_log_time'):
                    if time.time() - self._last_gpu_log_time > 30:  # 每30秒記錄一次
                        logger.info(f"🎮 GPU 記憶體狀態: 使用中 {memory_after:.1f}MB, 已保留 {memory_reserved:.1f}MB, 峰值 {memory_peak:.1f}MB")
                        self._last_gpu_log_time = time.time()
                else:
                    self._last_gpu_log_time = time.time()
                    logger.info(f"🎮 GPU 記憶體狀態: 使用中 {memory_after:.1f}MB, 已保留 {memory_reserved:.1f}MB")
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"生成 embeddings 時發生錯誤: {str(e)}")
            # 如果 GPU 出錯，嘗試回退到 CPU
            if self.device.startswith('cuda'):
                logger.warning("GPU 處理失敗，嘗試使用 CPU...")
                try:
                    self.model = self.model.to('cpu')
                    self.device = 'cpu'
                    embeddings = self.model.encode(
                        valid_contents, 
                        convert_to_tensor=False, 
                        show_progress_bar=False,
                        device='cpu'
                    )
                    return embeddings.astype(np.float32)
                except:
                    pass
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
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM {self.source_table}
                    WHERE content IS NOT NULL 
                    AND content != ''
                """))
                return result.scalar()
        except Exception as e:
            logger.error(f"估算總記錄數時發生錯誤: {str(e)}")
            raise
            
    def _save_embeddings_batch(self, pos_tids: list, embeddings: np.ndarray, embeddings_array: np.ndarray):
        """批次儲存 embeddings 到 memmap 並更新處理狀態表"""
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
            
            # 更新處理狀態表
            with self.engine.begin() as conn:
                values = ','.join([f"('{pid}')" for pid in pos_tids])
                conn.execute(text(f"""
                    INSERT INTO embedding_processed (pos_tid) 
                    VALUES {values}
                    ON CONFLICT (pos_tid) DO NOTHING
                """))
            
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
            
            # 性能統計
            total_db_time = 0
            total_embedding_time = 0
            total_save_time = 0
            
            while processed < remaining_posts:
                batch_start_time = time.time()
                
                # 1. 資料庫查詢時間
                db_start = time.time()
                df = self._get_posts_without_embeddings(offset)
                db_time = time.time() - db_start
                total_db_time += db_time
                
                if df.empty:
                    logger.info("沒有更多資料需要處理")
                    break
                    
                # 2. Embedding 生成時間
                embedding_start = time.time()
                embeddings = self._generate_embeddings(df['content'].tolist())
                embedding_time = time.time() - embedding_start
                total_embedding_time += embedding_time
                
                # 3. 儲存時間
                save_start = time.time()
                self._save_embeddings_batch(df['pos_tid'].tolist(), embeddings, embeddings_array)
                save_time = time.time() - save_start
                total_save_time += save_time
                
                processed += len(df)
                
                # 計算處理時間和進度
                total_batch_time = time.time() - batch_start_time
                progress = (processed / remaining_posts) * 100
                
                # 詳細性能報告
                logger.info(f"✅ 已完成 {processed}/{remaining_posts} ({progress:.2f}%) - 總耗時: {total_batch_time:.2f}秒")
                logger.info(f"   📊 性能分析: DB查詢 {db_time:.2f}s ({db_time/total_batch_time*100:.1f}%) | "
                          f"Embedding生成 {embedding_time:.2f}s ({embedding_time/total_batch_time*100:.1f}%) | "
                          f"資料儲存 {save_time:.2f}s ({save_time/total_batch_time*100:.1f}%)")
                
                # 定期保存索引
                if processed % (self.batch_size * 3) == 0:  # 每 3 個批次儲存一次
                    self._save_index()
                    self._save_metadata()
                    logger.info("📝 已保存索引和 metadata")
                    
                    # 顯示累積性能統計
                    batches_processed = processed // self.batch_size
                    if batches_processed > 0:
                        avg_db_time = total_db_time / batches_processed
                        avg_embedding_time = total_embedding_time / batches_processed
                        avg_save_time = total_save_time / batches_processed
                        
                        logger.info(f"📈 累積性能統計 (平均每批次):")
                        logger.info(f"   DB查詢: {avg_db_time:.2f}s | Embedding: {avg_embedding_time:.2f}s | 儲存: {avg_save_time:.2f}s")
                
                # 避免過度負載
                time.sleep(0.1)
                
            # 最終保存
            self._save_index()
            self._save_metadata()
            
            # 最終性能報告
            batches_total = processed // self.batch_size
            if batches_total > 0:
                logger.info("🎉 所有貼文的 embeddings 處理完成！")
                logger.info(f"📊 最終性能統計:")
                logger.info(f"   平均DB查詢時間: {total_db_time/batches_total:.2f}s ({total_db_time/(total_db_time+total_embedding_time+total_save_time)*100:.1f}%)")
                logger.info(f"   平均Embedding時間: {total_embedding_time/batches_total:.2f}s ({total_embedding_time/(total_db_time+total_embedding_time+total_save_time)*100:.1f}%)")
                logger.info(f"   平均儲存時間: {total_save_time/batches_total:.2f}s ({total_save_time/(total_db_time+total_embedding_time+total_save_time)*100:.1f}%)")
            
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
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self.source_table}"))
                total_posts = result.scalar()
                logger.info(f"資料庫連接正常，{self.source_table} 表格共有 {total_posts} 筆資料")
                
            # 測試模型和裝置
            test_text = "這是一個測試文本"
            start_time = time.time()
            test_embedding = self.model.encode(test_text)
            encode_time = time.time() - start_time
            
            logger.info(f"模型測試正常，embedding 維度: {test_embedding.shape}")
            logger.info(f"當前使用裝置: {self.device}")
            logger.info(f"單個文本編碼耗時: {encode_time:.4f}秒")
            
            # 如果使用 GPU，測試 GPU 性能
            if self.device.startswith('cuda'):
                logger.info("🎮 進行 GPU 性能測試...")
                test_texts = ["測試文本"] * 100
                start_time = time.time()
                test_embeddings = self.model.encode(test_texts, batch_size=32)
                batch_time = time.time() - start_time
                avg_time = batch_time / 100
                logger.info(f"批次處理 100 個文本耗時: {batch_time:.4f}秒 (平均每個: {avg_time:.6f}秒)")
                
                # 顯示 GPU 記憶體使用情況
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    memory_cached = torch.cuda.memory_reserved() / 1024**2
                    logger.info(f"GPU 記憶體使用: {memory_allocated:.1f} MB (已分配) / {memory_cached:.1f} MB (已快取)")
            
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
            batch_size=65536,  # 調整到 65536 以提升處理效率
            embeddings_dir="embeddings_data",
            source_table="posts_deduplicated",  # 指定來源表
            device="auto"  # 使用自動偵測裝置
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