import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from candidates_dataloader_to_sql import fetch_candidate_posts
import logging
import datetime
import subprocess
import os
from logging import Filter, Formatter
import socket
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import time
import psutil
import gc
from typing import List, Dict, Optional
import random
import atexit
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager

# 自定義的 IP 過濾器
class IPFilter(Filter):
    def filter(self, record):
        try:
            # 使用 st.context.headers 獲取請求標頭
            headers = st.context.headers if hasattr(st, 'context') else None
            if headers:
                # 嘗試從 X-Forwarded-For 獲取真實 IP（適用於 ngrok）
                ip = headers.get('X-Forwarded-For', '').split(',')[0].strip()
                if not ip:
                    # 如果沒有 X-Forwarded-For，則使用 X-Real-IP
                    ip = headers.get('X-Real-IP', '')
                if not ip:
                    # 如果都沒有，則使用 Remote-Addr
                    ip = headers.get('Remote-Addr', '')
                record.ip = ip if ip else 'unknown'
            else:
                record.ip = 'unknown'
        except Exception:
            record.ip = 'unknown'
        return True

# 建立 logs 目錄（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 設定日誌檔案名稱（使用當前日期）
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f'logs/candidates_label_{current_date}.log'

# 建立自定義的格式化器
class SafeFormatter(Formatter):
    def format(self, record):
        # 確保 record 有 ip 屬性
        if not hasattr(record, 'ip'):
            record.ip = 'unknown'
        return super().format(record)

formatter = SafeFormatter(
    fmt='%(asctime)s [%(ip)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 設定檔案處理器
file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
file_handler.setFormatter(formatter)

# 設定控制台處理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 設定根 logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 移除所有現有的處理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 添加自定義的處理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 添加 IP 過濾器
ip_filter = IPFilter()
logger.addFilter(ip_filter)

# 設定資料庫連線（標記資料）
LABELING_DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
SOURCE_DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash"

def init_database_engines():
    """初始化資料庫引擎"""
    try:
        # 建立兩個資料庫的連線引擎，添加連接池設定
        labeling_engine = create_engine(
            LABELING_DB_URL,
            pool_size=5,  # 連接池大小
            max_overflow=10,  # 最大溢出連接數
            pool_pre_ping=True,  # 連接前檢查
            pool_recycle=3600,  # 連接回收時間（秒）
            pool_timeout=30  # 連接超時時間
        )
        source_engine = create_engine(
            SOURCE_DB_URL,
            pool_size=5,  # 連接池大小
            max_overflow=10,  # 最大溢出連接數
            pool_pre_ping=True,  # 連接前檢查
            pool_recycle=3600,  # 連接回收時間（秒）
            pool_timeout=30  # 連接超時時間
        )
        
        # 測試連接
        with labeling_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        with source_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        logger.info("資料庫連接初始化成功")
        return labeling_engine, source_engine
        
    except Exception as e:
        logger.error(f"資料庫連接初始化失敗: {str(e)}")
        raise

def initialize_app():
    """初始化應用程式，只在第一次載入時執行"""
    if 'app_initialized' not in st.session_state:
        # 記錄程式啟動
        logger.info("程式啟動")
        
        # 初始化資料庫引擎
        try:
            labeling_engine, source_engine = init_database_engines()
            
            # 確保有 system_settings 資料表
            try:
                with labeling_engine.begin() as conn:
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS system_settings (
                            key VARCHAR(50) PRIMARY KEY,
                            value TEXT,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
            except Exception as e:
                logger.error(f"建立 system_settings 資料表失敗: {str(e)}")
                st.error(f"資料庫初始化失敗: {str(e)}")
                st.stop()
            
            # 將引擎存儲到 session state
            st.session_state.labeling_engine = labeling_engine
            st.session_state.source_engine = source_engine
            st.session_state.app_initialized = True
            
        except Exception as e:
            st.error(f"無法連接到資料庫：{str(e)}")
            st.error("請檢查 PostgreSQL 服務是否正在運行，以及連接設定是否正確")
            st.stop()

# 初始化應用程式
initialize_app()

# 從 session state 獲取引擎
labeling_engine = st.session_state.labeling_engine
source_engine = st.session_state.source_engine

# 新增 ScamDetectorMemmap 類別
class ScamDetectorMemmap:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 8192,
                 embeddings_dir: str = "embeddings_data",
                 memory_optimized: bool = True):
        """
        初始化詐騙檢測器 (使用 memmap 存儲)
        """
        self.db_url = db_url
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.memory_optimized = memory_optimized
        self.engine = None
        self.model = None
        self.embeddings_array = None
        
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
        self._init_embeddings_memmap()
        
    def _init_db_connection(self):
        """初始化資料庫連接"""
        try:
            self.engine = create_engine(
                self.db_url,
                pool_size=3,  # 連接池大小
                max_overflow=5,  # 最大溢出連接數
                pool_pre_ping=True,  # 連接前檢查
                pool_recycle=3600,  # 連接回收時間（秒）
                pool_timeout=30  # 連接超時時間
            )
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
            if not os.path.exists(self.index_file):
                raise FileNotFoundError(f"索引檔案不存在: {self.index_file}")
                
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.pos_tid_to_index = json.load(f)
                
            if not os.path.exists(self.metadata_file):
                raise FileNotFoundError(f"Metadata 檔案不存在: {self.metadata_file}")
                
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            self.embedding_dim = self.metadata['embedding_dim']
            self.total_embeddings = self.metadata['total_embeddings']
            
            logger.info(f"載入 embeddings metadata：")
            logger.info(f"  - 總記錄數: {self.total_embeddings}")
            logger.info(f"  - Embedding 維度: {self.embedding_dim}")
            
            if not os.path.exists(self.embeddings_file):
                raise FileNotFoundError(f"Embeddings 檔案不存在: {self.embeddings_file}")
                
        except Exception as e:
            logger.error(f"載入 embeddings metadata 失敗: {str(e)}")
            raise
            
    def _init_embeddings_memmap(self):
        """初始化 embeddings memmap"""
        try:
            total_records = self.total_embeddings
            
            self.embeddings_array = np.memmap(
                self.embeddings_file,
                dtype=np.float32,
                mode='r',
                shape=(total_records, self.embedding_dim)
            )
            
        except Exception as e:
            logger.error(f"初始化 embeddings memmap 失敗: {str(e)}")
            raise
            
    def _get_posts_batch(self, offset: int = 0, limit: Optional[int] = None) -> pd.DataFrame:
        """獲取批次貼文資料"""
        try:
            if limit is None:
                limit = self.batch_size
                
            valid_pos_tids = list(self.pos_tid_to_index.keys())
            
            if not valid_pos_tids:
                return pd.DataFrame()
                
            start_idx = offset
            end_idx = min(offset + limit, len(valid_pos_tids))
            batch_pos_tids = valid_pos_tids[start_idx:end_idx]
            
            if not batch_pos_tids:
                return pd.DataFrame()
                
            placeholders = ','.join([f"'{pid}'" for pid in batch_pos_tids])
            sql = f"""
                SELECT pos_tid, content, page_name, created_time
                FROM posts_deduplicated 
                WHERE pos_tid IN ({placeholders})
                ORDER BY pos_tid
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
            return df
            
        except Exception as e:
            logger.error(f"獲取貼文資料時發生錯誤: {str(e)}")
            raise
            
    def _get_embeddings_for_pos_tids_optimized(self, pos_tids: List[str], batch_size: int = 100) -> Dict[str, np.ndarray]:
        """優化版本：分批獲取指定 pos_tids 的 embeddings"""
        try:
            current_memory = get_memory_usage()
            if current_memory['percent'] > 85:
                logger.warning(f"記憶體使用過高: {current_memory['percent']:.1f}%，強制垃圾回收")
                gc.collect()
            
            result = {}
            
            for pos_tid in pos_tids:
                if pos_tid in self.pos_tid_to_index:
                    index = self.pos_tid_to_index[pos_tid]
                    result[pos_tid] = self.embeddings_array[index].copy()
                    
            return result
            
        except Exception as e:
            logger.error(f"獲取 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def search_similar_posts(self, 
                           query_text: str, 
                           limit: int = 20,
                           threshold: float = 0.3,
                           random_search: bool = False,
                           progress_callback=None) -> pd.DataFrame:
        """
        搜尋相似貼文
        
        Args:
            query_text: 查詢文字
            limit: 返回結果數量
            threshold: 相似度閾值
            random_search: 是否隨機搜尋
            progress_callback: 進度回調函數，用於即時更新進度
        Returns:
            搜尋結果 DataFrame
        """
        try:
            # 生成查詢文字的 embedding
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            device = query_embedding.device
            
            results = []
            processed = 0
            offset = 0
            
            # 取得所有 pos_tid 並根據 random_search 決定是否打亂
            valid_pos_tids = list(self.pos_tid_to_index.keys())
            total_pos_tids = len(valid_pos_tids)
            if random_search:
                random.shuffle(valid_pos_tids)
            
            while len(results) < limit and processed < total_pos_tids:
                # 取出這一批的 pos_tid
                batch_pos_tids = valid_pos_tids[offset:offset + self.batch_size]
                if not batch_pos_tids:
                    break
                    
                # 查詢這一批貼文
                placeholders = ','.join([f"'{pid}'" for pid in batch_pos_tids])
                sql = f"""
                    SELECT pos_tid, content, page_name, created_time
                    FROM posts_deduplicated 
                    WHERE pos_tid IN ({placeholders})
                    ORDER BY pos_tid
                """
                with self.engine.connect() as conn:
                    df = pd.read_sql_query(text(sql), conn)
                if df.empty:
                    break
                    
                # 取得這批的 embeddings
                embeddings_dict = self._get_embeddings_for_pos_tids_optimized(batch_pos_tids)
                for _, row in df.iterrows():
                    pos_tid = row['pos_tid']
                    if pos_tid not in embeddings_dict:
                        continue
                    # 計算相似度
                    content_emb = embeddings_dict[pos_tid]
                    from torch import tensor
                    content_tensor = tensor(content_emb, device=device).unsqueeze(0)
                    similarity = util.cos_sim(content_tensor, query_embedding).squeeze().cpu().numpy()
                    similarity_score = float(similarity)
                    if similarity_score >= threshold:
                        result_row = row.copy()
                        result_row['similarity_score'] = similarity_score
                        results.append(result_row)
                        if len(results) >= limit:
                            break
                            
                processed += len(batch_pos_tids)
                offset += len(batch_pos_tids)
                
                # 記錄進度
                progress_msg = f"已處理 {processed} 筆，找到 {len(results)} 筆符合的結果"
                logger.info(progress_msg)
                
                # 如果提供了進度回調函數，則調用它
                if progress_callback:
                    progress_callback({
                        'processed': processed,
                        'total': total_pos_tids,
                        'found': len(results),
                        'message': progress_msg
                    })
            
            # 搜尋完成，回傳結果
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('similarity_score', ascending=False)
                return results_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"相似貼文搜尋時發生錯誤: {str(e)}")
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
        try:
            if hasattr(self, 'embeddings_array') and self.embeddings_array is not None:
                del self.embeddings_array
                self.embeddings_array = None
                logger.info("已清理 memmap 資源")
            
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                logger.info("已清理模型資源")
                
            if hasattr(self, 'engine') and self.engine is not None:
                self.engine.dispose()
                self.engine = None
                logger.info("已清理資料庫連接")
                
        except Exception as e:
            logger.warning(f"清理資源時發生警告: {str(e)}")
            
    def cleanup(self):
        """手動清理資源"""
        self.__del__()

def get_memory_usage():
    """獲取當前記憶體使用情況"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

# --- 獲取所有群組編號 ---
@st.cache_data
def get_all_group_ids() -> list:
    """從資料庫獲取所有不重複的群組編號"""
    query = "SELECT DISTINCT group_id FROM candidates ORDER BY group_id"
    result = pd.read_sql(query, labeling_engine)
    return result['group_id'].tolist()

# --- 載入資料的函數（帶快取） ---
@st.cache_data
def load_data_from_db(group_id: int) -> pd.DataFrame:
    """從sql載入該group的data"""
    query = f"SELECT * FROM candidates WHERE group_id = {group_id}"
    logger.info(f"🔄 載入群組 {group_id} 的sql資料")
    return pd.read_sql(query, labeling_engine)

def get_current_data(group_id: int) -> pd.DataFrame:
    """智慧取得當前資料"""
    
    # 如果群組改變，強制重新載入並重置題號
    if st.session_state.current_group != group_id:
        st.session_state.current_group = group_id
        st.session_state.need_update = False
        st.session_state.label_index = 0  # 重置題號為0 (第1題)
        load_data_from_db.clear()  # 清除舊群組的快取
        # 載入新群組資料
        db = load_data_from_db(group_id)
        # 計算並設置到最新進度
        latest_index = get_latest_progress(db)
        st.session_state.label_index = latest_index
        logger.info(f"🔄 切換到群組 {group_id}，題號導向到第{latest_index+1}題")
        st.success(f"已恢復進度到第{latest_index+1}題")
        return db
    
    # 檢查是否為導航動作且需要更新
    is_navigation = st.session_state.get('just_navigated', False)
    if is_navigation and st.session_state.need_update:
        logger.info("📥 導航時檢測到資料需要更新，重新載入...")
        load_data_from_db.clear()  # 清除快取
        st.session_state.need_update = False
        st.session_state.just_navigated = False
        db = load_data_from_db(group_id)
        return db
    
    # 重置導航標記
    if st.session_state.get('just_navigated', False):
        st.session_state.just_navigated = False
    
    # 其他情況使用快取
    return load_data_from_db(group_id)

# --- 儲存標記結果（只更新資料庫） ---
def save_label_only(pos_tid: str, label: str, note: str, group_id: int) -> None:
    """儲存到資料庫，如果是關鍵字搜尋的結果(group_id=999)且不存在則新增記錄"""
    # 先檢查貼文是否存在
    check_sql = "SELECT COUNT(*) FROM candidates WHERE pos_tid = :pos_tid"
    
    with labeling_engine.begin() as conn:
        result = conn.execute(text(check_sql), {"pos_tid": pos_tid})
        exists = result.scalar() > 0
        
        if group_id == 999 and not exists:
            # 如果是關鍵字搜尋且貼文不存在，則從原始資料庫獲取內容並新增記錄
            try:
                # 先從原始資料庫獲取貼文內容
                source_query = "SELECT pos_tid, content FROM posts_deduplicated WHERE pos_tid = :pos_tid"
                with source_engine.connect() as source_conn:
                    source_result = source_conn.execute(text(source_query), {"pos_tid": pos_tid})
                    post_data = source_result.fetchone()
                    
                    if post_data is None:
                        st.error(f"在原始資料庫中找不到貼文：{pos_tid}")
                        return
                    
                    # 插入到標記資料庫
                    insert_sql = """
                        INSERT INTO candidates (pos_tid, content, group_id, label, note)
                        VALUES (:pos_tid, :content, :group_id, :label, :note)
                    """
                    conn.execute(text(insert_sql), {
                        "pos_tid": pos_tid,
                        "content": post_data.content,
                        "group_id": group_id,
                        "label": label,
                        "note": note
                    })
                    logger.info(f"📝 新增關鍵字搜尋結果到資料庫：{pos_tid}")
            except Exception as e:
                logger.error(f"❌ 新增記錄失敗：{str(e)}")
                st.error(f"無法新增記錄：{str(e)}")
                return
        else:
            # 更新現有記錄
            update_sql = """
                UPDATE candidates
                SET label = :label, note = :note
                WHERE pos_tid = :pos_tid
            """
            result = conn.execute(text(update_sql), {
                "label": label,
                "note": note,
                "pos_tid": pos_tid
            })
            if result.rowcount == 0 and group_id != 999:
                st.warning(f"警告：沒有找到 pos_tid = {pos_tid} 的記錄")
    
    if group_id != 999:
        logger.info(f"💾 儲存標記：{pos_tid} -> {label} from group {group_id} 第{st.session_state.label_index+1}題")
    else:
        logger.info(f"🔑 從關鍵字搜尋儲存標記：{pos_tid} -> {label} from group {group_id}")
    
    # 標記需要更新，但不立即載入
    st.session_state.need_update = True

# --- 顯示一筆貼文進行標記 ---
def show_labeling_ui(group_id: int) -> None:
    index = st.session_state.label_index
    row = df.iloc[index]
    st.markdown(f"### 目前第 {index + 1} / {len(df)} 筆")
    st.markdown(f"**pos_tid：** `{row['pos_tid']}`")
    st.text_area("貼文內容", row["content"], height=400, disabled=False)

    # 顯示當前標記狀態
    current_label = row.get('label')
    if pd.isna(current_label) or current_label is None:
        current_label = '尚未判斷'
    if current_label != '尚未判斷':
        st.info(f"當前標記：{current_label}")

    # 顯示更新狀態（除錯用）
    #if st.session_state.need_update:
    #    st.warning("資料按上下題會自動更新")

    # 備註欄位
    note = st.text_input("備註（可選）", value=row.get('note', ''))

    # 手動跳轉題號
    st.markdown("---")
    st.markdown("**跳轉到第幾題**")
    col_nav1, col_nav2, col_nav3 = st.columns([2, 1, 2])
    with col_nav1:
        target_question = st.number_input(
            "編號", 
            min_value=1, 
            max_value=len(df), 
            value=index + 1,
            key="jump_to_question",
            label_visibility="collapsed"
        )
    with col_nav2:
        if st.button("🎯 跳轉", type="secondary"):
            st.session_state.label_index = target_question - 1
            st.rerun()
    with col_nav3:
        if st.button("📍 移動到未完成的題目", type="secondary"):
            unlabeled_mask = df['label'].isna() | (df['label'] == '尚未判斷') | (df['label'] == '') | df['label'].isnull()
            unlabeled_indices = df[unlabeled_mask].index.tolist()
            if unlabeled_indices:
                st.session_state.label_index = min(unlabeled_indices)
                st.rerun()
            else:
                st.info("所有題目都已完成！")

    # 按鈕區域
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⬅️ 上一題", disabled = index<=0, key="labeling_prev"):
            st.session_state.just_navigated = True
            st.session_state.label_index -= 1
            st.rerun()
    
    with col2:
        if st.button("✅ 是", type="secondary", disabled=(index == len(df)), key="labeling_yes"):
            save_label_only(row["pos_tid"], "是", note, group_id)
            # 防止超出範圍
            if not index >= (len(df) - 1): # if index < 799:
                st.session_state.label_index += 1
            st.rerun()
    
    with col3:
        if st.button("❌ 否", type="secondary", disabled=(index == len(df)), key="labeling_no"):
            save_label_only(row["pos_tid"], "否", note, group_id)
            # 防止超出範圍
            if not index >= (len(df) - 1):
                st.session_state.label_index += 1
            st.rerun()
    
    with col4:
        if st.button("下一題 ➡️", disabled = index >= (len(df) - 1), key="labeling_next"):
            st.session_state.just_navigated = True
            st.session_state.label_index += 1
            st.rerun()

    # 顯示進度
    total = len(df)
    labeled = len(df[df['label'].isin(['是', '否'])])
    if labeled == total:
        st.success("🎉 本組貼文已全部標記完畢！")
    st.progress(labeled / total)
    st.caption(f"已完成：{labeled}/{total} 題")

def get_latest_progress(df: pd.DataFrame) -> int:
    """計算當前最新進度（下一個未標記的題目索引）"""
    # 找出所有未標記的題目
    unlabeled_mask = df['label'].isna() | (df['label'] == '尚未判斷') | (df['label'] == '') | df['label'].isnull()
    unlabeled_indices = df[unlabeled_mask].index.tolist()
    
    if unlabeled_indices:
        # 回傳第一個未標記題目的索引
        return unlabeled_indices[0]
    else:
        # 全部標記完畢，回傳最後一題
        return len(df) - 1

def show_scam_posts_view() -> None:
    """顯示所有被標記為詐騙的貼文"""
    st.markdown("### 📱 詐騙貼文瀏覽")
    
    # 初始化 session state 用於跳轉到相似搜尋
    if 'jump_to_similar_search' not in st.session_state:
        st.session_state.jump_to_similar_search = False
    if 'similar_search_content' not in st.session_state:
        st.session_state.similar_search_content = ""
    if 'auto_switch_to_similar' not in st.session_state:
        st.session_state.auto_switch_to_similar = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "tab1"
    
    # 取得所有被標記為詐騙的貼文
    query = """
        SELECT pos_tid, content, label, note, group_id
        FROM candidates 
        WHERE label = '是'
        ORDER BY pos_tid DESC
    """
    scam_posts = pd.read_sql(query, labeling_engine)
    
    if len(scam_posts) == 0:
        st.info("目前還沒有被標記為詐騙的貼文")
        return
    
    # 顯示貼文數量
    st.caption(f"共找到 {len(scam_posts)} 則詐騙貼文")
    
    # 顯示每則貼文
    for idx, post in scam_posts.iterrows():
        with st.container():
            st.markdown("---")
            
            # 貼文標題和按鈕區域
            col_title, col_button = st.columns([3, 1])
            with col_title:
                st.markdown(f"**貼文 ID：** `{post['pos_tid']}`")
            with col_button:
                if st.button("🔍 尋找類似", key=f"find_similar_{post['pos_tid']}", help="點擊尋找與此貼文相似的貼文"):
                    st.session_state.jump_to_similar_search = True
                    st.session_state.similar_search_content = post['content']
                    st.session_state.auto_switch_to_similar = True
                    st.session_state.current_tab = "🔍 相似貼文搜尋"  # 直接設置要跳轉的分頁
                    st.rerun()
            
            # 貼文內容
            st.text_area("貼文內容", post['content'], height=200, disabled=True, 
                        label_visibility="collapsed", 
                        key=f"scam_posts_{idx}_{post['pos_tid']}")
            
            # 貼文資訊
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"群組：{post['group_id']}")
            with col2:
                if pd.notna(post['note']) and post['note']:
                    st.caption(f"備註：{post['note']}")
    
    # 如果點擊了尋找類似按鈕，顯示跳轉提示
    if st.session_state.jump_to_similar_search:
        st.success("✅ 已準備跳轉到相似貼文搜尋頁面")
        st.info("💡 請切換到「🔍 相似貼文搜尋」分頁查看結果")
        
        # 重置跳轉狀態
        st.session_state.jump_to_similar_search = False

def show_word_analysis() -> None:
    """顯示詞彙分析結果"""
    st.markdown("### 📊 詞彙分析")
    
    # 從資料庫讀取上次生成時間
    with labeling_engine.connect() as conn:
        result = conn.execute(text("SELECT value, updated_at FROM system_settings WHERE key = 'last_word_analysis_time'"))
        row = result.fetchone()
        last_generation_time = row[0] if row else None
    
    # 顯示上次生成時間
    if last_generation_time:
        st.info(f"上次生成時間：{last_generation_time}")
    
    # 手動生成按鈕
    if st.button("🔄 生成詞彙分析圖表", type="primary", key="generate_word_analysis"):
        try:
            
            # 執行分析程式
            subprocess.run(['python', 'analyze_scam_posts.py'], check=True)
            
            # 更新資料庫中的生成時間
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with labeling_engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO system_settings (key, value, updated_at)
                        VALUES ('last_word_analysis_time', :value, CURRENT_TIMESTAMP)
                        ON CONFLICT (key) DO UPDATE
                        SET value = :value, updated_at = CURRENT_TIMESTAMP
                    """),
                    {"value": current_time}
                )
            
            st.success("✅ 詞彙分析圖表生成成功！")
            st.rerun()
            
        except Exception as e:
            st.error(f"生成圖表時發生錯誤：{str(e)}")
            return
    
    # 顯示圖表
    try:
        # 檢查圖表檔案是否存在
        if os.path.exists('word_frequency.png') and os.path.exists('wordcloud.png'):
            # 顯示詞頻分析圖
            st.markdown("#### 📈 詞頻分析圖")
            st.image('word_frequency.png', use_container_width=True)
            
            # 顯示文字雲圖
            st.markdown("#### ☁️ 文字雲")
            st.image('wordcloud.png', use_container_width=True)
        else:
            st.info("請點擊上方按鈕生成詞彙分析圖表")
    except Exception as e:
        st.error(f"讀取圖表時發生錯誤：{str(e)}")

def show_post_search() -> None:
    """根據 pos_tid 查詢特定貼文"""
    
    # 建立分頁
    search_tab1, search_tab2 = st.tabs(["📝 標記資料庫查詢", "🔍 原始資料庫查詢"])
    
    # 初始化共享的搜尋 ID
    if 'shared_search_id' not in st.session_state:
        st.session_state.shared_search_id = ""
    
    with search_tab1:
        st.text("非全貼文查詢，需要有標記過是或否的資料才可以查詢")
        
        # 初始化編輯狀態
        if 'has_unsaved_changes' not in st.session_state:
            st.session_state.has_unsaved_changes = False
        if 'edited_label' not in st.session_state:
            st.session_state.edited_label = None
        if 'edited_note' not in st.session_state:
            st.session_state.edited_note = None
        if 'current_post_id' not in st.session_state:
            st.session_state.current_post_id = None
        
        # 搜尋輸入框
        pos_tid = st.text_input("請輸入貼文 ID (pos_tid)", 
                               value=st.session_state.shared_search_id,
                               key="labeling_search")
        
        # 更新共享的搜尋 ID
        if pos_tid != st.session_state.shared_search_id:
            st.session_state.shared_search_id = pos_tid
        
        if pos_tid:
            # 查詢貼文
            query = """
                SELECT pos_tid, content, label, note, group_id
                FROM candidates 
                WHERE pos_tid = :pos_tid
            """
            result = pd.read_sql(text(query), labeling_engine, params={"pos_tid": pos_tid})
            
            if len(result) == 0:
                st.warning(f"找不到 ID 為 {pos_tid} 的貼文")
                st.info("💡 您可以切換到「原始資料庫查詢」分頁查看此貼文是否在原始資料庫中")
            else:
                post = result.iloc[0]
                
                # 如果是新貼文，重置編輯狀態
                if st.session_state.current_post_id != pos_tid:
                    st.session_state.current_post_id = pos_tid
                    st.session_state.has_unsaved_changes = False
                    st.session_state.edited_label = post['label']
                    st.session_state.edited_note = post['note']
                
                # 顯示貼文內容
                st.markdown("---")
                st.markdown(f"**貼文 ID：** `{post['pos_tid']}`")
                # 貼文內容（改為純文字顯示）
                st.text_area("貼文內容", post['content'], height=200, disabled=True, label_visibility="collapsed", key=f"scam_posts_search_{post['pos_tid']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
                
                # 編輯區域
                st.markdown("### 編輯標記")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption(f"群組：{post['group_id']}")
                    # 標記選擇
                    new_label = st.radio(
                        "標記",
                        options=["是", "否", "尚未判斷"],
                        index=["是", "否", "尚未判斷"].index(st.session_state.edited_label if st.session_state.edited_label else "尚未判斷"),
                        key=f"label_edit_{post['pos_tid']}"
                    )
                    # 只有當實際值改變時才標記為未存檔
                    if new_label != post['label']:
                        st.session_state.edited_label = new_label
                        st.session_state.has_unsaved_changes = True
                    elif new_label == post['label'] and st.session_state.edited_label != post['label']:
                        st.session_state.edited_label = new_label
                        st.session_state.has_unsaved_changes = False
                
                with col2:
                    # 備註編輯
                    new_note = st.text_area(
                        "備註",
                        value=st.session_state.edited_note if pd.notna(st.session_state.edited_note) else "",
                        key=f"note_edit_{post['pos_tid']}"
                    )
                    # 只有當實際值改變時才標記為未存檔
                    if new_note != (post['note'] if pd.notna(post['note']) else ""):
                        st.session_state.edited_note = new_note
                        st.session_state.has_unsaved_changes = True
                    elif new_note == (post['note'] if pd.notna(post['note']) else "") and st.session_state.edited_note != post['note']:
                        st.session_state.edited_note = new_note
                        st.session_state.has_unsaved_changes = False
                
                # 存檔按鈕
                col_save1, col_save2 = st.columns([1, 3])
                with col_save1:
                    if st.button("💾 儲存更改", type="primary", disabled=not st.session_state.has_unsaved_changes):
                        save_label_only(post['pos_tid'], st.session_state.edited_label, st.session_state.edited_note, post['group_id'])
                        st.session_state.has_unsaved_changes = False
                        st.success("✅ 已儲存更改")
                        st.rerun()
                
                # 顯示未存檔提醒
                if st.session_state.has_unsaved_changes:
                    st.warning("⚠️ 您有未存檔的更改！")
    
    with search_tab2:
        st.text("查詢原始資料庫中的所有貼文")
        
        # 搜尋輸入框（使用共享的搜尋 ID）
        source_pos_tid = st.text_input("請輸入貼文 ID (pos_tid)", 
                                      value=st.session_state.shared_search_id,
                                      key="source_search")
        
        # 更新共享的搜尋 ID
        if source_pos_tid != st.session_state.shared_search_id:
            st.session_state.shared_search_id = source_pos_tid
        
        if source_pos_tid:
            # 查詢原始資料庫
            query = """
                SELECT pos_tid, content, created_time, date, post_type, page_name, 
                       reaction_all, comment_count, share_count
                FROM posts_deduplicated 
                WHERE pos_tid = :pos_tid
            """
            try:
                result = pd.read_sql(text(query), source_engine, params={"pos_tid": source_pos_tid})
                
                if len(result) == 0:
                    st.warning(f"在原始資料庫中找不到 ID 為 {source_pos_tid} 的貼文")
                else:
                    post = result.iloc[0]
                    
                    # 顯示貼文內容
                    st.markdown("---")
                    st.markdown(f"**貼文 ID：** `{post['pos_tid']}`")
                    st.text_area("貼文內容", post['content'], height=200, disabled=True, label_visibility="collapsed", key=f"source_posts_search_{post['pos_tid']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
                    
                    # 顯示貼文資訊
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"建立時間：{post['created_time']}")
                        st.caption(f"日期：{post['date']}")
                    with col2:
                        st.caption(f"貼文類型：{post['post_type']}")
                        st.caption(f"頁面名稱：{post['page_name']}")
                    with col3:
                        st.caption(f"互動數：{post['reaction_all']}")
                        st.caption(f"留言數：{post['comment_count']}")
                        st.caption(f"分享數：{post['share_count']}")
                    
                    # 檢查是否已在標記資料庫中
                    check_query = "SELECT label FROM candidates WHERE pos_tid = :pos_tid"
                    check_result = pd.read_sql(text(check_query), labeling_engine, params={"pos_tid": source_pos_tid})
                    
                    if len(check_result) > 0:
                        st.info(f"此貼文已在標記資料庫中，當前標記：{check_result.iloc[0]['label']}")
                    else:
                        st.info("此貼文尚未加入標記資料庫")
                        
                        # 提供快速加入標記資料庫的按鈕
                        if st.button("📝 加入標記資料庫", type="primary"):
                            try:
                                # 插入到標記資料庫
                                insert_sql = """
                                    INSERT INTO candidates (pos_tid, content, group_id, label, note)
                                    VALUES (:pos_tid, :content, 999, '尚未判斷', '')
                                """
                                with labeling_engine.begin() as conn:
                                    conn.execute(text(insert_sql), {
                                        "pos_tid": post['pos_tid'],
                                        "content": post['content']
                                    })
                                st.success("✅ 已成功加入標記資料庫！")
                                st.rerun()
                            except Exception as e:
                                st.error(f"加入標記資料庫失敗：{str(e)}")
                
            except Exception as e:
                st.error(f"查詢原始資料庫時發生錯誤：{str(e)}")

def show_keyword_search() -> None:
    """顯示關鍵字搜尋模式的介面"""
    
    # 初始化分頁相關的 session state
    if 'search_page' not in st.session_state:
        st.session_state.search_page = 0
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_keywords' not in st.session_state:
        st.session_state.search_keywords = None
    if 'exclude_keywords' not in st.session_state:
        st.session_state.exclude_keywords = None
    if 'search_logic' not in st.session_state:
        st.session_state.search_logic = None
    if 'search_table' not in st.session_state:
        st.session_state.search_table = 'posts_deduplicated'
    if 'label_message' not in st.session_state:
        st.session_state.label_message = None
    if 'label_message_pos_tid' not in st.session_state:
        st.session_state.label_message_pos_tid = None
    
    # 資料表選擇
    table_options = {
        'posts_deduplicated': '去重後貼文 (posts_deduplicated)',
        'posts': '原始貼文 (posts)'
    }
    
    selected_table = st.selectbox(
        "選擇要搜尋的資料表",
        options=list(table_options.keys()),
        format_func=lambda x: table_options[x],
        index=list(table_options.keys()).index(st.session_state.search_table),
        help="選擇要搜尋的資料表。posts_deduplicated 是去重後的資料，posts 是原始資料"
    )
    
    # 更新 session state
    if selected_table != st.session_state.search_table:
        st.session_state.search_table = selected_table
        # 清除之前的搜尋結果
        st.session_state.search_results = None
        st.session_state.search_page = 0
    
    # 關鍵字輸入區域
    keywords_input = st.text_area(
        "請輸入關鍵字(每行一個) ",
        value="\n".join(st.session_state.search_keywords) if st.session_state.search_keywords else "",
        help="每行輸入一個關鍵字，系統會根據選擇的邏輯進行搜尋"
    )
    
    exclude_keywords_input = st.text_area(
        "請輸入要排除的關鍵字(每行一個)",
        value="\n".join(st.session_state.exclude_keywords) if st.session_state.exclude_keywords else "",
        help="每行輸入一個要排除的關鍵字，符合這些關鍵字的貼文將不會顯示"
    )

    st.text("(最多500筆結果) \n (時間最長需要30秒) \n (貼文出現的順序是隨機的)")
    
    # 將輸入轉換為關鍵字列表
    keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
    exclude_keywords = [kw.strip() for kw in exclude_keywords_input.split('\n') if kw.strip()]
    
    # 搜尋邏輯選擇
    search_logic = st.radio(
        "搜尋邏輯",
        options=["OR", "AND"],
        index=0 if st.session_state.search_logic != "AND" else 1,
        help="OR：符合任一關鍵字即顯示\nAND：必須符合所有關鍵字才顯示"
    )
    
    # 搜尋按鈕
    if st.button("🔍 開始搜尋", type="primary", disabled=not keywords):
        try:
            # 執行搜尋
            results_df = fetch_candidate_posts(
                source_engine=source_engine,
                keywords=keywords,
                exclude_keywords=exclude_keywords,  # 新增排除關鍵字參數
                limit=500,  # 先取得較多結果，但分頁顯示
                group_count=1,  # 搜尋模式下不需要分組
                search_logic=search_logic,
                table_name=st.session_state.search_table  # 使用選擇的資料表
            )
            
            if len(results_df) == 0:
                st.warning("沒有找到符合條件的貼文")
                st.session_state.search_results = None
                st.session_state.search_page = 0
                return
            
            # 儲存搜尋結果和參數到 session state
            st.session_state.search_results = results_df
            st.session_state.search_keywords = keywords
            st.session_state.exclude_keywords = exclude_keywords
            st.session_state.search_logic = search_logic
            st.session_state.search_page = 0  # 重置頁碼
            
            st.success(f"找到 {len(results_df)} 則符合條件的貼文")
            st.rerun()
            
        except Exception as e:
            st.error(f"搜尋時發生錯誤：{str(e)}")
    
    # 如果有搜尋結果，顯示分頁內容
    if st.session_state.search_results is not None:
        num_per_page = 20
        df = st.session_state.search_results
        total_pages = (len(df) + (num_per_page - 1)) // num_per_page  # 向上取整，計算總頁數
        
        # 顯示分頁資訊
        st.markdown(f"---\n#### 搜尋結果（第 {st.session_state.search_page + 1} 頁，共 {total_pages} 頁）")
        st.caption(f"搜尋資料表：{table_options[st.session_state.search_table]}")
        
        # 計算當前頁的資料範圍
        start_idx = st.session_state.search_page * num_per_page
        end_idx = min(start_idx + num_per_page, len(df))
        
        # 顯示當前頁的資料
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            with st.container():
                st.markdown("---")
                # 貼文標題
                st.markdown(f"**貼文 ID：** `{row['pos_tid']}`")
                # 貼文內容
                st.text_area("貼文內容", row['content'], height=200, disabled=True, label_visibility="collapsed", key=f"keyword_search_{st.session_state.search_page}_{idx}")
                
                # 標記區域
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("✅ 是", key=f"keyword_yes_{st.session_state.search_page}_{idx}_{row['pos_tid']}"):
                        try:
                            save_label_only(row['pos_tid'], "是", "", 999)
                            st.session_state.label_message = "已標記為「是」"
                            st.session_state.label_message_pos_tid = row['pos_tid']
                            st.rerun()
                        except Exception as e:
                            st.error(f"標記失敗：{str(e)}")
                with col2:
                    if st.button("❌ 否", key=f"keyword_no_{st.session_state.search_page}_{idx}_{row['pos_tid']}"):
                        try:
                            save_label_only(row['pos_tid'], "否", "", 999)
                            st.session_state.label_message = "已標記為「否」"
                            st.session_state.label_message_pos_tid = row['pos_tid']
                            st.rerun()
                        except Exception as e:
                            st.error(f"標記失敗：{str(e)}")
                with col3:
                    # 顯示當前標記狀態
                    current_label = row.get('label')
                    if pd.notna(current_label) and current_label:
                        st.info(f"當前標記：{current_label}")
                
                # 顯示標記訊息（如果有的話）
                if st.session_state.label_message and st.session_state.label_message_pos_tid == row['pos_tid']:
                    st.success(st.session_state.label_message, icon="✅" if "是" in st.session_state.label_message else "❌")
                    # 清除訊息，避免重複顯示
                    st.session_state.label_message = None
                    st.session_state.label_message_pos_tid = None
        
        # 分頁導航按鈕
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ 上一頁", disabled=st.session_state.search_page <= 0, key="keyword_prev_page"):
                st.session_state.search_page -= 1
                # 使用 JavaScript 跳轉到頁面頂部
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()
        with col2:
            # 頁碼輸入框
            target_page = st.number_input(
                "前往頁碼",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.search_page + 1,
                label_visibility="collapsed",
                key="keyword_page_input"
            )
            # 當頁碼改變時跳轉
            if target_page != st.session_state.search_page + 1:
                st.session_state.search_page = target_page - 1
                # 使用 JavaScript 跳轉到頁面頂部
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()
        with col3:
            if st.button("下一頁 ➡️", disabled=st.session_state.search_page >= total_pages - 1, key="keyword_next_page"):
                st.session_state.search_page += 1
                # 使用 JavaScript 跳轉到頁面頂部
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()

def show_similar_posts_search():
    """顯示相似貼文搜尋模式的介面"""
    
    # 初始化 session state
    if 'similar_search_results' not in st.session_state:
        st.session_state.similar_search_results = None
    if 'similar_search_query' not in st.session_state:
        st.session_state.similar_search_query = None
    if 'similar_search_page' not in st.session_state:
        st.session_state.similar_search_page = 0
    if 'similar_label_message' not in st.session_state:
        st.session_state.similar_label_message = None
    if 'similar_label_message_pos_tid' not in st.session_state:
        st.session_state.similar_label_message_pos_tid = None
    if 'similar_search_content' not in st.session_state:
        st.session_state.similar_search_content = ""
    # 新增進度顯示相關的 session state
    if 'search_progress' not in st.session_state:
        st.session_state.search_progress = None
    if 'search_progress_message' not in st.session_state:
        st.session_state.search_progress_message = ""
    # 新增搜尋進程相關的 session state
    if 'search_process' not in st.session_state:
        st.session_state.search_process = None
    if 'search_in_progress' not in st.session_state:
        st.session_state.search_in_progress = False
    
    st.markdown("### 🔍 相似貼文搜尋")
    st.markdown("輸入一段文字，系統會找到語意相似的貼文")
    
    # 搜尋參數設定
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("最大結果數量", min_value=5, max_value=100, value=20, step=5)
    with col2:
        threshold = st.slider("相似度閾值", min_value=0.1, max_value=0.9, value=0.7, step=0.01, help="數值越高，結果越相似")
    
    # 新增隨機搜尋選項
    random_search = st.checkbox("隨機搜尋 (Random Search)", value=False, key="similar_random_search")
    
    # 停止搜尋按鈕
    if st.session_state.search_in_progress:
        if st.button("⏹️ 停止搜尋", type="secondary", key="stop_search_button"):
            if st.session_state.search_process:
                st.session_state.search_process.stop_search()
            st.session_state.search_in_progress = False
            st.success("已發送停止搜尋指令")
            st.rerun()
    
    # 查詢文字輸入
    query_text = st.text_area(
        "請輸入要搜尋的文字",
        value=st.session_state.similar_search_content if st.session_state.similar_search_content else 
              (st.session_state.similar_search_query if st.session_state.similar_search_query else ""),
        height=100,
        help="輸入任何文字，系統會找到語意相似的貼文"
    )
    
    # 如果從詐騙貼文瀏覽跳轉過來，自動執行搜尋
    if st.session_state.similar_search_content and not st.session_state.similar_search_query:
        st.session_state.similar_search_query = st.session_state.similar_search_content
        # 清除跳轉內容，避免重複執行
        st.session_state.similar_search_content = ""
        
        # 設定搜尋狀態
        st.session_state.search_in_progress = True
        st.session_state.search_progress = None
        st.session_state.search_progress_message = ""
        
        # 初始化搜尋進程
        if st.session_state.search_process is None:
            st.session_state.search_process = SimilarSearchProcess()
        
        # 啟動搜尋
        st.session_state.search_process.start_search(
            query_text=st.session_state.similar_search_query,
            limit=limit,
            threshold=0.7,  # 固定使用 0.7 閾值
            random_search=random_search
        )
        
        # 顯示搜尋狀態並重新載入頁面
        st.info("正在準備自動搜尋...")
        st.rerun()
    
    # 手動搜尋按鈕
    if st.button("🔍 開始搜尋", type="primary", disabled=not query_text.strip(), key="similar_search_button"):
        # 設定搜尋狀態
        st.session_state.search_in_progress = True
        st.session_state.similar_search_query = query_text
        st.session_state.similar_search_results = None  # 清除之前的結果
        st.session_state.search_progress = None
        st.session_state.search_progress_message = ""
        
        # 初始化搜尋進程
        if st.session_state.search_process is None:
            st.session_state.search_process = SimilarSearchProcess()
        
        # 啟動搜尋
        st.session_state.search_process.start_search(
            query_text=query_text,
            limit=limit,
            threshold=threshold,
            random_search=random_search
        )
        
        # 顯示搜尋狀態並重新載入頁面
        st.info("正在準備搜尋...")
        st.rerun()
    
    # 檢查搜尋進度
    if st.session_state.search_in_progress and st.session_state.search_process:
        # 檢查進度
        progress = st.session_state.search_process.get_progress()
        if progress:
            st.session_state.search_progress = progress
            st.session_state.search_progress_message = progress['message']
        
        # 檢查結果
        result = st.session_state.search_process.get_result()
        if result:
            st.session_state.search_in_progress = False
            
            if 'error' in result:
                st.error(f"搜尋時發生錯誤：{result['error']}")
                st.session_state.similar_search_results = None
                st.session_state.similar_search_page = 0
            else:
                # 將結果轉換回 DataFrame
                if result['data']:
                    results_df = pd.DataFrame(result['data'], columns=result['columns'])
                    st.session_state.similar_search_results = results_df
                    st.session_state.similar_search_page = 0
                    st.success(f"找到 {len(results_df)} 則相似貼文")
                else:
                    st.warning("沒有找到相似的貼文，請嘗試降低相似度閾值或修改搜尋文字")
                    st.session_state.similar_search_results = None
                    st.session_state.similar_search_page = 0
            
            st.rerun()
        
        # 顯示進度
        if st.session_state.search_progress_message:
            st.info(st.session_state.search_progress_message)
        else:
            st.info("正在進行搜尋...")
        
        # 自動重新載入以更新進度
        time.sleep(0.5)
        st.rerun()
    
    # 清理資源按鈕（可選）
    if st.button("🧹 清理記憶體", key="cleanup_memory", help="如果遇到記憶體問題，可以點擊此按鈕清理資源"):
        if st.session_state.search_process:
            st.session_state.search_process.stop_search()
            st.session_state.search_process = None
            st.success("已清理記憶體資源")
            st.rerun()
        else:
            st.info("沒有需要清理的資源")
        
        # 重置搜尋狀態
        st.session_state.search_in_progress = False
        st.session_state.search_progress = None
        st.session_state.search_progress_message = ""
    
    # 如果有搜尋結果，顯示分頁內容
    if st.session_state.similar_search_results is not None:
        num_per_page = 10
        df = st.session_state.similar_search_results
        total_pages = (len(df) + (num_per_page - 1)) // num_per_page
        
        # 顯示搜尋資訊
        st.markdown(f"---\n#### 搜尋結果（第 {st.session_state.similar_search_page + 1} 頁，共 {total_pages} 頁）")
        st.caption(f"查詢文字：{st.session_state.similar_search_query}")
        st.caption(f"共找到 {len(df)} 則相似貼文")
        
        # 計算當前頁的資料範圍
        start_idx = st.session_state.similar_search_page * num_per_page
        end_idx = min(start_idx + num_per_page, len(df))
        
        # 顯示當前頁的資料
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            with st.container():
                st.markdown("---")
                
                # 相似度分數
                similarity_score = row['similarity_score']
                st.markdown(f"**相似度：** {similarity_score:.3f}")
                
                # 貼文標題
                st.markdown(f"**貼文 ID：** `{row['pos_tid']}`")
                st.caption(f"頁面：{row['page_name']} | 建立時間：{row['created_time']}")
                
                # 貼文內容
                st.text_area("貼文內容", row['content'], height=150, disabled=True, 
                           label_visibility="collapsed", 
                           key=f"similar_search_{st.session_state.similar_search_page}_{idx}")
                
                # 標記區域
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("✅ 是", key=f"similar_yes_{st.session_state.similar_search_page}_{idx}_{row['pos_tid']}"):
                        try:
                            save_label_only(row['pos_tid'], "是", "", 999)
                            st.session_state.similar_label_message = "已標記為「是」"
                            st.session_state.similar_label_message_pos_tid = row['pos_tid']
                            st.rerun()
                        except Exception as e:
                            st.error(f"標記失敗：{str(e)}")
                with col2:
                    if st.button("❌ 否", key=f"similar_no_{st.session_state.similar_search_page}_{idx}_{row['pos_tid']}"):
                        try:
                            save_label_only(row['pos_tid'], "否", "", 999)
                            st.session_state.similar_label_message = "已標記為「否」"
                            st.session_state.similar_label_message_pos_tid = row['pos_tid']
                            st.rerun()
                        except Exception as e:
                            st.error(f"標記失敗：{str(e)}")
                with col3:
                    # 檢查當前標記狀態
                    check_query = "SELECT label FROM candidates WHERE pos_tid = :pos_tid"
                    check_result = pd.read_sql(text(check_query), labeling_engine, params={"pos_tid": row['pos_tid']})
                    
                    if len(check_result) > 0:
                        current_label = check_result.iloc[0]['label']
                        if pd.notna(current_label) and current_label:
                            st.info(f"當前標記：{current_label}")
                
                # 顯示標記訊息
                if (st.session_state.similar_label_message and 
                    st.session_state.similar_label_message_pos_tid == row['pos_tid']):
                    st.success(st.session_state.similar_label_message, 
                             icon="✅" if "是" in st.session_state.similar_label_message else "❌")
                    st.session_state.similar_label_message = None
                    st.session_state.similar_label_message_pos_tid = None
        
        # 分頁導航按鈕
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ 上一頁", disabled=st.session_state.similar_search_page <= 0, key="similar_prev_page"):
                st.session_state.similar_search_page -= 1
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()
        with col2:
            target_page = st.number_input(
                "前往頁碼",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.similar_search_page + 1,
                label_visibility="collapsed",
                key="similar_page_input"
            )
            if target_page != st.session_state.similar_search_page + 1:
                st.session_state.similar_search_page = target_page - 1
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()
        with col3:
            if st.button("下一頁 ➡️", disabled=st.session_state.similar_search_page >= total_pages - 1, key="similar_next_page"):
                st.session_state.similar_search_page += 1
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()

#======================================================================================

def cleanup_database_connections():
    """清理所有資料庫連接"""
    try:
        # 清理 session state 中的引擎
        if 'labeling_engine' in st.session_state:
            st.session_state.labeling_engine.dispose()
            logger.info("已清理 labeling_engine 連接")
        if 'source_engine' in st.session_state:
            st.session_state.source_engine.dispose()
            logger.info("已清理 source_engine 連接")
            
        # 清理全域變數中的引擎（如果存在）
        if 'labeling_engine' in globals():
            labeling_engine.dispose()
            logger.info("已清理全域 labeling_engine 連接")
        if 'source_engine' in globals():
            source_engine.dispose()
            logger.info("已清理全域 source_engine 連接")
    except Exception as e:
        logger.warning(f"清理資料庫連接時發生警告: {str(e)}")

# 註冊程式結束時的清理函數
atexit.register(cleanup_database_connections)

# 新增相似貼文搜尋進程類別
class SimilarSearchProcess:
    """獨立的相似貼文搜尋進程，避免 PyTorch 與 Streamlit 衝突"""
    
    def __init__(self, embeddings_dir="embeddings_data", batch_size=32768):
        self.embeddings_dir = embeddings_dir
        self.batch_size = batch_size
        self.process = None
        self.result_queue = Queue()
        self.progress_queue = Queue()
        self.stop_event = mp.Event()
        
    def start_search(self, query_text, limit=20, threshold=0.7, random_search=False):
        """啟動搜尋進程"""
        # 停止之前的進程（如果有的話）
        self.stop_search()
        
        # 重置停止事件
        self.stop_event.clear()
        
        # 啟動新進程
        self.process = Process(
            target=self._search_worker,
            args=(
                query_text, limit, threshold, random_search,
                self.embeddings_dir, self.batch_size,
                self.result_queue, self.progress_queue, self.stop_event
            )
        )
        self.process.start()
        
    def stop_search(self):
        """停止搜尋進程"""
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
            self.process = None
            # 清空隊列
            try:
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
                while not self.progress_queue.empty():
                    self.progress_queue.get_nowait()
            except:
                pass
        
    def get_progress(self):
        """獲取進度資訊"""
        try:
            while not self.progress_queue.empty():
                progress = self.progress_queue.get_nowait()
                return progress
        except:
            pass
        return None
        
    def get_result(self):
        """獲取搜尋結果"""
        try:
            if self.process and not self.process.is_alive():
                # 進程已完成，獲取結果
                result = self.result_queue.get(timeout=1)
                self.process = None
                return result
        except:
            pass
        return None
        
    def is_running(self):
        """檢查進程是否正在運行"""
        return self.process is not None and self.process.is_alive()
        
    @staticmethod
    def _search_worker(query_text, limit, threshold, random_search, 
                      embeddings_dir, batch_size, result_queue, progress_queue, stop_event):
        """搜尋工作進程"""
        try:
            # 在子進程中導入 PyTorch 相關模組
            import numpy as np
            import json
            import os
            from sentence_transformers import SentenceTransformer, util
            from sqlalchemy import create_engine, text
            import pandas as pd
            import random
            from torch import tensor
            
            # 初始化 detector
            detector = ScamDetectorMemmap(
                embeddings_dir=embeddings_dir,
                batch_size=batch_size
            )
            
            # 執行搜尋
            results_df = detector.search_similar_posts(
                query_text=query_text,
                limit=limit,
                threshold=threshold,
                random_search=random_search,
                progress_callback=lambda progress: progress_queue.put(progress) if not stop_event.is_set() else None
            )
            
            # 檢查是否被停止
            if stop_event.is_set():
                result_queue.put({'data': [], 'columns': []})
                return
            
            # 清理資源
            detector.cleanup()
            
            # 將結果序列化並放入隊列
            if not results_df.empty:
                # 將 DataFrame 轉換為字典格式以便序列化
                result_dict = {
                    'data': results_df.to_dict('records'),
                    'columns': results_df.columns.tolist()
                }
                result_queue.put(result_dict)
            else:
                result_queue.put({'data': [], 'columns': []})
                
        except Exception as e:
            result_queue.put({'error': str(e)})
        finally:
            # 確保進程結束時清理資源
            try:
                import gc
                gc.collect()
            except:
                pass

if __name__ == '__main__':
    st.title("詐騙貼文人工標記工具")
    
    # 初始化 session state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "📝 標記模式"
    if 'auto_switch_to_similar' not in st.session_state:
        st.session_state.auto_switch_to_similar = False
    
    # 使用 selectbox 來實現分頁切換
    tab_options = ["📝 標記模式", "👀 瀏覽模式", "🔑 關鍵字搜尋", "🔍 相似貼文搜尋"]
    selected_tab = st.selectbox(
        "選擇功能分頁",
        tab_options,
        index=tab_options.index(st.session_state.current_tab),
        label_visibility="collapsed"
    )
    
    # 更新當前分頁
    if selected_tab != st.session_state.current_tab:
        st.session_state.current_tab = selected_tab
        st.rerun()
    
    # 根據選擇的分頁顯示對應內容
    if selected_tab == "📝 標記模式":
        # 動態獲取群組編號
        group_ids = get_all_group_ids()
        group_id = st.selectbox("請選擇你的群組編號 (999是關鍵字搜尋的標記)", group_ids)
        
        # --- 初始化 session state ---
        if 'label_index' not in st.session_state:
            st.session_state.label_index = 0
        if 'need_update' not in st.session_state:
            st.session_state.need_update = False
        if 'current_group' not in st.session_state:
            st.session_state.current_group = None
        
        # --- 取得資料 ---
        df = get_current_data(group_id)
        
        # --- 確保有 label/note 欄位 ---
        with labeling_engine.begin() as conn:
            conn.execute(text("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS label TEXT"))
            conn.execute(text("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS note TEXT"))
        
        # --- 啟動標記 UI ---
        show_labeling_ui(group_id)
        
    elif selected_tab == "👀 瀏覽模式":
        # 瀏覽模式的子頁籤
        subtab1, subtab2, subtab3 = st.tabs(["📱 詐騙貼文瀏覽", "📖 貼文查詢", "📊 詞彙分析"])
        
        with subtab1:
            show_scam_posts_view()
        
        with subtab2:
            show_post_search()
            
        with subtab3:
            show_word_analysis()
    
    elif selected_tab == "🔑 關鍵字搜尋":
        show_keyword_search()
        
    elif selected_tab == "🔍 相似貼文搜尋":
        # 檢查是否需要自動跳轉到相似搜尋
        if st.session_state.get('auto_switch_to_similar', False):
            st.session_state.auto_switch_to_similar = False
            st.success("✅ 已自動跳轉到相似貼文搜尋頁面")
            st.info("💡 搜尋文字已自動填入，系統將自動執行搜尋")
        
        show_similar_posts_search()