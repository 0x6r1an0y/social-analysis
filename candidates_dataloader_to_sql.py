from sqlalchemy import create_engine, text
import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import List, Optional
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_db_connection(dbname: str, user: str = 'postgres', password: str = '00000000', host: str = 'localhost') -> psycopg2.extensions.connection:
    """
    建立資料庫連接
    
    Args:
        dbname: 資料庫名稱
        user: 資料庫使用者名稱
        password: 資料庫密碼
        host: 資料庫主機位置
    
    Returns:
        psycopg2 連接物件
    """
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error(f"資料庫連接失敗: {str(e)}")
        raise

def _create_db_if_not_exists(dbname: str, user: str = 'postgres', password: str = '00000000', host: str = 'localhost') -> None:
    """
    檢查並建立資料庫（如果不存在）
    
    Args:
        dbname: 要建立的資料庫名稱
        user: 資料庫使用者名稱
        password: 資料庫密碼
        host: 資料庫主機位置
    """
    try:
        conn = _get_db_connection('postgres', user, password, host)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
        exists = cur.fetchone()
        
        if not exists:
            cur.execute(f'CREATE DATABASE {dbname}')
            logger.info(f'✅ 資料庫 {dbname} 已建立')
        else:
            logger.info(f'✅ 資料庫 {dbname} 已存在')
            
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"建立資料庫時發生錯誤: {str(e)}")
        raise

def _get_db_engine(dbname: str, user: str = 'postgres', password: str = '00000000', host: str = 'localhost', port: int = 5432) -> create_engine:
    """
    建立 SQLAlchemy 引擎
    
    Args:
        dbname: 資料庫名稱
        user: 資料庫使用者名稱
        password: 資料庫密碼
        host: 資料庫主機位置
        port: 資料庫連接埠
    
    Returns:
        SQLAlchemy 引擎物件
    """
    try:
        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        return create_engine(connection_string)
    except Exception as e:
        logger.error(f"建立資料庫引擎時發生錯誤: {str(e)}")
        raise

def fetch_candidate_posts(
    source_engine: create_engine,
    keywords: List[str],
    limit: int,
    group_count: int = 5
) -> pd.DataFrame:
    """
    從來源資料庫獲取候選貼文
    
    Args:
        source_engine: 來源資料庫引擎
        keywords: 關鍵字列表
        limit: 要獲取的貼文數量
        group_count: 要分成的組別數量
    
    Returns:
        包含候選貼文的 DataFrame
    """
    try:
        where_clause = " OR ".join([f"content ILIKE '%{kw}%'" for kw in keywords])
        sql = f"SELECT pos_tid, content FROM posts WHERE {where_clause} LIMIT {limit}"
        
        df = pd.read_sql_query(text(sql), source_engine)
        
        if len(df) == 0:
            logger.warning("沒有找到符合條件的貼文")
            return pd.DataFrame()
            
        df["group_id"] = pd.Series(range(len(df))) % group_count
        df["label"] = None
        df["note"] = None
        
        return df
    except Exception as e:
        logger.error(f"獲取候選貼文時發生錯誤: {str(e)}")
        raise

def _save_candidates_to_db(
    df: pd.DataFrame,
    target_engine: create_engine,
    table_name: str = "candidates",
    if_exists: str = "replace"
) -> None:
    """
    將候選貼文儲存到目標資料庫
    
    Args:
        df: 要儲存的 DataFrame
        target_engine: 目標資料庫引擎
        table_name: 目標表格名稱
        if_exists: 表格已存在時的處理方式
    """
    try:
        df.to_sql(table_name, target_engine, if_exists=if_exists, index=False)
        logger.info(f"✅ 已複製 {len(df)} 筆資料到 {table_name}")
    except Exception as e:
        logger.error(f"儲存資料到資料庫時發生錯誤: {str(e)}")
        raise

def process_wrapper(
    source_db: str = 'social_media_analysis',
    target_db: str = 'labeling_db',
    keywords: Optional[List[str]] = None,
    pick_number: int = 4000,
    group_count: int = 5,
    user: str = 'postgres',
    password: str = '00000000',
    host: str = 'localhost',
    port: int = 5432
) -> None:
    """
    主要處理函數：建立資料庫、獲取候選貼文並儲存
    
    Args:
        source_db: 來源資料庫名稱
        target_db: 目標資料庫名稱
        keywords: 關鍵字列表，如果為 None 則使用預設關鍵字
        pick_number: 要獲取的貼文數量
        group_count: 要分成的組別數量
        user: 資料庫使用者名稱
        password: 資料庫密碼
        host: 資料庫主機位置
        port: 資料庫連接埠
    """
    if keywords is None:
        keywords = ['穩賺', '借貸', '老師帶你賺', '量化交易', '虛擬貨幣', '急用', '輕鬆', '投資']
    
    try:
        # 建立目標資料庫（如果不存在）
        _create_db_if_not_exists(target_db, user, password, host)
        
        # 建立資料庫引擎
        source_engine = _get_db_engine(source_db, user, password, host, port)
        target_engine = _get_db_engine(target_db, user, password, host, port)
        
        # 獲取並處理候選貼文
        df = fetch_candidate_posts(source_engine, keywords, pick_number, group_count)
        
        if not df.empty:
            # 儲存到目標資料庫
            _save_candidates_to_db(df, target_engine)
        else:
            logger.warning("沒有資料需要儲存")
            
    except Exception as e:
        logger.error(f"處理候選貼文時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    # 使用預設參數執行
    process_wrapper()





