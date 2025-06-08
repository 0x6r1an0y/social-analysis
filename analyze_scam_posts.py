import pandas as pd
import jieba
import jieba.analyse
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from typing import List, Dict
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 資料庫連線設定
LABELING_DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
engine = create_engine(LABELING_DB_URL)

def fetch_scam_posts() -> pd.DataFrame:
    """從資料庫獲取所有標記為詐騙的貼文"""
    query = """
        SELECT pos_tid, content, note
        FROM candidates 
        WHERE label = '是'
    """
    logger.info("正在從資料庫獲取詐騙貼文...")
    return pd.read_sql(query, engine)

def preprocess_text(text: str) -> str:
    """文字預處理：移除特殊字元、空白等"""
    if not isinstance(text, str):
        return ""
    # 移除換行符號
    text = text.replace('\n', ' ')
    # 移除多餘空白
    text = ' '.join(text.split())
    return text

def segment_text(text: str) -> List[str]:
    """使用jieba進行分詞"""
    # 使用jieba的TF-IDF模式進行分詞
    words = jieba.analyse.extract_tags(
        text,
        topK=50,  # 取前20個關鍵詞
        withWeight=False,
        allowPOS=('n', 'vn', 'v', 'a')  # 只保留名詞、動名詞、動詞、形容詞
    )
    return words

def analyze_word_frequency(df: pd.DataFrame) -> Dict[str, int]:
    """分析所有貼文的詞頻"""
    all_words = []
    
    for content in df['content']:
        processed_text = preprocess_text(content)
        words = segment_text(processed_text)
        all_words.extend(words)
    
    # 計算詞頻
    word_freq = Counter(all_words)
    return dict(word_freq)

def plot_word_frequency(word_freq: Dict[str, int], top_n: int = 20):
    """繪製詞頻分析圖"""
    # 取前N個最常出現的詞
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 建立圖表
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
    plt.title('詐騙貼文關鍵詞頻率分析')
    plt.xlabel('出現次數')
    plt.ylabel('關鍵詞')
    plt.tight_layout()
    
    # 儲存圖表
    plt.savefig('word_frequency.png')
    plt.close()

def generate_wordcloud(word_freq: Dict[str, int]):
    """生成詞雲圖"""
    # 設定中文字型
    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/msjh.ttc',  # Windows 中文字型路徑
        width=800,
        height=400,
        background_color='white'
    )
    
    # 生成詞雲
    wordcloud.generate_from_frequencies(word_freq)
    
    # 儲存詞雲圖
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('wordcloud.png')
    plt.close()

def main():
    try:
        # 獲取詐騙貼文
        df = fetch_scam_posts()
        logger.info(f"成功獲取 {len(df)} 則詐騙貼文")
        
        # 分析詞頻
        word_freq = analyze_word_frequency(df)
        logger.info("完成詞頻分析")
        
        # 繪製詞頻分析圖
        plot_word_frequency(word_freq)
        logger.info("已生成詞頻分析圖 (word_frequency.png)")
        
        # 生成詞雲圖
        generate_wordcloud(word_freq)
        logger.info("已生成詞雲圖 (wordcloud.png)")
        
        # 輸出統計資訊
        print(f"\n總共分析了 {len(df)} 則詐騙貼文")
        print(f"共找出 {len(word_freq)} 個關鍵詞")
        print("\n前10個最常出現的關鍵詞：")
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{word}: {freq}次")
            
    except Exception as e:
        logger.error(f"分析過程中發生錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    main() 