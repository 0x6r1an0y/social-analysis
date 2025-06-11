from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import pickle
import logging
from typing import List, Dict, Optional
import argparse

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScamDetector:
    def __init__(self, 
                 db_url: str = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis_hash",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size: int = 1000):
        """
        初始化詐騙檢測器
        
        Args:
            db_url: 資料庫連接字串
            model_name: 使用的 sentence-transformers 模型
            batch_size: 批次處理大小
        """
        self.db_url = db_url
        self.batch_size = batch_size
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
            
    def _load_embeddings_from_db(self, pos_tids: List[str]) -> Dict[str, np.ndarray]:
        """從資料庫載入預計算的 embeddings"""
        try:
            embeddings_dict = {}
            
            with self.engine.connect() as conn:
                for pos_tid in pos_tids:
                    result = conn.execute(text("""
                        SELECT content_emb 
                        FROM posts 
                        WHERE pos_tid = :pos_tid AND content_emb IS NOT NULL
                    """), {'pos_tid': pos_tid})
                    
                    row = result.fetchone()
                    if row and row[0]:
                        # 反序列化 embedding
                        embedding = pickle.loads(row[0])
                        embeddings_dict[pos_tid] = embedding
                        
            return embeddings_dict
            
        except Exception as e:
            logger.error(f"載入 embeddings 時發生錯誤: {str(e)}")
            raise
            
    def _get_posts_batch(self, offset: int = 0, limit: Optional[int] = None) -> pd.DataFrame:
        """獲取批次貼文資料"""
        try:
            if limit is None:
                limit = self.batch_size
                
            sql = """
                SELECT pos_tid, content, page_name, created_time, content_emb
                FROM posts 
                WHERE content IS NOT NULL 
                AND content != ''
                AND content_emb IS NOT NULL
                ORDER BY pos_tid
                LIMIT %s OFFSET %s
            """
            
            df = pd.read_sql_query(text(sql), self.engine, params=(limit, offset))
            return df
            
        except Exception as e:
            logger.error(f"獲取貼文資料時發生錯誤: {str(e)}")
            raise
            
    def calculate_scam_scores(self, 
                            contents: List[str], 
                            content_embeddings: List[np.ndarray],
                            scam_phrases: Optional[List[str]] = None,
                            aggregation_method: str = 'max') -> List[float]:
        """
        計算詐騙分數
        
        Args:
            contents: 貼文內容列表
            content_embeddings: 貼文的 embeddings
            scam_phrases: 詐騙提示詞列表
            aggregation_method: 聚合方法 ('max', 'mean', 'weighted_mean')
            
        Returns:
            詐騙風險分數列表
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        try:
            # 生成詐騙提示詞的 embeddings
            phrase_embeddings = self.model.encode(scam_phrases, convert_to_tensor=True)
            
            scam_scores = []
            
            for content, content_emb in zip(contents, content_embeddings):
                if content_emb is None:
                    scam_scores.append(0.0)
                    continue
                    
                # 轉換為 tensor
                content_tensor = util.pytorch_cos_sim(
                    self.model.encode(content, convert_to_tensor=True).unsqueeze(0),
                    phrase_embeddings
                )
                
                # 計算相似度分數
                similarities = content_tensor.squeeze().cpu().numpy()
                
                # 根據聚合方法計算最終分數
                if aggregation_method == 'max':
                    score = float(np.max(similarities))
                elif aggregation_method == 'mean':
                    score = float(np.mean(similarities))
                elif aggregation_method == 'weighted_mean':
                    # 給高相似度更高的權重
                    weights = np.exp(similarities * 2)  # 指數權重
                    score = float(np.average(similarities, weights=weights))
                else:
                    score = float(np.max(similarities))
                    
                scam_scores.append(score)
                
            return scam_scores
            
        except Exception as e:
            logger.error(f"計算詐騙分數時發生錯誤: {str(e)}")
            raise
            
    def detect_scam_in_batch(self, 
                           scam_phrases: Optional[List[str]] = None,
                           threshold: float = 0.6,
                           output_file: Optional[str] = None,
                           max_results: Optional[int] = None) -> pd.DataFrame:
        """
        批次檢測詐騙貼文
        
        Args:
            scam_phrases: 自定義詐騙提示詞
            threshold: 詐騙風險閾值
            output_file: 輸出檔案路徑
            max_results: 最大結果數量
            
        Returns:
            包含詐騙分數的 DataFrame
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        logger.info(f"使用的詐騙提示詞: {scam_phrases}")
        logger.info(f"詐騙風險閾值: {threshold}")
        
        try:
            # 獲取總數
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM posts 
                    WHERE content IS NOT NULL 
                    AND content != ''
                    AND content_emb IS NOT NULL
                """))
                total_posts = result.scalar()
                
            logger.info(f"待檢測的貼文總數: {total_posts}")
            
            all_results = []
            processed = 0
            offset = 0
            
            while processed < total_posts:
                # 獲取批次資料
                df = self._get_posts_batch(offset)
                
                if df.empty:
                    break
                    
                logger.info(f"正在處理第 {processed + 1} 到 {processed + len(df)} 筆資料...")
                
                # 載入 embeddings
                embeddings = []
                valid_indices = []
                
                for idx, row in df.iterrows():
                    if row['content_emb'] is not None:
                        embedding = pickle.loads(row['content_emb'])
                        embeddings.append(embedding)
                        valid_indices.append(idx)
                    else:
                        embeddings.append(None)
                        
                # 計算詐騙分數
                scam_scores = self.calculate_scam_scores(
                    df['content'].tolist(),
                    embeddings,
                    scam_phrases
                )
                
                # 添加分數到 DataFrame
                df['scam_score'] = scam_scores
                df['is_potential_scam'] = df['scam_score'] >= threshold
                
                # 只保留可能的詐騙貼文
                high_risk_posts = df[df['is_potential_scam']].copy()
                
                if not high_risk_posts.empty:
                    all_results.append(high_risk_posts)
                    
                processed += len(df)
                offset += self.batch_size
                
                # 如果達到最大結果數量，提前結束
                if max_results and sum(len(r) for r in all_results) >= max_results:
                    break
                    
                logger.info(f"已處理 {processed}/{total_posts} 筆，發現 {len(high_risk_posts)} 筆可疑貼文")
                
            # 合併所有結果
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                final_results = final_results.sort_values('scam_score', ascending=False)
                
                if max_results:
                    final_results = final_results.head(max_results)
                    
                logger.info(f"🎯 檢測完成！共發現 {len(final_results)} 筆可疑詐騙貼文")
                
                # 輸出結果
                if output_file:
                    final_results.to_csv(output_file, index=False, encoding='utf-8-sig')
                    logger.info(f"結果已儲存到: {output_file}")
                    
                return final_results
            else:
                logger.info("沒有發現可疑的詐騙貼文")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"批次檢測時發生錯誤: {str(e)}")
            raise
            
    def detect_single_post(self, content: str, scam_phrases: Optional[List[str]] = None) -> Dict:
        """
        檢測單一貼文
        
        Args:
            content: 貼文內容
            scam_phrases: 詐騙提示詞
            
        Returns:
            包含詐騙分數和詳細資訊的字典
        """
        if scam_phrases is None:
            scam_phrases = self.default_scam_phrases
            
        try:
            # 生成內容 embedding
            content_embedding = self.model.encode(content, convert_to_tensor=False)
            
            # 計算詐騙分數
            scam_scores = self.calculate_scam_scores(
                [content], 
                [content_embedding], 
                scam_phrases
            )
            
            scam_score = scam_scores[0]
            
            # 計算與每個詐騙短語的相似度
            phrase_embeddings = self.model.encode(scam_phrases, convert_to_tensor=True)
            content_tensor = self.model.encode(content, convert_to_tensor=True)
            
            similarities = util.cos_sim(content_tensor, phrase_embeddings).squeeze().cpu().numpy()
            
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
                'top_matching_phrases': top_matches[:5],
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
            
            results = []
            processed = 0
            offset = 0
            
            while len(results) < limit:
                # 獲取批次資料
                df = self._get_posts_batch(offset, self.batch_size)
                
                if df.empty:
                    break
                    
                for _, row in df.iterrows():
                    if row['content_emb'] is None:
                        continue
                        
                    # 載入 embedding
                    content_embedding = pickle.loads(row['content_emb'])
                    content_tensor = self.model.encode(row['content'], convert_to_tensor=True)
                    
                    # 計算與關鍵字的相似度
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
                offset += self.batch_size
                
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

def main():
    """主要執行函數"""
    parser = argparse.ArgumentParser(description='詐騙貼文檢測工具')
    parser.add_argument('--mode', choices=['batch', 'single', 'search'], default='batch',
                       help='執行模式: batch=批次檢測, single=單一貼文檢測, search=關鍵字搜尋')
    parser.add_argument('--content', type=str, help='單一貼文內容 (single 模式使用)')
    parser.add_argument('--keywords', nargs='+', help='搜尋關鍵字 (search 模式使用)')
    parser.add_argument('--threshold', type=float, default=0.6, help='詐騙風險閾值')
    parser.add_argument('--output', type=str, help='輸出檔案路徑')
    parser.add_argument('--limit', type=int, default=1000, help='最大結果數量')
    parser.add_argument('--scam-phrases', nargs='+', help='自定義詐騙提示詞')
    
    args = parser.parse_args()
    
    try:
        # 創建檢測器
        detector = ScamDetector()
        
        if args.mode == 'batch':
            # 批次檢測
            results = detector.detect_scam_in_batch(
                scam_phrases=args.scam_phrases,
                threshold=args.threshold,
                output_file=args.output,
                max_results=args.limit
            )
            
            if not results.empty:
                print(f"\n🎯 發現 {len(results)} 筆可疑詐騙貼文:")
                print(results[['pos_tid', 'page_name', 'scam_score', 'content']].head(10).to_string())
                
        elif args.mode == 'single':
            if not args.content:
                print("❌ 請提供貼文內容 (--content)")
                return
                
            # 單一貼文檢測
            result = detector.detect_single_post(args.content, args.scam_phrases)
            
            print(f"\n📝 貼文內容: {result['content']}")
            print(f"🎯 詐騙分數: {result['scam_score']:.3f}")
            print(f"⚠️  風險等級: {result['risk_level']}")
            print(f"🚨 是否可疑: {'是' if result['is_potential_scam'] else '否'}")
            print("\n🔍 最相似的詐騙短語:")
            for match in result['top_matching_phrases'][:3]:
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