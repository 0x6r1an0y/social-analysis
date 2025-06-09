#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
詐騙檢測器 Top-K 相似度使用範例
"""

from scam_detector_memmap import ScamDetectorMemmap

def example_single_post_with_top_k():
    """單一貼文檢測範例 - 顯示 top_k 相似度"""
    print("=== 單一貼文檢測範例 ===")
    
    # 創建檢測器
    detector = ScamDetectorMemmap()
    
    # 測試貼文
    test_content = "加入我們的投資群組，老師帶單保證穩賺不賠，一天可以賺萬元以上！"
    
    # 檢測並顯示前 3 個最相似的短語
    result = detector.detect_single_post(test_content, top_k=3)
    
    print(f"📝 貼文內容: {result['content']}")
    print(f"🎯 詐騙分數: {result['scam_score']:.3f}")
    print(f"⚠️  風險等級: {result['risk_level']}")
    print(f"🚨 是否可疑: {'是' if result['is_potential_scam'] else '否'}")
    
    print(f"\n🔍 前 3 個最相似的詐騙短語:")
    for i, match in enumerate(result['top_matching_phrases'], 1):
        print(f"  {i}. {match['phrase']}: {match['similarity']:.3f}")

def example_batch_with_top_k():
    """批次檢測範例 - 包含 top_k 相似度"""
    print("\n=== 批次檢測範例 ===")
    
    # 創建檢測器
    detector = ScamDetectorMemmap()
    
    # 批次檢測，包含 top_k 相似度
    results = detector.detect_scam_in_batch(
        threshold=0.5,  # 降低閾值以獲得更多結果
        max_results=5,  # 只取前 5 筆
        top_k=3,        # 顯示前 3 個最相似的短語
        return_top_k=True  # 啟用 top_k 相似度
    )
    
    if not results.empty:
        print(f"🎯 發現 {len(results)} 筆可疑詐騙貼文:")
        print("\n詳細結果:")
        
        for idx, row in results.iterrows():
            print(f"\n--- 貼文 {idx + 1} ---")
            print(f"ID: {row['pos_tid']}")
            print(f"頁面: {row['page_name']}")
            print(f"詐騙分數: {row['scam_score']:.3f}")
            print(f"內容: {row['content'][:100]}...")
            
            print("前 3 個最相似的詐騙短語:")
            for i, similarity_info in enumerate(row['top_k_similarities'], 1):
                print(f"  {i}. {similarity_info['phrase']}: {similarity_info['similarity']:.3f}")
    else:
        print("沒有發現可疑的詐騙貼文")

def example_custom_scam_phrases():
    """自定義詐騙短語範例"""
    print("\n=== 自定義詐騙短語範例 ===")
    
    # 創建檢測器
    detector = ScamDetectorMemmap()
    
    # 自定義詐騙短語
    custom_phrases = [
        "免費試用", "限時優惠", "獨家代理", 
        "保證獲利", "零風險投資", "快速致富"
    ]
    
    # 測試貼文
    test_content = "我們提供免費試用，限時優惠，保證獲利，零風險投資機會！"
    
    # 檢測
    result = detector.detect_single_post(test_content, custom_phrases, top_k=5)
    
    print(f"📝 貼文內容: {result['content']}")
    print(f"🎯 詐騙分數: {result['scam_score']:.3f}")
    print(f"⚠️  風險等級: {result['risk_level']}")
    
    print(f"\n🔍 前 5 個最相似的自定義詐騙短語:")
    for i, match in enumerate(result['top_matching_phrases'], 1):
        print(f"  {i}. {match['phrase']}: {match['similarity']:.3f}")

if __name__ == "__main__":
    try:
        # 執行範例
        example_single_post_with_top_k()
        example_batch_with_top_k()
        example_custom_scam_phrases()
        
    except Exception as e:
        print(f"執行範例時發生錯誤: {str(e)}")
        print("請確保資料庫連接正常且 embeddings 檔案存在") 