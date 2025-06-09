import torch
from sentence_transformers import SentenceTransformer
import time

def test_large_batch_sizes():
    print('🧪 測試大批次大小性能...')
    print(f'GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
    test_texts = ['測試文本 ' + str(i) for i in range(2048)]  # 準備更多測試文本
    
    batch_sizes = [128, 256, 512, 1024, 2048]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            print(f'\n測試批次大小: {batch_size}')
            
            # 只使用相應數量的文本來測試
            current_texts = test_texts[:batch_size]
            
            start_time = time.time()
            embeddings = model.encode(current_texts, batch_size=batch_size, show_progress_bar=False)
            elapsed_time = time.time() - start_time
            
            texts_per_second = len(current_texts) / elapsed_time
            memory_used = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            
            print(f'  ⏱️ 時間: {elapsed_time:.3f}秒')
            print(f'  🚀 速度: {texts_per_second:.1f} texts/秒')
            print(f'  💾 記憶體使用: {memory_used:.1f} MB (已分配) / {memory_cached:.1f} MB (已快取)')
            
            results[batch_size] = texts_per_second
            
        except Exception as e:
            print(f'  ❌ 錯誤: {str(e)}')
            results[batch_size] = 0
    
    print(f'\n🏆 最佳批次大小: {max(results, key=results.get)} ({max(results.values()):.1f} texts/秒)')
    
    # 分析結果
    print('\n📊 詳細分析:')
    for batch_size, speed in sorted(results.items()):
        if speed > 0:
            efficiency = speed / batch_size  # 每個文本的處理效率
            print(f'  批次 {batch_size:4d}: {speed:8.1f} texts/秒 (效率: {efficiency:.3f})')
    
    return results

def test_embedding_generator_batch_size():
    """測試在實際 embedding generator 中使用大批次大小"""
    try:
        from embedding_generator_memmap import EmbeddingGeneratorMemmap
        
        print('\n🔧 測試 EmbeddingGeneratorMemmap 的大批次處理...')
        
        # 測試不同 batch_size 設定
        for batch_size in [512, 1024, 2048]:
            try:
                print(f'\n測試 batch_size={batch_size}:')
                
                generator = EmbeddingGeneratorMemmap(
                    batch_size=batch_size,
                    device="cuda"
                )
                
                print(f'  ✅ 成功創建 generator (batch_size={batch_size})')
                print(f'  📱 使用裝置: {generator.device}')
                
                # 測試連接
                if generator.test_connection():
                    print(f'  ✅ 連接測試通過')
                else:
                    print(f'  ❌ 連接測試失敗')
                    
            except Exception as e:
                print(f'  ❌ batch_size={batch_size} 失敗: {str(e)}')
                
    except ImportError:
        print('⚠️ 無法導入 EmbeddingGeneratorMemmap，跳過實際測試')

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_large_batch_sizes()
        test_embedding_generator_batch_size()
    else:
        print('❌ GPU 不可用，無法測試大批次大小') 