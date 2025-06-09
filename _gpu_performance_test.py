import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_device_performance():
    """測試不同裝置的 embedding 生成性能"""
    
    # 測試文本
    test_texts = [
        "這是一個測試文本，用於評估模型性能。",
        "Machine learning is transforming the way we analyze data.",
        "人工智慧在各個領域都有廣泛的應用。",
        "The quick brown fox jumps over the lazy dog.",
        "深度學習模型需要大量的計算資源。"
    ] * 100  # 500 個文本
    
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # 測試結果
    results = {}
    
    # 測試 CPU
    logger.info("🖥️ 測試 CPU 性能...")
    try:
        cpu_model = SentenceTransformer(model_name, device='cpu')
        
        start_time = time.time()
        cpu_embeddings = cpu_model.encode(test_texts, batch_size=16, show_progress_bar=False)
        cpu_time = time.time() - start_time
        
        results['cpu'] = {
            'time': cpu_time,
            'texts_per_second': len(test_texts) / cpu_time,
            'embedding_shape': cpu_embeddings.shape
        }
        
        logger.info(f"✅ CPU 測試完成: {cpu_time:.2f}秒, {results['cpu']['texts_per_second']:.2f} texts/秒")
        
    except Exception as e:
        logger.error(f"CPU 測試失敗: {str(e)}")
        results['cpu'] = {'error': str(e)}
    
    # 測試 GPU (如果可用)
    if torch.cuda.is_available():
        logger.info("🎮 測試 GPU 性能...")
        try:
            gpu_model = SentenceTransformer(model_name, device='cuda')
            
            # 預熱 GPU
            logger.info("預熱 GPU...")
            gpu_model.encode(test_texts[:10])
            torch.cuda.empty_cache()
            
            start_time = time.time()
            gpu_embeddings = gpu_model.encode(test_texts, batch_size=32, show_progress_bar=False)
            gpu_time = time.time() - start_time
            
            results['gpu'] = {
                'time': gpu_time,
                'texts_per_second': len(test_texts) / gpu_time,
                'embedding_shape': gpu_embeddings.shape,
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
            
            logger.info(f"✅ GPU 測試完成: {gpu_time:.2f}秒, {results['gpu']['texts_per_second']:.2f} texts/秒")
            
            # 計算加速比
            if 'cpu' in results and 'time' in results['cpu']:
                speedup = results['cpu']['time'] / gpu_time
                results['speedup'] = speedup
                logger.info(f"🚀 GPU 加速比: {speedup:.2f}x")
                
            # GPU 記憶體使用情況
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"GPU 記憶體使用: {memory_allocated:.1f} MB (已分配) / {memory_cached:.1f} MB (已快取)")
            
        except Exception as e:
            logger.error(f"GPU 測試失敗: {str(e)}")
            results['gpu'] = {'error': str(e)}
    else:
        logger.warning("⚠️ 系統不支援 CUDA，跳過 GPU 測試")
        results['gpu'] = {'error': 'CUDA not available'}
    
    return results

def benchmark_batch_sizes():
    """測試不同批次大小的性能影響 (僅 GPU)"""
    if not torch.cuda.is_available():
        logger.warning("GPU 不可用，跳過批次大小測試")
        return {}
        
    logger.info("🔧 測試不同批次大小的性能...")
    
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cuda')
    test_texts = ["測試文本 " + str(i) for i in range(200)]
    
    batch_sizes = [8, 16, 32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            start_time = time.time()
            embeddings = model.encode(test_texts, batch_size=batch_size, show_progress_bar=False)
            elapsed_time = time.time() - start_time
            
            results[batch_size] = {
                'time': elapsed_time,
                'texts_per_second': len(test_texts) / elapsed_time
            }
            
            logger.info(f"批次大小 {batch_size}: {elapsed_time:.2f}秒, {results[batch_size]['texts_per_second']:.2f} texts/秒")
            
        except Exception as e:
            logger.error(f"批次大小 {batch_size} 測試失敗: {str(e)}")
            results[batch_size] = {'error': str(e)}
    
    return results

def main():
    """主要測試函數"""
    logger.info("🧪 開始 GPU 性能測試...")
    
    # 系統資訊
    logger.info("📊 系統資訊:")
    logger.info(f"   - PyTorch 版本: {torch.__version__}")
    logger.info(f"   - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   - CUDA 版本: {torch.version.cuda}")
        logger.info(f"   - GPU 數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n" + "="*50)
    
    # 性能測試
    performance_results = test_device_performance()
    
    print("\n" + "="*50)
    print("📊 性能測試結果摘要:")
    print("="*50)
    
    for device, result in performance_results.items():
        if device == 'speedup':
            continue
            
        print(f"\n{device.upper()}:")
        if 'error' in result:
            print(f"  ❌ 錯誤: {result['error']}")
        else:
            print(f"  ⏱️ 時間: {result['time']:.2f} 秒")
            print(f"  🚀 速度: {result['texts_per_second']:.2f} texts/秒")
            if 'gpu_name' in result:
                print(f"  🎮 GPU: {result['gpu_name']}")
                print(f"  💾 記憶體: {result['gpu_memory_gb']:.1f} GB")
    
    if 'speedup' in performance_results:
        print(f"\n🚀 GPU 加速比: {performance_results['speedup']:.2f}x 倍快")
    
    # 批次大小測試
    if torch.cuda.is_available():
        print("\n" + "="*50)
        batch_results = benchmark_batch_sizes()
        
        if batch_results:
            print("🔧 最佳批次大小建議:")
            best_batch = max(batch_results.items(), 
                           key=lambda x: x[1].get('texts_per_second', 0) if 'error' not in x[1] else 0)
            print(f"   推薦批次大小: {best_batch[0]} (速度: {best_batch[1]['texts_per_second']:.2f} texts/秒)")
    
    print("\n" + "="*50)
    print("💡 使用建議:")
    if torch.cuda.is_available():
        print("  ✅ 您的系統支援 GPU 加速！")
        print("  📝 修改程式碼時使用 device='cuda' 或 device='auto'")
        print("  🔧 建議使用較大的批次大小 (32-64)")
        print("  ⚡ 預期可獲得 2-10x 的性能提升")
    else:
        print("  ⚠️ 您的系統不支援 GPU 加速")
        print("  📝 需要安裝 CUDA 和 GPU 版本的 PyTorch")
        print("  💻 或繼續使用 CPU (device='cpu')")

if __name__ == "__main__":
    main() 