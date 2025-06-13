# 社群網路媒體詐騙貼文分析系統

## 📋 專案簡介

這是一個專門用於分析社群網路媒體中高風險投資詐騙內容的智能分析系統。系統結合了自然語言處理、機器學習和人工標記技術，能夠有效識別和分類潛在的詐騙貼文。

## 🎯 主要功能

### 🔍 智能貼文分析
- **語意相似度搜尋**：使用 Sentence Transformers 模型進行語意相似度計算
- **關鍵字搜尋**：支援多關鍵字組合搜尋，可設定 OR/AND 邏輯
- **詐騙模式識別**：基於預定義的詐騙關鍵詞進行初步篩選

### 📝 人工標記系統
- **批次標記模式**：支援按群組進行批次標記作業
- **即時標記**：可對搜尋結果進行即時標記
- **進度追蹤**：自動記錄標記進度，支援斷點續標
- **標記歷史**：瀏覽已標記的詐騙貼文

### 📊 資料分析與視覺化
- **詞頻分析**：生成詐騙貼文詞頻統計圖表
- **文字雲**：視覺化展示詐騙貼文中的關鍵詞
- **統計報表**：提供詳細的標記統計資訊

### 🗄️ 資料管理
- **資料庫整合**：支援 PostgreSQL 資料庫
- **資料清洗**：自動去重、內容驗證
- **批次處理**：支援大量資料的批次處理

## 🏗️ 系統架構

### 📁 目錄結構
```
社群網路媒體分析系統/
├── src/                    # 核心程式碼目錄
│   ├── web/               # Web 介面層
│   │   └── candidates_label_server.py  # Streamlit 標記介面
│   ├── data/              # 資料處理層
│   │   ├── raw_dataloader_to_sql.py    # 原始資料載入
│   │   ├── deduplicate_posts.py        # 貼文去重
│   │   ├── add_content_hash.py         # 內容雜湊生成
│   │   ├── candidates_dataloader_to_sql.py  # 候選資料載入
│   │   ├── delete_specific_category.py # 特定類別刪除
│   │   ├── delete_specific_page.py     # 特定頁面刪除
│   │   └── delete_short_content_posts.py # 短內容刪除
│   ├── core/              # 核心功能層
│   │   ├── scam_detector.py            # 詐騙檢測器（標準版）
│   │   ├── scam_detector_memmap.py     # 詐騙檢測器（記憶體優化版）
│   │   ├── scam_detector_lightweight.py # 詐騙檢測器（輕量版）
│   │   ├── embedding_generator.py      # 語意向量生成器（標準版）
│   │   ├── embedding_generator_memmap.py # 語意向量生成器（記憶體優化版）
│   │   ├── triple_extractor.py         # 三元組提取器
│   │   └── knowledge_graph_visualizer.py # 知識圖譜視覺化器
│   ├── analysis/          # 分析模組
│   │   └── analyze_scam_posts.py       # 詐騙貼文分析
│   └── utils/             # 工具函數
│       └── LLM_responder.py            # LLM 回應器
├── data/                  # 資料檔案
│   ├── csv/               # CSV 資料檔案
│   └── embeddings/        # 語意向量檔案
├── logs/                  # 系統日誌
├── tools/                 # 輔助工具
└── knowledge_graphs/      # 知識圖譜
```

### 🔧 核心功能模組詳細說明

#### 📊 資料處理層 (src/data/)

##### 1. `raw_dataloader_to_sql.py` - 原始資料載入器
**功能描述**：將清洗後的 CSV 資料載入到 PostgreSQL 資料庫

**主要功能**：
- 自動建立 PostgreSQL 資料庫和資料表
- 分塊處理大型 CSV 檔案（預設 100,000 筆/批次）
- 資料型別轉換和驗證
- 錯誤處理和詳細錯誤分析
- 進度追蹤和效能監控

**使用方式**：
```bash
python src/data/raw_dataloader_to_sql.py
```

##### 2. `deduplicate_posts.py` - 貼文去重處理器
**功能描述**：基於 content_hash 進行貼文去重，建立去重後的資料表

**主要功能**：
- 基於內容雜湊值識別重複貼文
- 保留最新時間戳的重複貼文
- 自動建立索引提升查詢效能
- 生成詳細的去重報告
- 統計重複率和處理結果

**使用方式**：
```bash
python src/data/deduplicate_posts.py
```

##### 3. `add_content_hash.py` - 內容雜湊生成器
**功能描述**：為貼文內容生成 MD5 雜湊值，用於去重識別

##### 4. `candidates_dataloader_to_sql.py` - 候選資料載入器
**功能描述**：將篩選後的候選貼文載入到標記資料庫

##### 5. `delete_specific_category.py` - 特定類別刪除器
**功能描述**：刪除指定類別的貼文資料

##### 6. `delete_specific_page.py` - 特定頁面刪除器
**功能描述**：刪除指定頁面的貼文資料

##### 7. `delete_short_content_posts.py` - 短內容貼文刪除器
**功能描述**：刪除內容過短的貼文

#### 🧠 核心功能層 (src/core/)

##### 1. `scam_detector.py` - 詐騙檢測器（標準版）
**功能描述**：使用語意相似度進行詐騙貼文檢測

**主要功能**：
- 載入 Sentence Transformers 模型
- 計算貼文與詐騙關鍵詞的相似度
- 批次檢測大量貼文
- 風險等級評估
- 結果匯出和統計

**核心方法**：
```python
class ScamDetector:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.default_scam_phrases = [
            "加入LINE", "加入Telegram", "快速賺錢", "被動收入",
            "投資包你賺", "私訊我", "老師帶單", "穩賺不賠"
        ]
```

##### 2. `scam_detector_memmap.py` - 詐騙檢測器（記憶體優化版）
**功能描述**：使用 memmap 技術優化記憶體使用的詐騙檢測器

**主要功能**：
- 使用 numpy.memmap 處理大型向量檔案
- 支援數百萬筆貼文的快速搜尋
- 記憶體使用量大幅降低
- 批次處理和進度追蹤
- 多進程支援

##### 3. `scam_detector_lightweight.py` - 詐騙檢測器（輕量版）
**功能描述**：適用於資源受限環境的輕量級檢測器

##### 4. `embedding_generator.py` - 語意向量生成器（標準版）
**功能描述**：為貼文內容生成語意向量

##### 5. `embedding_generator_memmap.py` - 語意向量生成器（記憶體優化版）
**功能描述**：使用 memmap 技術的大規模向量生成器

**主要功能**：
- 支援數百萬筆貼文的向量生成
- 記憶體優化的批次處理
- GPU 加速支援
- 斷點續傳功能
- 處理狀態追蹤

##### 6. `triple_extractor.py` - 三元組提取器
**功能描述**：從貼文內容中提取實體關係三元組

##### 7. `knowledge_graph_visualizer.py` - 知識圖譜視覺化器
**功能描述**：將提取的三元組轉換為可視化的知識圖譜

#### 📈 分析模組層 (src/analysis/)

##### 1. `analyze_scam_posts.py` - 詐騙貼文分析器
**功能描述**：對已標記的詐騙貼文進行深度分析

**主要功能**：
- 詞頻統計和分析
- 文字雲生成
- 關鍵詞提取
- 統計圖表生成
- 分析報告輸出

**使用方式**：
```bash
python src/analysis/analyze_scam_posts.py
```

#### 🌐 Web 介面層 (src/web/)

##### 1. `candidates_label_server.py` - 標記系統伺服器
**功能描述**：基於 Streamlit 的 Web 標記介面

**主要功能**：
- **標記模式**：批次標記貼文（是/否）
- **瀏覽模式**：查看已標記的詐騙貼文
- **關鍵字搜尋**：多關鍵字組合搜尋
- **相似貼文搜尋**：語意相似度搜尋
- **詞彙分析**：生成詞頻圖表和文字雲
- **貼文查詢**：根據 ID 查詢特定貼文

**核心特色**：
```python
# 多進程相似搜尋
class SimilarSearchProcess:
    def __init__(self, embeddings_dir="data", batch_size=32768):
        self.process = None
        self.result_queue = Queue()
        self.progress_queue = Queue()
        self.stop_event = mp.Event()
```

**使用方式**：
```bash
streamlit run src/web/candidates_label_server.py
```

#### 🛠️ 工具函數層 (src/utils/)

##### 1. `LLM_responder.py` - LLM 回應器
**功能描述**：整合大型語言模型進行智能回應

### 🔄 資料流程

#### 1. 資料預處理流程
```
原始 CSV → raw_dataloader_to_sql.py → PostgreSQL (posts)
         ↓
add_content_hash.py → 添加 content_hash
         ↓
deduplicate_posts.py → PostgreSQL (posts_deduplicated)
```

#### 2. 向量生成流程
```
posts_deduplicated → embedding_generator_memmap.py → embeddings.dat
                  ↓
               pos_tid_index.json (索引檔案)
               metadata.json (元資料)
```

#### 3. 詐騙檢測流程
```
embeddings.dat → scam_detector_memmap.py → 候選貼文
              ↓
candidates_dataloader_to_sql.py → 標記資料庫
```

#### 4. 人工標記流程
```
標記資料庫 → candidates_label_server.py → Web 介面
         ↓
analyze_scam_posts.py → 分析報告和圖表
```

## 🚀 快速開始

### 環境需求

- **Python**: 3.8+
- **PostgreSQL**: 12+
- **記憶體**: 建議 8GB+
- **硬碟空間**: 建議 10GB+

### 安裝步驟

1. **克隆專案**
   ```bash
   git clone <repository-url>
   cd 社群網路媒體分析
   ```

2. **安裝 Python 依賴**
   ```bash
   pip install -r requirements.txt
   ```

3. **設定 PostgreSQL 資料庫**
   ```sql
   -- 建立標記資料庫
   CREATE DATABASE labeling_db;
   
   -- 建立原始資料庫
   CREATE DATABASE social_media_analysis_hash;
   ```

4. **設定環境變數**
   ```bash
   # 資料庫連線設定
   LABELING_DB_URL="postgresql+psycopg2://postgres:password@localhost:5432/labeling_db"
   SOURCE_DB_URL="postgresql+psycopg2://postgres:password@localhost:5432/social_media_analysis_hash"
   ```

5. **啟動應用程式**
   ```bash
   # Windows
   _run_server.cmd
   
   # 或直接執行
   streamlit run src/web/candidates_label_server.py
   ```

## 📖 使用指南

### 1. 標記模式
- 選擇群組編號開始標記
- 使用「是/否」按鈕進行快速標記
- 支援備註功能記錄詳細資訊
- 自動保存進度，支援斷點續標

### 2. 瀏覽模式
- **詐騙貼文瀏覽**：查看所有已標記的詐騙貼文
- **貼文查詢**：根據 pos_tid 查詢特定貼文
- **詞彙分析**：生成詞頻圖表和文字雲

### 3. 關鍵字搜尋
- 輸入多個關鍵字（每行一個）
- 選擇搜尋邏輯（OR/AND）
- 支援排除關鍵字功能
- 結果分頁顯示，每頁 20 筆

### 4. 相似貼文搜尋
- 輸入任意文字進行語意搜尋
- 可調整相似度閾值
- 支援隨機搜尋模式
- 即時顯示搜尋進度

## 🔧 進階功能

### 資料預處理
```bash
# 原始資料清洗
python src/data/raw_dataloader_to_sql.py

# 貼文去重
python src/data/deduplicate_posts.py

# 生成語意向量
python src/core/embedding_generator_memmap.py
```

### 分析報告生成
```bash
# 生成詞彙分析圖表
python src/analysis/analyze_scam_posts.py
```

### 外部存取設定
```bash
# 使用 ngrok 進行外部存取
ngrok http 8501
```

## 📊 系統特色

### 高效能處理
- **記憶體優化**：使用 memmap 技術處理大型向量檔案
- **批次處理**：支援大量資料的批次處理
- **多進程搜尋**：避免 UI 阻塞，提供即時進度回饋

### 智能分析
- **多語言支援**：使用 multilingual 模型支援中文分析
- **語意理解**：基於 Transformer 的深度語意分析
- **模式識別**：預定義詐騙關鍵詞庫

### 使用者體驗
- **直觀介面**：Streamlit 提供的現代化 Web 介面
- **即時回饋**：搜尋進度即時顯示
- **錯誤處理**：完善的錯誤處理和日誌記錄

## 🎯 效能優化特色

### 記憶體優化
- **Memmap 技術**：處理大型向量檔案
- **批次處理**：減少記憶體峰值使用
- **垃圾回收**：自動清理未使用資源
- **連線池**：資料庫連線優化

### 計算優化
- **GPU 加速**：支援 CUDA 加速
- **多進程處理**：避免 UI 阻塞
- **索引優化**：資料庫查詢效能提升
- **向量化運算**：NumPy 高效能計算

### 使用者體驗
- **即時進度**：搜尋進度即時顯示
- **斷點續傳**：支援中斷後繼續處理
- **錯誤處理**：完善的異常處理機制
- **資源監控**：記憶體使用率監控

## 🛠️ 技術架構

### 核心技術
- **Streamlit**: Web 應用框架
- **Sentence Transformers**: 語意向量模型
- **PostgreSQL**: 資料庫管理
- **Pandas**: 資料處理
- **NumPy**: 數值計算
- **Matplotlib/Seaborn**: 資料視覺化

### 模型架構
- **預訓練模型**: paraphrase-multilingual-MiniLM-L12-v2
- **向量維度**: 384 維
- **相似度計算**: Cosine Similarity
- **批次大小**: 可配置（預設 32,768）

## 📈 系統效能指標

### 處理能力
- **向量生成**：每秒 1000+ 筆貼文
- **相似搜尋**：數百萬筆資料中秒級回應
- **記憶體使用**：優化後記憶體使用率 < 85%
- **並發處理**：支援多使用者同時標記

### 準確性指標
- **相似度計算**：基於 Cosine Similarity
- **詐騙識別**：可調整閾值平衡精度與召回率
- **多語言支援**：支援中文和英文混合文本

## 🔧 配置和部署

### 環境配置
```bash
# 安裝依賴
pip install -r requirements.txt

# 資料庫設定
CREATE DATABASE social_media_analysis_hash;
CREATE DATABASE labeling_db;

# 環境變數
export LABELING_DB_URL="postgresql+psycopg2://postgres:password@localhost:5432/labeling_db"
export SOURCE_DB_URL="postgresql+psycopg2://postgres:password@localhost:5432/social_media_analysis_hash"
```

### 啟動順序
1. 啟動 PostgreSQL 服務
2. 執行資料預處理腳本
3. 生成語意向量
4. 啟動 Web 標記介面

### 監控和維護
- 定期檢查日誌檔案
- 監控記憶體使用情況
- 備份重要資料檔案
- 更新模型和依賴套件

## 🔒 安全性

- **資料庫連線池**: 防止連線洩漏
- **進程隔離**: 搜尋進程獨立運行
- **資源清理**: 自動清理記憶體和連線資源
- **錯誤處理**: 完善的異常處理機制

## 📝 開發指南

### 程式碼結構
- 模組化設計，功能分離
- 統一的日誌記錄格式
- 完整的錯誤處理
- 詳細的程式碼註解

### 擴展功能
1. 新增詐騙模式識別規則
2. 整合其他 ML 模型
3. 增加更多視覺化圖表
4. 支援更多資料來源

## 🔮 未來擴展方向

### 功能擴展
1. **多模型支援**：整合更多預訓練模型
2. **即時學習**：根據標記結果調整模型
3. **自動化標記**：減少人工標記工作量
4. **API 介面**：提供 RESTful API 服務

### 技術升級
1. **分散式處理**：支援叢集部署
2. **雲端整合**：支援雲端儲存和計算
3. **即時串流**：支援即時資料處理
4. **移動端支援**：開發移動應用程式

## 🤝 貢獻指南

1. Fork 專案
2. 建立功能分支
3. 提交變更
4. 發起 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款，詳見 LICENSE 檔案。

## 📞 聯絡資訊

如有問題或建議，請透過以下方式聯絡：
- 專案 Issues
- 開發團隊信箱

## 🔄 更新日誌

### v1.0.0
- 初始版本發布
- 基本標記功能
- 相似貼文搜尋
- 關鍵字搜尋
- 詞彙分析

---

**注意事項**：
- 首次使用需要下載預訓練模型，可能需要一些時間
- 建議在 SSD 硬碟上運行以獲得更好的效能
- 大量資料處理時請確保有足夠的記憶體空間
- GPU 加速需要安裝 CUDA 和對應的 PyTorch 版本 