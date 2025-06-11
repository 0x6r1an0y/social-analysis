import json
from dotenv import load_dotenv
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from utils import LLM_responder

load_dotenv()

# ===================== 設定區 =====================
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"

PROMPT_TEMPLATE = """
你是一個資訊抽取模型，專門用來從社群媒體投資貼文中抽取知識圖譜三元組。

請依照以下規則進行：

1. 從貼文中擷取出具有意義的知識性關係，整理成三元組格式：(主詞, 關係, 受詞)。
2. 重點擷取投資行為、收益保證、系統名稱、專家資格、招募行為等資訊。
3. 忽略廣告話術、重複詞彙、單純感嘆詞等無關資訊。
4. 若有網址、聯絡資訊、表單、群組，也請適當抽出其所代表的行為關係，網址不用把後面的parameter抽出，只需抽網址的網域，例如:https://bit.ly/3dn0mLR，只需抽bit.ly。
5. 結果請以 JSON 格式輸出，格式如下：
{{
  "triples": [
    ["主詞1", "關係1", "受詞1"],
    ["主詞2", "關係2", "受詞2"],
    ...
  ]
}}

以下是要處理的貼文內容：
-------------------------
{content}
-------------------------

請開始抽取。
"""

# 實際的資料表名稱和欄位
CANDIDATES_TABLE = "candidates"
# 本地 JSON 檔案路徑
OUTPUT_FILE = "knowledge_graph.json"


def load_existing_triples():
    """載入已存在的三元組資料"""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告：{OUTPUT_FILE} 格式錯誤，重新建立")
    return {}


def save_triples_to_json(triples_data):
    """將三元組資料儲存到 JSON 檔案"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(triples_data, f, ensure_ascii=False, indent=2)
    print(f"三元組資料已儲存到 {OUTPUT_FILE}")


def fetch_fraud_posts(engine):
    """抓取被標記為詐騙的貼文"""
    query = text("SELECT pos_tid, content, group_id, label, note FROM candidates WHERE label = '是'")
    
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
    return rows  # List of tuples (pos_tid, content, group_id, label, note)


def extract_triples_from_post(llm: LLM_responder, content: str, temperature: float = 0.2):
    """呼叫 LLM 取得三元組"""
    prompt = PROMPT_TEMPLATE.format(content=content)
    response = llm.chat_gpt_4o(prompt, temperature)

    try:
        # 清理回應，移除可能的 markdown 代碼塊標記
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # 移除 ```json
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]  # 移除 ```
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # 移除結尾的 ```
        
        cleaned_response = cleaned_response.strip()
        
        parsed = json.loads(cleaned_response)
        triples = parsed.get("triples", [])
        print(f"成功解析 {len(triples)} 筆三元組")
    except json.JSONDecodeError as e:
        print(f"JSON 解析失敗，錯誤：{e}")
        print("原始回應：", response)
        print("清理後回應：", cleaned_response)
        triples = []
    return triples


def main():
    # 初始化 LLM
    llm = LLM_responder()

    # 載入已存在的三元組資料
    triples_data = load_existing_triples()
    
    # 使用 SQLAlchemy 建立資料庫連線
    engine = create_engine(DB_URL)

    try:
        posts = fetch_fraud_posts(engine)
        print(f"共取得 {len(posts)} 則詐騙貼文，開始抽取...")

        for pos_tid, content, group_id, label, note in posts:
            # 檢查是否已經處理過這個貼文
            if str(pos_tid) in triples_data:
                print(f"跳過已處理的貼文 (pos_tid={pos_tid})")
                continue
                
            triples = extract_triples_from_post(llm, content)
            if triples:
                # 將三元組資料加入映射表
                triples_data[str(pos_tid)] = {
                    "content": content,
                    "group_id": group_id,
                    "label": label,
                    "note": note,
                    "triples": triples,
                    "extracted_at": datetime.now().isoformat()
                }
                print(f"已處理 {len(triples)} 筆三元組 (pos_tid={pos_tid})")
            else:
                print(f"未取得三元組 (pos_tid={pos_tid})")
                
            # 每處理完一個貼文就儲存一次，避免資料遺失
            save_triples_to_json(triples_data)
            
    finally:
        engine.dispose()
    
    print(f"處理完成！共處理 {len(triples_data)} 個貼文的三元組資料")


if __name__ == "__main__":
    main() 