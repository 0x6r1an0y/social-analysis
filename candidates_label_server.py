import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from candidates_dataloader_to_sql import fetch_candidate_posts
import logging
import datetime
import subprocess
import os

# 建立 logs 目錄（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 設定日誌檔案名稱（使用當前日期）
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_filename = f'logs/candidates_label_{current_date}.log'

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 記錄程式啟動
logger.info("程式啟動")

# 設定資料庫連線（標記資料）
LABELING_DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
SOURCE_DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/social_media_analysis"

# 建立兩個資料庫的連線引擎
labeling_engine = create_engine(LABELING_DB_URL)
source_engine = create_engine(SOURCE_DB_URL)

# 確保有 system_settings 資料表
with labeling_engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key VARCHAR(50) PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """))

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
                source_query = "SELECT pos_tid, content FROM posts WHERE pos_tid = :pos_tid"
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
        if st.button("⬅️ 上一題", disabled = index<=0):
            st.session_state.just_navigated = True
            st.session_state.label_index -= 1
            st.rerun()
    
    with col2:
        if st.button("✅ 是", type="secondary", disabled=(index == len(df))):
            save_label_only(row["pos_tid"], "是", note, group_id)
            # 防止超出範圍
            if not index >= (len(df) - 1): # if index < 799:
                st.session_state.label_index += 1
            st.rerun()
    
    with col3:
        if st.button("❌ 否", type="secondary", disabled=(index == len(df))):
            save_label_only(row["pos_tid"], "否", note, group_id)
            # 防止超出範圍
            if not index >= (len(df) - 1):
                st.session_state.label_index += 1
            st.rerun()
    
    with col4:
        if st.button("下一題 ➡️", disabled = index >= (len(df) - 1)):
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
    for _, post in scam_posts.iterrows():
        with st.container():
            st.markdown("---")
            # 貼文標題
            st.markdown(f"**貼文 ID：** `{post['pos_tid']}`")
            # 貼文內容
            #st.markdown(post['content'])
            # 貼文內容（改為純文字顯示）
            st.text_area("貼文內容", post['content'], height=200, disabled=True, label_visibility="collapsed", key=f"scam_posts_{post['pos_tid']}")
            # 貼文資訊
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"群組：{post['group_id']}")
            with col2:
                if pd.notna(post['note']) and post['note']:
                    st.caption(f"備註：{post['note']}")

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
    if st.button("🔄 生成詞彙分析圖表", type="primary"):
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
                st.text_area("貼文內容", post['content'], height=200, disabled=True, label_visibility="collapsed", key=f"scam_posts_search_{post['pos_tid']}")
                
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
                FROM posts 
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
                    st.text_area("貼文內容", post['content'], height=200, disabled=True, label_visibility="collapsed", key=f"source_posts_search_{post['pos_tid']}")
                    
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
    if 'label_message' not in st.session_state:
        st.session_state.label_message = None
    if 'label_message_pos_tid' not in st.session_state:
        st.session_state.label_message_pos_tid = None
    
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
                search_logic=search_logic
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
                st.text_area("貼文內容", row['content'], height=200, disabled=True, label_visibility="collapsed", key=f"keyword_search_{row['pos_tid']}")
                
                # 標記區域
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("✅ 是", key=f"yes_{row['pos_tid']}"):
                        try:
                            save_label_only(row['pos_tid'], "是", "", 999)
                            st.session_state.label_message = "已標記為「是」"
                            st.session_state.label_message_pos_tid = row['pos_tid']
                            st.rerun()
                        except Exception as e:
                            st.error(f"標記失敗：{str(e)}")
                with col2:
                    if st.button("❌ 否", key=f"no_{row['pos_tid']}"):
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
            if st.button("⬅️ 上一頁", disabled=st.session_state.search_page <= 0):
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
                key="page_input"
            )
            # 當頁碼改變時跳轉
            if target_page != st.session_state.search_page + 1:
                st.session_state.search_page = target_page - 1
                # 使用 JavaScript 跳轉到頁面頂部
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()
        with col3:
            if st.button("下一頁 ➡️", disabled=st.session_state.search_page >= total_pages - 1):
                st.session_state.search_page += 1
                # 使用 JavaScript 跳轉到頁面頂部
                st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
                st.rerun()

#======================================================================================

if __name__ == '__main__':
    st.title("詐騙貼文人工標記工具")
    
    # 建立頁籤
    tab1, tab2, tab3 = st.tabs(["📝 標記模式", "👀 瀏覽模式", "🔑 關鍵字搜尋"])
    
    with tab1:
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
    
    with tab2:
        # 瀏覽模式的子頁籤
        subtab1, subtab2, subtab3 = st.tabs(["📱 詐騙貼文瀏覽", "📖 貼文查詢", "📊 詞彙分析"])
        
        with subtab1:
            show_scam_posts_view()
        
        with subtab2:
            show_post_search()
            
        with subtab3:
            show_word_analysis()
    
    with tab3:
        show_keyword_search()