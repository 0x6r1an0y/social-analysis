import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# 設定資料庫連線（標記資料）
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
engine = create_engine(DB_URL)

# --- 載入資料的函數（帶快取） ---
@st.cache_data
def load_data_from_db(group_id):
    """從sql載入該group的data"""
    query = f"SELECT * FROM candidates WHERE group_id = {group_id}"
    print(f"🔄 載入群組 {group_id} 的sql資料")
    return pd.read_sql(query, engine)

def get_current_data(group_id):
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
        print(f"🔄 切換到群組 {group_id}，題號導向到第{latest_index+1}題")
        st.success(f"已恢復進度到第{latest_index+1}題")
        return db
    
    # 檢查是否為導航動作且需要更新
    is_navigation = st.session_state.get('just_navigated', False)
    if is_navigation and st.session_state.need_update:
        print("📥 導航時檢測到資料需要更新，重新載入...")
        load_data_from_db.clear()  # 清除快取
        st.session_state.need_update = False
        st.session_state.just_navigated = False
        db = load_data_from_db(group_id)
        #st.rerun()
        return db
    
    # 重置導航標記
    if st.session_state.get('just_navigated', False):
        st.session_state.just_navigated = False
    
    # 其他情況使用快取
    return load_data_from_db(group_id)

# --- 儲存標記結果（只更新資料庫） ---
def save_label_only(pos_tid:str, label:str, note:str, group_id):
    """只儲存到資料庫，不重新載入資料"""
    update_sql = """
        UPDATE candidates
        SET label = :label, note = :note
        WHERE pos_tid = :pos_tid
    """
    print(f"💾 儲存標記：{pos_tid} -> {label} from group {group_id} 第{st.session_state.label_index+1}題")
    
    with engine.begin() as conn:
        result = conn.execute(text(update_sql), {"label": label, "note": note, "pos_tid": pos_tid})
        if result.rowcount == 0:
            st.warning(f"警告：沒有找到 pos_tid = {pos_tid} 的記錄")
    
    # 標記需要更新，但不立即載入
    st.session_state.need_update = True

# --- 顯示一筆貼文進行標記 ---
def show_labeling_ui(group_id):
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

def get_latest_progress(df):
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

def show_scam_posts_view():
    """顯示所有被標記為詐騙的貼文"""
    st.markdown("### 📱 詐騙貼文瀏覽")
    
    # 取得所有被標記為詐騙的貼文
    query = """
        SELECT pos_tid, content, label, note, group_id
        FROM candidates 
        WHERE label = '是'
        ORDER BY pos_tid DESC
    """
    scam_posts = pd.read_sql(query, engine)
    
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

def show_post_search():
    """根據 pos_tid 查詢特定貼文"""
    st.markdown("### 🔍 貼文查詢")
    
    # 搜尋輸入框
    pos_tid = st.text_input("請輸入貼文 ID (pos_tid)")
    
    if pos_tid:
        # 查詢貼文
        query = """
            SELECT pos_tid, content, label, note, group_id
            FROM candidates 
            WHERE pos_tid = :pos_tid
        """
        result = pd.read_sql(text(query), engine, params={"pos_tid": pos_tid})
        
        if len(result) == 0:
            st.warning(f"找不到 ID 為 {pos_tid} 的貼文")
            return
        
        post = result.iloc[0]
        
        # 顯示貼文內容
        st.markdown("---")
        st.markdown(f"**貼文 ID：** `{post['pos_tid']}`")
        # 貼文內容（改為純文字顯示）
        st.text_area("貼文內容", post['content'], height=200, disabled=True, label_visibility="collapsed", key=f"scam_posts_search_{post['pos_tid']}")
        
        # 貼文資訊
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"群組：{post['group_id']}")
            if pd.notna(post['label']):
                st.caption(f"標記：{post['label']}")
        with col2:
            if pd.notna(post['note']) and post['note']:
                st.caption(f"備註：{post['note']}")

#======================================================================================

if __name__ == '__main__':
    st.title("詐騙貼文人工標記工具")
    
    # 建立頁籤
    tab1, tab2 = st.tabs(["📝 標記模式", "🔍 瀏覽模式"])
    
    with tab1:
        # 原有的標記功能
        group_id = st.selectbox("請選擇你的群組編號", list(range(5)))
        
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
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS label TEXT"))
            conn.execute(text("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS note TEXT"))
        
        # --- 啟動標記 UI ---
        show_labeling_ui(group_id)
    
    with tab2:
        # 瀏覽模式的子頁籤
        subtab1, subtab2 = st.tabs(["📱 詐騙貼文瀏覽", "🔍 貼文查詢"])
        
        with subtab1:
            show_scam_posts_view()
        
        with subtab2:
            show_post_search()