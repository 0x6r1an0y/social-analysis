import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

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
        print(f"🔄 切換到群組 {group_id}，題號導向到 {latest_index}")
        st.success(f"已恢復進度到第{latest_index+1}題")
        return db
    
    # 檢查是否為導航動作且需要更新
    is_navigation = st.session_state.get('just_navigated', False)
    if is_navigation and st.session_state.need_update:
        print("📥 導航時檢測到資料需要更新，重新載入...")
        load_data_from_db.clear()  # 清除快取
        st.session_state.need_update = False
        st.session_state.just_navigated = False
        return load_data_from_db(group_id)
    
    # 重置導航標記
    if st.session_state.get('just_navigated', False):
        st.session_state.just_navigated = False
    
    # 其他情況使用快取
    return load_data_from_db(group_id)

# --- 儲存標記結果（只更新資料庫） ---
def save_label_only(pos_tid, label, note, group_id):
    """只儲存到資料庫，不重新載入資料"""
    update_sql = """
        UPDATE candidates
        SET label = :label, note = :note
        WHERE pos_tid = :pos_tid
    """
    print(f"💾 儲存標記：{pos_tid} -> {label} from group {group_id} 第{st.session_state.label_index}題")
    
    with engine.begin() as conn:
        result = conn.execute(text(update_sql), {"label": label, "note": note, "pos_tid": pos_tid})
        if result.rowcount == 0:
            st.warning(f"警告：沒有找到 pos_tid = {pos_tid} 的記錄")
    
    # 標記需要更新，但不立即載入
    st.session_state.need_update = True

# --- 顯示一筆貼文進行標記 ---
def show_labeling_ui(index, group_id):

    row = df.iloc[index]
    st.markdown(f"### 目前第 {index + 1} / {len(df)} 筆")
    st.markdown(f"**pos_tid：** `{row['pos_tid']}`")
    st.text_area("貼文內容", row["content"], height=400, disabled=False)

    # 檢查是否為最新進度（尚未標記的題目）
    is_latest_progress = index == len(df[df['label'].isna() | (df['label'] == '尚未判斷')].index) - 1 if len(df[df['label'].isna() | (df['label'] == '尚未判斷')]) > 0 else False
    
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
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        target_question = st.number_input(
            "", 
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

    # 按鈕區域
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⬅️ 上一題", disabled=index == 0):
            st.session_state.just_navigated = True
            st.session_state.label_index = max(0, st.session_state.label_index - 1)
            st.rerun()
    
    with col2:
        if st.button("✅ 是", type="secondary"):
            save_label_only(row["pos_tid"], "是", note, group_id)
            # 防止超出範圍
            if st.session_state.label_index < len(df) - 1:
                st.session_state.label_index += 1
            st.rerun()
    
    with col3:
        if st.button("❌ 否", type="secondary"):
            save_label_only(row["pos_tid"], "否", note, group_id)
            # 防止超出範圍
            if st.session_state.label_index < len(df) - 1:
                st.session_state.label_index += 1
            st.rerun()
    
    with col4:
        if st.button("下一題 ➡️", disabled=index >= len(df) - 1):
            st.session_state.just_navigated = True
            st.session_state.label_index = min(len(df) - 1, st.session_state.label_index + 1)
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

#======================================================================================

if __name__ == '__main__':

    # --- UI：選擇群組 ---
    st.title("詐騙貼文人工標記工具")
    group_id = st.selectbox("請選擇你的群組編號", list(range(5)))  # 假設 group_id 0~4

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

    # --- 啟動 UI ---
    show_labeling_ui(st.session_state.label_index, group_id)