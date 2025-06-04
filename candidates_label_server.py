import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

# 設定資料庫連線（標記資料）
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
engine = create_engine(DB_URL)

# --- UI：選擇群組 ---
st.title("詐騙貼文人工標記工具")
group_id = st.selectbox("請選擇你的群組編號", list(range(5)))  # 假設 group_id 0~4

# --- 抓出該群組的資料 ---
@st.cache_data
def load_data(group_id):
    query = f"SELECT * FROM candidates WHERE group_id = {group_id}"
    return pd.read_sql(query, engine)

df = load_data(group_id)

# 加欄位：目前進度
if 'label_index' not in st.session_state:
    st.session_state.label_index = 0

# --- 顯示一筆貼文進行標記 ---
def show_labeling_ui(index):
    if index >= len(df):
        st.success("🎉 本組貼文已全部標記完畢！")
        return

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

    # 備註欄位
    note = st.text_input("備註（可選）", value=row.get('note', ''))

    # 按鈕區域
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⬅️ 上一題", disabled=index == 0):
            st.session_state.label_index -= 1
            st.rerun()
    
    with col2:
        if st.button("✅ 是", type="primary"):
            save_label(row["pos_tid"], "是", note)
            st.session_state.label_index += 1
            st.rerun()
    
    with col3:
        if st.button("❌ 否", type="primary"):
            save_label(row["pos_tid"], "否", note)
            st.session_state.label_index += 1
            st.rerun()
    
    with col4:
        if st.button("下一題 ➡️", disabled=is_latest_progress):
            st.session_state.label_index += 1
            st.rerun()

    # 顯示進度
    total = len(df)
    labeled = len(df[df['label'].isin(['是', '否'])])
    st.progress(labeled / total)
    st.caption(f"已完成：{labeled}/{total} 題")

# --- 儲存標記結果 ---
def save_label(pos_tid, label, note):
    global df  # 需要修改全域變數
    update_sql = """
        UPDATE candidates
        SET label = :label, note = :note
        WHERE pos_tid = :pos_tid
    """
    with engine.begin() as conn:
        conn.execute(text(update_sql), {"label": label, "note": note, "pos_tid": pos_tid})
    
    # 更新本地 DataFrame
    df.loc[df['pos_tid'] == pos_tid, 'label'] = label
    df.loc[df['pos_tid'] == pos_tid, 'note'] = note

# --- 確保有 label/note 欄位 ---
with engine.begin() as conn:
    conn.execute(text("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS label TEXT"))
    conn.execute(text("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS note TEXT"))

# --- 啟動 UI ---
show_labeling_ui(st.session_state.label_index)
