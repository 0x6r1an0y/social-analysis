from sqlalchemy import create_engine, text, inspect
import pandas as pd
from datetime import datetime
import os

# 設定資料庫連線（標記資料）
DB_URL = "postgresql+psycopg2://postgres:00000000@localhost:5432/labeling_db"
engine = create_engine(DB_URL)

def export_all_tables_to_excel():
    """匯出所有資料表到Excel檔案"""
    
    # 建立inspector來檢查資料庫結構
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    if not table_names:
        print("資料庫中沒有找到任何資料表")
        return
    
    print(f"發現 {len(table_names)} 個資料表: {table_names}")
    
    # 產生檔案名稱（包含時間戳記）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"database_export_{timestamp}.xlsx"
    
    # 使用ExcelWriter來建立多個工作表
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        for table_name in table_names:
            try:
                print(f"正在匯出資料表: {table_name}")
                
                # 讀取資料表，並處理可能的資料問題
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql(query, engine)
                
                # 處理資料中的問題字元，特別是Excel不支援的字元
                for col in df.columns:
                    if df[col].dtype == 'object':  # 只處理文字欄位
                        # 將NaN轉為空字串
                        df[col] = df[col].fillna('')
                        # 轉為字串並移除Excel不支援的控制字元
                        df[col] = df[col].astype(str).str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', regex=True)
                        # 處理可能造成問題的URL（在前面加上單引號讓Excel視為文字）
                        df[col] = df[col].str.replace(r'(https?://[^\s]+)', r"'\1", regex=True)
                        # 限制單一儲存格的長度（Excel單一儲存格上限約32,767字元）
                        df[col] = df[col].str[:30000]  # 保留一些緩衝空間
                
                # 工作表名稱不能超過31個字元，且不能包含特殊字元
                # 移除所有可能造成問題的字元
                import re
                sheet_name = re.sub(r'[^\w\s-]', '_', table_name)  # 只保留字母、數字、空格、連字號
                sheet_name = sheet_name.replace(' ', '_')  # 空格改為底線
                sheet_name = sheet_name[:31]  # 限制長度
                
                # 寫入Excel工作表
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"  - 資料表 '{table_name}' 匯出完成，共 {len(df)} 筆資料")
                
            except Exception as e:
                print(f"  - 匯出資料表 '{table_name}' 時發生錯誤: {str(e)}")
                continue
    
    print(f"\n匯出完成！檔案儲存為: {excel_filename}")
    print(f"檔案大小: {os.path.getsize(excel_filename) / 1024:.2f} KB")

def export_specific_table_to_excel(table_name):
    """匯出指定資料表到Excel檔案"""
    
    try:
        print(f"正在匯出資料表: {table_name}")
        
        # 讀取指定資料表，並處理資料問題
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        # 處理資料中的問題字元
        for col in df.columns:
            if df[col].dtype == 'object':  # 只處理文字欄位
                # 將NaN轉為空字串
                df[col] = df[col].fillna('')
                # 移除Excel不支援的控制字元
                df[col] = df[col].astype(str).str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', regex=True)
                # 處理URL（在前面加上單引號）
                df[col] = df[col].str.replace(r'(https?://[^\s]+)', r"'\1", regex=True)
                # 限制單一儲存格的長度
                df[col] = df[col].str[:30000]
        
        # 產生檔案名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"{table_name}_{timestamp}.xlsx"
        
        # 匯出到Excel
        df.to_excel(excel_filename, index=False)
        
        print(f"匯出完成！")
        print(f"檔案名稱: {excel_filename}")
        print(f"資料筆數: {len(df)}")
        print(f"欄位數量: {len(df.columns)}")
        print(f"檔案大小: {os.path.getsize(excel_filename) / 1024:.2f} KB")
        
        return excel_filename
        
    except Exception as e:
        print(f"匯出資料表 '{table_name}' 時發生錯誤: {str(e)}")
        return None

def list_tables():
    """列出資料庫中的所有資料表"""
    
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        if table_names:
            print("資料庫中的資料表:")
            for i, table_name in enumerate(table_names, 1):
                # 取得資料表的基本資訊
                try:
                    result = engine.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = result.scalar()
                    print(f"{i}. {table_name} ({row_count} 筆資料)")
                except:
                    print(f"{i}. {table_name}")
        else:
            print("資料庫中沒有找到任何資料表")
            
        return table_names
        
    except Exception as e:
        print(f"取得資料表列表時發生錯誤: {str(e)}")
        return []

def export_with_custom_query(query, filename_prefix="custom_query"):
    """使用自訂SQL查詢匯出資料"""
    
    try:
        print(f"執行自訂查詢...")
        print(f"查詢語句: {query}")
        
        # 執行查詢並處理資料
        df = pd.read_sql(query, engine)
        
        # 處理資料中的問題字元
        for col in df.columns:
            if df[col].dtype == 'object':  # 只處理文字欄位
                # 移除Excel不支援的控制字元
                df[col] = df[col].astype(str).str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', regex=True)
                # 限制單一儲存格的長度
                df[col] = df[col].str[:32000]
        
        # 產生檔案名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
        
        # 匯出到Excel
        df.to_excel(excel_filename, index=False)
        
        print(f"匯出完成！")
        print(f"檔案名稱: {excel_filename}")
        print(f"資料筆數: {len(df)}")
        print(f"欄位數量: {len(df.columns)}")
        
        return excel_filename
        
    except Exception as e:
        print(f"執行自訂查詢時發生錯誤: {str(e)}")
        return None

if __name__ == "__main__":
    print("=== 資料庫匯出工具 ===\n")
    
    # 測試連線
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ 資料庫連線成功\n")
    except Exception as e:
        print(f"✗ 資料庫連線失敗: {str(e)}")
        exit(1)
    
    # 列出所有資料表
    tables = list_tables()
    
    if not tables:
        print("沒有資料表可以匯出")
        exit(1)
    
    print("\n請選擇匯出方式:")
    print("1. 匯出所有資料表到一個Excel檔案（多個工作表）")
    print("2. 匯出單一資料表")
    print("3. 使用自訂SQL查詢匯出")
    
    choice = input("\n請輸入選項 (1-3): ").strip()
    
    if choice == "1":
        export_all_tables_to_excel()
        
    elif choice == "2":
        print("\n可用的資料表:")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
        
        try:
            table_index = int(input("\n請選擇資料表編號: ")) - 1
            if 0 <= table_index < len(tables):
                export_specific_table_to_excel(tables[table_index])
            else:
                print("無效的選項")
        except ValueError:
            print("請輸入有效的數字")
            
    elif choice == "3":
        query = input("\n請輸入SQL查詢語句: ").strip()
        if query:
            filename_prefix = input("請輸入檔案名稱前綴 (預設: custom_query): ").strip()
            if not filename_prefix:
                filename_prefix = "custom_query"
            export_with_custom_query(query, filename_prefix)
        else:
            print("查詢語句不能為空")
    
    else:
        print("無效的選項")