import pandas as pd
import sys
import os
from pathlib import Path
import re

def clean_sheet_name(sheet_name):
    """
    清理工作表名稱，移除或替換不允許的字符
    
    Excel 工作表名稱限制：
    - 不能超過 31 個字符
    - 不能包含以下字符：\ / ? * [ ] : 
    """
    # 替換不允許的字符為底線
    invalid_chars = r'[\\/?*\[\]:]'
    cleaned_name = re.sub(invalid_chars, '_', str(sheet_name))
    
    # 如果名稱為空，使用默認名稱
    if not cleaned_name.strip():
        cleaned_name = "Sheet1"
    
    # 限制長度為 31 個字符
    return cleaned_name[:31]

def clean_dataframe(df):
    """
    清理 DataFrame 中的特殊字符
    
    參數:
        df (pandas.DataFrame): 需要清理的數據框
    返回:
        pandas.DataFrame: 清理後的數據框
    """
    # 複製 DataFrame 以避免修改原始數據
    df_clean = df.copy()
    
    # 對所有字符串類型的列進行清理
    for col in df_clean.select_dtypes(include=['object']).columns:
        # 將 NaN 值轉換為空字符串
        df_clean[col] = df_clean[col].fillna('')
        # 將所有值轉換為字符串
        df_clean[col] = df_clean[col].astype(str)
        # 替換特殊字符
        df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', x))
    
    return df_clean

def convert_csv_to_excel(csv_path, excel_path=None):
    """
    將 CSV 文件轉換為 Excel 文件
    
    參數:
        csv_path (str): CSV 文件的路徑
        excel_path (str, optional): 輸出的 Excel 文件路徑。如果未提供，將使用與 CSV 相同的名稱
    """
    try:
        # 檢查輸入文件是否存在
        if not os.path.exists(csv_path):
            print(f"錯誤：找不到文件 '{csv_path}'")
            return False
            
        # 如果沒有指定輸出路徑，則使用相同的文件名（但改為 .xlsx 擴展名）
        if excel_path is None:
            excel_path = str(Path(csv_path).with_suffix('.xlsx'))
            
        # 讀取 CSV 文件
        print(f"正在讀取 CSV 文件：{csv_path}")
        try:
            # 嘗試使用 UTF-8 編碼讀取
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # 如果 UTF-8 失敗，嘗試使用 big5 編碼
                df = pd.read_csv(csv_path, encoding='big5')
            except UnicodeDecodeError:
                # 如果都失敗，嘗試使用 cp950 編碼
                df = pd.read_csv(csv_path, encoding='cp950')
        
        # 清理數據
        print("正在清理數據...")
        df_clean = clean_dataframe(df)
        
        # 將數據保存為 Excel 文件
        print(f"正在轉換為 Excel 文件：{excel_path}")
        
        # 創建 ExcelWriter 對象
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 使用清理後的工作表名稱
            sheet_name = clean_sheet_name(Path(csv_path).stem)
            df_clean.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"轉換完成！Excel 文件已保存至：{excel_path}")
        return True
        
    except Exception as e:
        print(f"轉換過程中發生錯誤：{str(e)}")
        return False

def main():
    # 檢查命令行參數
    if len(sys.argv) < 2:
        print("使用方法：")
        print("python csv_to_excel.py <csv文件路徑> [excel文件路徑]")
        print("\n例如：")
        print("python csv_to_excel.py data.csv")
        print("python csv_to_excel.py data.csv output.xlsx")
        return
        
    csv_path = sys.argv[1]
    excel_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_csv_to_excel(csv_path, excel_path)

if __name__ == "__main__":
    main() 