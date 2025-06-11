import pandas as pd
import os
import argparse
import math
from pathlib import Path

class CSVSplitter:
    def __init__(self, input_file: str, output_dir: str = None):
        """
        CSV 檔案分割器
        
        Args:
            input_file: 輸入的 CSV 檔案路徑
            output_dir: 輸出目錄，預設為與輸入檔案相同目錄
        """
        self.input_file = input_file
        self.output_dir = output_dir or os.path.dirname(input_file)
        self.input_filename = Path(input_file).stem  # 檔名不含副檔名
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def split_by_number_of_files(self, n_files: int):
        """
        將 CSV 拆分成指定數量的檔案
        
        Args:
            n_files: 要分割成的檔案數量
        """
        print(f"正在讀取檔案: {self.input_file}")
        df = pd.read_csv(self.input_file)
        total_rows = len(df)
        
        print(f"總共有 {total_rows} 筆資料，將分割成 {n_files} 個檔案")
        
        # 計算每個檔案應該有多少行
        rows_per_file = math.ceil(total_rows / n_files)
        
        for i in range(n_files):
            start_idx = i * rows_per_file
            end_idx = min((i + 1) * rows_per_file, total_rows)
            
            # 如果開始索引超過總行數，則跳出
            if start_idx >= total_rows:
                break
                
            # 切片資料
            chunk = df.iloc[start_idx:end_idx]
            
            # 生成輸出檔名
            output_filename = f"{self.input_filename}_part_{i+1:03d}.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 儲存檔案
            chunk.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"已建立: {output_filename} (共 {len(chunk)} 筆資料)")
            
        print(f"✅ 分割完成！檔案儲存在: {self.output_dir}")
        
    def split_by_rows_per_file(self, rows_per_file: int):
        """
        按每個檔案的行數來分割 CSV
        
        Args:
            rows_per_file: 每個檔案的行數
        """
        print(f"正在讀取檔案: {self.input_file}")
        
        # 使用 chunk 讀取大檔案，節省記憶體
        chunk_iter = pd.read_csv(self.input_file, chunksize=rows_per_file)
        
        file_count = 0
        total_processed = 0
        
        for chunk in chunk_iter:
            file_count += 1
            
            # 生成輸出檔名
            output_filename = f"{self.input_filename}_part_{file_count:03d}.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 儲存檔案
            chunk.to_csv(output_path, index=False, encoding='utf-8-sig')
            total_processed += len(chunk)
            
            print(f"已建立: {output_filename} (共 {len(chunk)} 筆資料)")
            
        print(f"✅ 分割完成！總共處理 {total_processed} 筆資料，分成 {file_count} 個檔案")
        print(f"檔案儲存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='CSV 檔案分割工具')
    parser.add_argument('input_file', help='輸入的 CSV 檔案路徑')
    parser.add_argument('-n', '--num-files', type=int, help='分割成的檔案數量')
    parser.add_argument('-r', '--rows-per-file', type=int, help='每個檔案的行數')
    parser.add_argument('-o', '--output-dir', help='輸出目錄 (預設為輸入檔案所在目錄)')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input_file):
        print(f"❌ 錯誤: 檔案不存在 - {args.input_file}")
        return
        
    # 檢查參數
    if not args.num_files and not args.rows_per_file:
        print("❌ 錯誤: 請指定 -n (檔案數量) 或 -r (每檔行數) 其中一個參數")
        return
        
    if args.num_files and args.rows_per_file:
        print("❌ 錯誤: 不能同時指定 -n 和 -r 參數")
        return
    
    try:
        splitter = CSVSplitter(args.input_file, args.output_dir)
        
        if args.num_files:
            if args.num_files <= 0:
                print("❌ 錯誤: 檔案數量必須大於 0")
                return
            splitter.split_by_number_of_files(args.num_files)
            
        elif args.rows_per_file:
            if args.rows_per_file <= 0:
                print("❌ 錯誤: 每檔行數必須大於 0")
                return
            splitter.split_by_rows_per_file(args.rows_per_file)
            
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")

# 簡單的使用範例
def example_usage():
    """使用範例"""
    print("=== CSV 分割工具使用範例 ===")
    print()
    print("1. 將 data.csv 分割成 5 個檔案:")
    print("   python csv_splitter.py data.csv -n 5")
    print()
    print("2. 每個檔案包含 1000 行:")
    print("   python csv_splitter.py data.csv -r 1000")
    print()
    print("3. 指定輸出目錄:")
    print("   python csv_splitter.py data.csv -n 3 -o ./output")
    print()

if __name__ == "__main__":
    import sys
    
    # 如果沒有參數，顯示使用範例
    if len(sys.argv) == 1:
        example_usage()
    else:
        main() 