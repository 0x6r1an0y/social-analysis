import pandas as pd

chunk_size = 10**6  # 一百萬筆
seen_pos_tid = set()
output_file = "cleaned_output.csv"

# 先寫入標題（第一次清空）
with open(output_file, "w", encoding="utf-8") as f:
    f.write("pos_tid,post_type,page_category,page_name,page_id,content,created_time,reaction_all,comment_count,share_count,date\n")

total_original = 0
total_kept = 0
total_removed_empty_pos_tid = 0
total_removed_empty_content = 0
total_removed_duplicates = 0

for chunk in pd.read_csv("merged_output.csv", chunksize=chunk_size):

    total_original += len(chunk)

    # 去除 pos_tid 空或空白的
    mask_pos_tid = chunk['pos_tid'].notna() & (chunk['pos_tid'].astype(str).str.strip() != '')
    removed_empty_pos_tid = (~mask_pos_tid).sum()
    total_removed_empty_pos_tid += removed_empty_pos_tid
    chunk = chunk[mask_pos_tid]

    # 去除 content 空或空白的
    mask_content = chunk['content'].notna() & (chunk['content'].astype(str).str.strip() != '')
    removed_empty_content = (~mask_content).sum()
    total_removed_empty_content += removed_empty_content
    chunk = chunk[mask_content]

    # 去除重複 pos_tid (跨 chunk 判重)
    def not_seen(pos_tid):
        if pos_tid in seen_pos_tid:
            return False
        else:
            seen_pos_tid.add(pos_tid)
            return True

    mask_not_dup = chunk['pos_tid'].apply(not_seen)
    removed_dup = (~mask_not_dup).sum()
    total_removed_duplicates += removed_dup
    chunk = chunk[mask_not_dup]

    total_kept += len(chunk)

    # 寫入清洗過的結果，mode="a"代表append
    chunk.to_csv(output_file, mode="a", index=False, header=False, encoding="utf-8")

print(f"原始筆數：{total_original}")
print(f"刪除空主鍵：{total_removed_empty_pos_tid}")
print(f"刪除空 content：{total_removed_empty_content}")
print(f"刪除重複主鍵：{total_removed_duplicates}")
print(f"清洗後筆數：{total_kept}")
