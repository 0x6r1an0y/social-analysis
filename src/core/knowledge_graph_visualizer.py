import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Noto Sans TC', 'Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定區
INPUT_FILE = "knowledge_graph.json"
OUTPUT_DIR = "knowledge_graphs"

# 可調整的參數
DEGREE_THRESHOLD = 3  # 連接度閾值，低於此值的節點會被過濾掉
NODE_FONT_SIZE = 20   # 節點字體大小
EDGE_FONT_SIZE = 20   # 邊標籤字體大小
TITLE_FONT_SIZE = 30  # 標題字體大小

def load_triples_data():
    """載入三元組資料"""
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤：找不到 {INPUT_FILE} 檔案")
        return {}
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_combined_knowledge_graph(all_data, degree_threshold=1):
    """從所有文章的三元組建立綜合知識圖譜，並過濾低連接度節點"""
    G = nx.DiGraph()
    
    # 收集所有三元組並加入圖譜
    for pos_tid, data in all_data.items():
        triples = data["triples"]
        for s, p, o in triples:
            G.add_edge(s, o, label=p)
    
    # 過濾低連接度的節點
    if degree_threshold > 1:
        print(f"過濾連接度 < {degree_threshold} 的節點...")
        original_nodes = len(G.nodes())
        
        # 計算節點度數
        node_degrees = dict(G.degree())
        
        # 找出要移除的節點
        nodes_to_remove = [node for node, degree in node_degrees.items() if degree < degree_threshold]
        
        # 移除節點
        G.remove_nodes_from(nodes_to_remove)
        
        print(f"移除了 {len(nodes_to_remove)} 個節點（從 {original_nodes} 個減少到 {len(G.nodes())} 個）")
    
    return G

def adjust_node_positions(pos, node_sizes, min_distance=0.1):
    """調整節點位置以避免重疊"""
    nodes = list(pos.keys())
    positions = np.array([pos[node] for node in nodes])
    
    # 計算節點大小對應的距離
    size_ratios = np.array([size / 10000 for size in node_sizes])  # 標準化大小
    
    # 迭代調整位置
    for iteration in range(50):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # 計算兩節點間的距離
                dist = np.linalg.norm(positions[i] - positions[j])
                
                # 計算最小所需距離
                min_required_dist = (size_ratios[i] + size_ratios[j]) * min_distance
                
                if dist < min_required_dist:
                    # 計算移動方向
                    direction = positions[i] - positions[j]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        
                    # 移動節點
                    move_distance = (min_required_dist - dist) / 2
                    positions[i] += direction * move_distance
                    positions[j] -= direction * move_distance
                    moved = True
        
        if not moved:
            break
    
    # 更新位置字典
    new_pos = {}
    for i, node in enumerate(nodes):
        new_pos[node] = positions[i]
    
    return new_pos

def visualize_combined_graph(all_data, output_dir, degree_threshold=1):
    """繪製所有文章的三元組綜合知識圖譜"""
    # 建立綜合圖譜
    G = create_combined_knowledge_graph(all_data, degree_threshold)
    
    if len(G.nodes()) == 0:
        print("綜合圖譜沒有節點")
        return
    
    # 計算節點重要性（度數）
    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1
    
    # 統計邊的權重（相同關係的數量）
    edge_weights = defaultdict(int)
    for u, v, data in G.edges(data=True):
        edge_weights[(u, v, data['label'])] += 1
    
    # 繪圖設定
    plt.figure(figsize=(30, 24))  # 增加畫布大小
    
    # 使用更好的布局算法，增加節點間距
    pos = nx.spring_layout(G, k=5, iterations=50, seed=42)  # 增加k值和迭代次數
    
    # 如果節點仍然重疊，使用更分散的布局
    if len(G.nodes()) > 20:
        pos = nx.kamada_kawai_layout(G, scale=2.0)  # 增加scale參數
    elif len(G.nodes()) > 10:
        pos = nx.spring_layout(G, k=8, iterations=100, seed=42)  # 進一步增加間距
    
    # 根據度數設定節點大小和顏色
    node_sizes = [6000 + (node_degrees[node] / max_degree) * 8000 for node in G.nodes()]
    
    # 調整節點位置以避免重疊
    pos = adjust_node_positions(pos, node_sizes, min_distance=0.15)
    
    # 使用亮色背景配黑色文字
    # 根據連接度選擇不同的亮色
    node_colors = []
    for node in G.nodes():
        degree_ratio = node_degrees[node] / max_degree
        if degree_ratio > 0.7:
            node_colors.append("#F9A4A4")  # 淺紅色 - 高連接度
        elif degree_ratio > 0.4:
            node_colors.append("#BCDFFF")  # 淺藍色 - 中連接度
        else:
            node_colors.append("#DFFFB8")  # 淺綠色 - 低連接度
    
    # 繪製節點
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors, 
            node_size=node_sizes, 
            font_size=NODE_FONT_SIZE, 
            font_family='Noto Sans TC',
            font_weight='bold',
            font_color='black',  # 確保文字是黑色
            edge_color='gray',
            arrows=True,
            arrowsize=30,  # 增加箭頭大小
            arrowstyle='->',
            width=2.5,  # 增加邊的寬度
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))  # 為節點文字添加背景
    
    # 繪製邊上的文字 (關係)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=edge_labels, 
                                font_family='Noto Sans TC', 
                                font_size=EDGE_FONT_SIZE,
                                font_weight='bold',
                                font_color='darkblue',  # 深藍色文字
                                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95, edgecolor='lightgray'))
    
    # 計算統計資訊
    total_triples = sum(len(data["triples"]) for data in all_data.values())
    
    plt.title(f"詐騙貼文綜合知識圖譜\n共 {len(all_data)} 個貼文，{total_triples} 個三元組，{len(G.nodes())} 個實體\n(連接度閾值: {degree_threshold})", 
              fontsize=TITLE_FONT_SIZE, pad=40, fontweight='bold', fontfamily='Noto Sans TC')
    plt.axis('off')
    
    # 儲存圖片
    output_path = os.path.join(output_dir, f"combined_knowledge_graph_threshold_{degree_threshold}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"已儲存綜合圖譜：{output_path}")
    return G

def analyze_graph_statistics(all_data, G, degree_threshold=1):
    """分析圖譜統計資訊"""
    print(f"\n=== 綜合知識圖譜統計分析 (連接度閾值: {degree_threshold}) ===")
    print(f"總貼文數：{len(all_data)}")
    
    total_triples = 0
    all_triples = []
    for pos_tid, data in all_data.items():
        triples = data["triples"]
        total_triples += len(triples)
        all_triples.extend(triples)
    
    print(f"總三元組數：{total_triples}")
    print(f"圖譜節點數：{len(G.nodes())}")
    print(f"圖譜邊數：{len(G.edges())}")
    
    # 統計關係類型
    relations = defaultdict(int)
    for s, p, o in all_triples:
        relations[p] += 1
    
    print(f"\n關係類型統計：")
    for relation, count in sorted(relations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {relation}: {count} 次")
    
    # 統計重要實體（度數最高的節點）
    node_degrees = dict(G.degree())
    top_entities = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print(f"\n重要實體（連接度最高）：")
    for entity, degree in top_entities:
        print(f"  {entity}: {degree} 個連接")
    
    print(f"\n平均每個貼文的三元組數：{total_triples / len(all_data):.2f}")

def generate_multiple_thresholds(all_data, output_dir):
    """生成多個不同閾值的圖譜"""
    thresholds = [2, 3, 4]  # 可以調整這些閾值
    
    for threshold in thresholds:
        print(f"\n生成連接度閾值 {threshold} 的圖譜...")
        G = visualize_combined_graph(all_data, output_dir, threshold)
        if G:
            analyze_graph_statistics(all_data, G, threshold)

def main():
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 載入資料
    all_data = load_triples_data()
    
    if not all_data:
        print("沒有資料可處理")
        return
    
    # 詢問用戶想要的閾值
    print("請選擇選項：")
    print("1. 生成單一閾值的圖譜")
    print("2. 生成多個閾值的圖譜")
    
    choice = input("請輸入選項 (1 或 2): ").strip()
    
    if choice == "1":
        threshold = int(input(f"請輸入連接度閾值 (預設: {DEGREE_THRESHOLD}): ") or DEGREE_THRESHOLD)
        print(f"開始繪製連接度閾值 {threshold} 的綜合知識圖譜...")
        G = visualize_combined_graph(all_data, OUTPUT_DIR, threshold)
        if G:
            analyze_graph_statistics(all_data, G, threshold)
    elif choice == "2":
        print("開始生成多個閾值的圖譜...")
        generate_multiple_thresholds(all_data, OUTPUT_DIR)
    else:
        print("無效選項，使用預設閾值...")
        G = visualize_combined_graph(all_data, OUTPUT_DIR, DEGREE_THRESHOLD)
        if G:
            analyze_graph_statistics(all_data, G, DEGREE_THRESHOLD)
    
    print(f"\n完成！圖譜已儲存到 {OUTPUT_DIR} 目錄")

if __name__ == "__main__":
    main() 