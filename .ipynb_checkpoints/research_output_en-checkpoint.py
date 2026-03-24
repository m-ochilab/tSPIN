# research_output_en.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from sklearn.cluster import SpectralCoclustering # ★追加: 共クラスタリング用

def make_output_dir(base="results") -> Path:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def _get_col(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================================================
# 1. Dataset Statistics
# =========================================================
def export_dataset_statistics(out: Path, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, nnif_df: pd.DataFrame):
    neg_users = len(nodes_df[nodes_df.get("is_negative", False) == True]) if "is_negative" in nodes_df.columns else "N/A"
    
    stats = {
        "Total Users (Nodes)": len(nodes_df),
        "Total Edges": len(edges_df),
        "Negative Users": neg_users,
        "NNIF Calculated Pairs (E-)": len(nnif_df),
        "Average Degree": len(edges_df) / len(nodes_df) if len(nodes_df) > 0 else 0,
    }
    
    with open(out / "table_4_1_dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]).to_csv(out / "table_4_1_dataset_statistics.csv", index=False)


# =========================================================
# 2. Rich Topic Information
# =========================================================
def export_rich_topic_info(out: Path, topic_model=None):
    if topic_model is None: return
    try:
        info_df = topic_model.get_topic_info()
        rep_docs_list = []
        
        with open(out / "topic_details_summary.md", "w", encoding="utf-8") as f:
            f.write("# Topic Details Summary\n")
            f.write("This document provides semantic evidence for each extracted topic.\n\n")
            
            for index, row in info_df.iterrows():
                tid = row['Topic']
                count = row['Count']
                
                if tid == -1:
                    rep_docs_list.append("N/A (Outlier Noise)")
                    continue 
                
                words = topic_model.get_topic(tid)
                if not words:
                    rep_docs_list.append("")
                    continue
                    
                word_str = ", ".join([f"{w[0]}" for w in words[:10]])
                docs = topic_model.get_representative_docs(tid)
                docs_clean = [d.replace('\n', ' ').replace('\t', ' ').strip() for d in docs] if docs else []
                rep_docs_list.append(" | ".join(docs_clean[:3]))
                
                f.write(f"## Topic {tid}: {words[0][0].capitalize()}\n")
                f.write(f"- **Size (Post Count)**: {count}\n")
                f.write(f"- **Top Keywords**: {word_str}\n")
                f.write(f"- **Representative Posts**:\n")
                
                for i, d in enumerate(docs_clean[:3], 1):
                    f.write(f"  {i}. \"{d}\"\n")
                f.write("\n")
                
        info_df['Representative_Docs'] = rep_docs_list
        info_df.to_csv(out / "table_topic_details.csv", index=False)
        
    except Exception as e:
        print(f"[WARN] Failed to export rich topic info: {e}")


# =========================================================
# 3. Topic Flow Breakdown
# =========================================================
def export_topic_flow_breakdown(out: Path, tspin_result: dict, topic_model=None):
    df_flow = pd.DataFrame(tspin_result.get("detailed_flow", []))
    if df_flow.empty: return
    
    flow_agg = df_flow.groupby(["src_topic", "type"])["flow_sum"].sum().unstack(fill_value=0)
    if "intra" not in flow_agg.columns: flow_agg["intra"] = 0.0
    if "inter" not in flow_agg.columns: flow_agg["inter"] = 0.0
    
    flow_agg["total_flow"] = flow_agg["intra"] + flow_agg["inter"]
    flow_agg = flow_agg[flow_agg.index != -1]
    flow_agg = flow_agg.sort_values("total_flow", ascending=False)
    
    def get_tname(tid):
        if topic_model and topic_model.get_topic(tid):
            return f"T{tid}: {topic_model.get_topic(tid)[0][0]}"
        return f"Topic {tid}"
        
    flow_agg["topic_name"] = [get_tname(idx) for idx in flow_agg.index]
    flow_agg.to_csv(out / "topic_flow_absolute_breakdown.csv")
    
    plt.figure(figsize=(14, 7))
    x = np.arange(len(flow_agg))
    plt.bar(x, flow_agg["intra"], label="Intra (Internal Flow)", color="#1f77b4")
    plt.bar(x, flow_agg["inter"], bottom=flow_agg["intra"], label="Inter (External Attack)", color="#ff7f0e")
    
    plt.xticks(x, flow_agg["topic_name"], rotation=45, ha="right", fontsize=9)
    plt.ylabel("Negative Information Flow Amount (Sum of NNIF)")
    plt.title("Absolute Volume of Negative Flow by Topic (Intra vs Inter)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out / "fig_topic_flow_absolute_volume.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 7))
    intra_ratio = flow_agg["intra"] / flow_agg["total_flow"].replace(0, 1)
    inter_ratio = flow_agg["inter"] / flow_agg["total_flow"].replace(0, 1)
    
    plt.bar(x, intra_ratio, label="Intra Ratio", color="#1f77b4", alpha=0.8)
    plt.bar(x, inter_ratio, bottom=intra_ratio, label="Inter Ratio", color="#ff7f0e", alpha=0.8)
    
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label="50% Threshold")
    plt.xticks(x, flow_agg["topic_name"], rotation=45, ha="right", fontsize=9)
    plt.ylabel("Composition Ratio")
    plt.title("Nature of Conflict by Topic (Internal Criticism vs External Attack)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_topic_flow_composition_ratio.png", dpi=300)
    plt.close()


# =========================================================
# 4. 20x20 Clustered Heatmap (Hierarchical Layout)
# ==========================================
def plot_tspin_heatmap_comm_topic_clustered(out: Path, tspin_result: dict, nodes_df: pd.DataFrame, topic_model=None, top_k_topics: int = 10):
    """
    論文用の20x20フルブロックマトリックス (共クラスタリング & 階層的レイアウト)
    Intra/Inter を同時可視化し、指定のPDFデザインを再現します。
    """
    df_flow = pd.DataFrame(tspin_result.get("detailed_flow", []))
    if df_flow.empty: return

    # 主要2コミュニティの特定
    all_labels = []
    for x in nodes_df["lp_labels"].dropna():
        if isinstance(x, list): all_labels.extend(x)
        else: all_labels.append(x)
    
    top_comms = pd.Series(all_labels).value_counts().head(2).index.tolist()
    if len(top_comms) < 2: return
    comm_a, comm_b = top_comms[0], top_comms[1]

    # トピックラベルのクリーンな取得（上位1単語をキャピタライズ）
    topic_labels = {}
    if topic_model:
        info = topic_model.get_topic_info()
        for _, row in info.iterrows():
            tid = row['Topic']
            if tid == -1: continue
            words = topic_model.get_topic(tid)
            if words:
                topic_labels[tid] = words[0][0].capitalize()

    # 有効なトピックを抽出（上位K件）
    valid_topics = sorted([t for t in df_flow["src_topic"].unique() if t != -1])
    valid_topics = valid_topics[:top_k_topics]
    K = len(valid_topics)
    if K == 0: return

    # 2K x 2K (20x20) マトリックスの初期化
    matrix_size = 2 * K
    data = np.zeros((matrix_size, matrix_size))
    
    clean_topic_names = [topic_labels.get(t, f"Topic {t}") for t in valid_topics]
    
    # データの埋め込み (Intra A, Inter A->B, Inter B->A, Intra B)
    for i, t_src in enumerate(valid_topics):
        for j, t_dst in enumerate(valid_topics):
            # A -> A (Top-Left)
            val = df_flow[(df_flow["src_comm"]==comm_a) & (df_flow["dst_comm"]==comm_a) & (df_flow["src_topic"]==t_src) & (df_flow["dst_topic"]==t_dst)]["tspin_score"]
            data[i, j] = val.iloc[0] if not val.empty else 0.0
            
            # A -> B (Top-Right)
            val = df_flow[(df_flow["src_comm"]==comm_a) & (df_flow["dst_comm"]==comm_b) & (df_flow["src_topic"]==t_src) & (df_flow["dst_topic"]==t_dst)]["tspin_score"]
            data[i, j + K] = val.iloc[0] if not val.empty else 0.0
            
            # B -> A (Bottom-Left)
            val = df_flow[(df_flow["src_comm"]==comm_b) & (df_flow["dst_comm"]==comm_a) & (df_flow["src_topic"]==t_src) & (df_flow["dst_topic"]==t_dst)]["tspin_score"]
            data[i + K, j] = val.iloc[0] if not val.empty else 0.0
            
            # B -> B (Bottom-Right)
            val = df_flow[(df_flow["src_comm"]==comm_b) & (df_flow["dst_comm"]==comm_b) & (df_flow["src_topic"]==t_src) & (df_flow["dst_topic"]==t_dst)]["tspin_score"]
            data[i + K, j + K] = val.iloc[0] if not val.empty else 0.0

    # 共クラスタリングの適用
    df_matrix = pd.DataFrame(data, index=clean_topic_names + clean_topic_names, columns=clean_topic_names + clean_topic_names)
    n_clusters = max(2, K // 2)
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
    model.fit(df_matrix.values)

    # 陣営(0=Comm A, 1=Comm B)を維持しつつクラスタラベルでソート
    camps = np.array([0 if i < K else 1 for i in range(matrix_size)])
    row_idx = np.lexsort((model.row_labels_, camps))
    col_idx = np.lexsort((model.column_labels_, camps))

    df_biclustered = df_matrix.iloc[row_idx, col_idx]

    # プロット設定 (Outside-In Hierarchical Layout)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.set(style="white", font="sans-serif")

    sns.heatmap(df_biclustered, 
                cmap='YlOrRd', 
                annot=True, 
                fmt='.2f', 
                annot_kws={"size": 9},
                linewidths=.5,
                cbar_kws={'label': 'Conflict Intensity (tSPIN Score)', 'pad': 0.04},
                ax=ax)

    # X軸を上部に移動
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # ラベルセット
    ax.set_xticklabels(df_biclustered.columns, rotation=45, ha='left', fontsize=12)
    ax.set_yticklabels(df_biclustered.index, rotation=0, fontsize=12)

    # 4象限の境界線
    ax.axhline(K, color='black', linewidth=3)
    ax.axvline(K, color='black', linewidth=3)

    # OSNEM論文に合わせた陣営名の設定
    name_a = "Ukraine-Leaning"
    name_b = "Russia-Leaning"

    # TARGET ラベル (上部)
    ax.annotate(f'Target: {name_a}', xy=(0.25, 1.0), xycoords='axes fraction',
                xytext=(0, 125), textcoords='offset points',
                ha='center', va='bottom', fontsize=16, weight='bold', annotation_clip=False)

    ax.annotate(f'Target: {name_b}', xy=(0.75, 1.0), xycoords='axes fraction',
                xytext=(0, 125), textcoords='offset points',
                ha='center', va='bottom', fontsize=16, weight='bold', annotation_clip=False)

    # SOURCE ラベル (右部)
    ax.annotate(f'Source: {name_a}', xy=(1.0, 0.75), xycoords='axes fraction',
                xytext=(150, 0), textcoords='offset points',
                ha='left', va='center', fontsize=16, weight='bold', rotation=-90, annotation_clip=False)

    ax.annotate(f'Source: {name_b}', xy=(1.0, 0.25), xycoords='axes fraction',
                xytext=(150, 0), textcoords='offset points',
                ha='left', va='center', fontsize=16, weight='bold', rotation=-90, annotation_clip=False)

    # タイトル (最外部)
    plt.title("Topic-Aware Conflict Matrix Reordered by Co-clustering", 
              fontsize=22, pad=220, weight='bold')

    # 保存
    plt.savefig(out / "fig_4_1_tspin_heatmap_coclustered_20x20.pdf", dpi=300, format="pdf", bbox_inches='tight')
    plt.savefig(out / "fig_4_1_tspin_heatmap_coclustered_20x20.png", dpi=300, bbox_inches='tight')
    plt.close()


# =========================================================
# 5. Asymmetry Table
# =========================================================
def export_asymmetry_table(out: Path, tspin_result: dict, nodes_df: pd.DataFrame, topic_model=None):
    df_flow = pd.DataFrame(tspin_result.get("detailed_flow", []))
    if df_flow.empty: return
    
    all_labels = []
    for x in nodes_df["lp_labels"].dropna():
        if isinstance(x, list): all_labels.extend(x)
        else: all_labels.append(x)
    top_comms = pd.Series(all_labels).value_counts().head(2).index.tolist()
    if len(top_comms) < 2: return
    comm_a, comm_b = top_comms[0], top_comms[1]

    pairs = []
    seen = set()
    
    for _, row in df_flow[(df_flow["src_comm"]==comm_a) & (df_flow["dst_comm"]==comm_b)].iterrows():
        t_src, t_tgt = int(row["src_topic"]), int(row["dst_topic"])
        if t_src == -1 or t_tgt == -1: continue
        
        pair_key = f"{t_src}_{t_tgt}"
        if pair_key in seen: continue
        seen.add(pair_key)
        
        score_ab = row["tspin_score"]
        flow_ab = row["flow_sum"] 
        
        rev = df_flow[(df_flow["src_comm"]==comm_b) & (df_flow["dst_comm"]==comm_a) & 
                      (df_flow["src_topic"]==t_tgt) & (df_flow["dst_topic"]==t_src)]
        score_ba = rev["tspin_score"].iloc[0] if not rev.empty else 0.0
        flow_ba = rev["flow_sum"].iloc[0] if not rev.empty else 0.0
        
        name_src = topic_model.get_topic(t_src)[0][0] if topic_model and topic_model.get_topic(t_src) else str(t_src)
        name_tgt = topic_model.get_topic(t_tgt)[0][0] if topic_model and topic_model.get_topic(t_tgt) else str(t_tgt)

        pairs.append({
            "Source Topic (A)": f"Comm {comm_a} ({name_src})",
            "Target Topic (B)": f"Comm {comm_b} ({name_tgt})",
            "tSPIN (A->B)": score_ab,
            "tSPIN (B->A)": score_ba,
            "Asymmetry Diff": abs(score_ab - score_ba),
            "Flow Volume (A->B)": flow_ab, 
            "Flow Volume (B->A)": flow_ba, 
            "Psi_intra (A)": row["psi_intra"],
            "Psi_inter (A->B)": row["psi_inter"]
        })
        
    res_df = pd.DataFrame(pairs).sort_values("Asymmetry Diff", ascending=False).head(30)
    res_df.to_csv(out / "table_4_2_asymmetry_analysis.csv", index=False)


# =========================================================
# Master Export Execution
# =========================================================
def export_all_results_en(
    *,
    tspin_result: dict,
    nnif_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    neg_users: list,
    topic_model=None,
):
    out = make_output_dir()
    print(f"\n[Research Output] Initiating robust export for peer-review to: {out}")
    
    export_dataset_statistics(out, nodes_df, edges_df, nnif_df)
    export_rich_topic_info(out, topic_model)
    export_topic_flow_breakdown(out, tspin_result, topic_model)
    
    # ★ 20x20フルマトリックス＆共クラスタリング階層配置ヒートマップの生成
    plot_tspin_heatmap_comm_topic_clustered(out, tspin_result, nodes_df, topic_model)
    
    export_asymmetry_table(out, tspin_result, nodes_df, topic_model)
    
    with open(out / "global_tspin_result.txt", "w") as f:
        f.write("=== Global Evaluation ===\n")
        f.write("This value represents the overall polarization of the network (Eq 3.10).\n")
        f.write(f"Global tSPIN Score: {tspin_result.get('tspin', 0.0):.4f}\n")

    print(f"[Research Output] Export Successfully Completed. Ready for Analysis.")