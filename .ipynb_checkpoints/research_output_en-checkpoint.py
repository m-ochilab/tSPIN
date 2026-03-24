# research_output_en.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

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
# 1. Dataset Statistics (論文 表4.1 相当)
# =========================================================
def export_dataset_statistics(out: Path, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, nnif_df: pd.DataFrame):
    """
    データセットの基本統計量を算出し、実験の妥当性・網羅性を証明する。
    """
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
# 2. Rich Topic Information (定性的妥当性の証明・強化版)
# =========================================================
def export_rich_topic_info(out: Path, topic_model=None):
    """
    抽出されたトピックの代表語に加え、トピックの規模(Count)や、
    代表的な投稿テキスト(Representative Docs)を抽出し、意味的妥当性を証明する。
    """
    if topic_model is None: return
    try:
        # BERTopicからトピックの基本情報を取得
        info_df = topic_model.get_topic_info()
        
        # CSV出力用のリスト準備
        rep_docs_list = []
        
        # 論文やレポートにそのまま貼り付けられるMarkdown形式で詳細を出力
        with open(out / "topic_details_summary.md", "w", encoding="utf-8") as f:
            f.write("# Topic Details Summary\n")
            f.write("This document provides semantic evidence for each extracted topic.\n\n")
            
            for index, row in info_df.iterrows():
                tid = row['Topic']
                count = row['Count']
                
                if tid == -1:
                    rep_docs_list.append("N/A (Outlier Noise)")
                    continue 
                
                # キーワードの取得
                words = topic_model.get_topic(tid)
                if not words:
                    rep_docs_list.append("")
                    continue
                    
                word_str = ", ".join([f"{w[0]}" for w in words[:10]])
                
                # 代表的な投稿の取得
                docs = topic_model.get_representative_docs(tid)
                docs_clean = [d.replace('\n', ' ').replace('\t', ' ').strip() for d in docs] if docs else []
                rep_docs_list.append(" | ".join(docs_clean[:3]))
                
                # Markdownへの書き込み
                f.write(f"## Topic {tid}: {words[0][0].capitalize()}\n")
                f.write(f"- **Size (Post Count)**: {count}\n")
                f.write(f"- **Top Keywords**: {word_str}\n")
                f.write(f"- **Representative Posts**:\n")
                
                for i, d in enumerate(docs_clean[:3], 1):
                    f.write(f"  {i}. \"{d}\"\n")
                f.write("\n")
                
        # DFに代表ドキュメントを追加してCSV化
        info_df['Representative_Docs'] = rep_docs_list
        info_df.to_csv(out / "table_topic_details.csv", index=False)
        
    except Exception as e:
        print(f"[WARN] Failed to export rich topic info: {e}")


# =========================================================
# 3. Topic Flow Breakdown (査読対策: Intra/Inter の絶対量)
# =========================================================
def export_topic_flow_breakdown(out: Path, tspin_result: dict, topic_model=None):
    """
    論文 5.1.2節「対立の質」の裏付け。
    各トピックを起点として、どれだけの情報量が内部循環(Intra)し、外部攻撃(Inter)に向けられたか。
    """
    df_flow = pd.DataFrame(tspin_result.get("detailed_flow", []))
    if df_flow.empty: return
    
    # Source Topic ごとに Intra と Inter の Flow Sum を集計
    flow_agg = df_flow.groupby(["src_topic", "type"])["flow_sum"].sum().unstack(fill_value=0)
    if "intra" not in flow_agg.columns: flow_agg["intra"] = 0.0
    if "inter" not in flow_agg.columns: flow_agg["inter"] = 0.0
    
    flow_agg["total_flow"] = flow_agg["intra"] + flow_agg["inter"]
    flow_agg = flow_agg[flow_agg.index != -1] # ノイズトピックを除外
    flow_agg = flow_agg.sort_values("total_flow", ascending=False) # 流量が多い順
    
    def get_tname(tid):
        if topic_model and topic_model.get_topic(tid):
            return f"T{tid}: {topic_model.get_topic(tid)[0][0]}"
        return f"Topic {tid}"
        
    flow_agg["topic_name"] = [get_tname(idx) for idx in flow_agg.index]
    flow_agg.to_csv(out / "topic_flow_absolute_breakdown.csv")
    
    # --- グラフ1: 情報量の絶対値 (Stack Bar) ---
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

    # --- グラフ2: 構成比率 (Ratio) ---
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
# 4. Clustered Heatmap (論文 図4.1)
# =========================================================
def plot_tspin_heatmap_comm_topic_clustered(out: Path, tspin_result: dict, nodes_df: pd.DataFrame, topic_model=None, top_k_topics: int = 15):
    """
    論文図4.1: コミュニティ×トピック間の対立ヒートマップ (共クラスタリング付き)
    """
    df_flow = pd.DataFrame(tspin_result.get("detailed_flow", []))
    if df_flow.empty: return

    # 主要2コミュニティ特定
    all_labels = []
    for x in nodes_df["lp_labels"].dropna():
        if isinstance(x, list): all_labels.extend(x)
        else: all_labels.append(x)
    
    top_comms = pd.Series(all_labels).value_counts().head(2).index.tolist()
    if len(top_comms) < 2: return
    comm_a, comm_b = top_comms[0], top_comms[1]

    # トピックラベル取得
    topic_labels = {}
    if topic_model:
        info = topic_model.get_topic_info()
        for _, row in info.iterrows():
            tid = row['Topic']
            if tid == -1: continue
            words = topic_model.get_topic(tid)
            if words: topic_labels[tid] = f"{tid}: {words[0][0]}"
    
    # A -> B のフロー抽出
    mask = (df_flow["src_comm"] == comm_a) & (df_flow["dst_comm"] == comm_b)
    df_target = df_flow[mask].copy()
    if df_target.empty: return

    # Pivot: Index=Src_Topic, Columns=Dst_Topic
    heatmap_data = df_target.pivot(index="src_topic", columns="dst_topic", values="tspin_score").fillna(0)
    
    valid_topics = sorted([t for t in heatmap_data.index if t != -1])[:top_k_topics]
    heatmap_data = heatmap_data.loc[valid_topics, valid_topics]
    
    xticklabels = [topic_labels.get(t, str(t)) for t in heatmap_data.columns]
    yticklabels = [topic_labels.get(t, str(t)) for t in heatmap_data.index]
    
    heatmap_data.columns = xticklabels
    heatmap_data.index = yticklabels

    # 共クラスタリングによる並べ替え
    cm = sns.clustermap(
        heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
        figsize=(12, 10), row_cluster=True, col_cluster=True
    )
    plt.setp(cm.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(cm.ax_heatmap.get_yticklabels(), rotation=0)
    cm.fig.suptitle(f"tSPIN Interaction Heatmap (Comm {comm_a} -> Comm {comm_b})\nClustered by Conflict Similarity", y=1.02)
    cm.savefig(out / f"fig_4_1_tspin_heatmap_clustered_{comm_a}_to_{comm_b}.png", bbox_inches='tight', dpi=300)
    plt.close()


# =========================================================
# 5. Asymmetry Table (論文 表4.2)
# =========================================================
def export_asymmetry_table(out: Path, tspin_result: dict, nodes_df: pd.DataFrame, topic_model=None):
    """
    論文表4.2: クロス・トピック分析 双方向の対立構造と非対称性
    査読対策として、スコアの根拠となる「Flow Sum(情報量絶対値)」を併記する。
    """
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
        
        # A -> B
        score_ab = row["tspin_score"]
        flow_ab = row["flow_sum"] 
        
        # 逆方向 B -> A
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
    
    try:
        res_df.to_latex(out / "table_4_2_asymmetry.tex", index=False, float_format="%.2f")
    except: pass


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
    
    # 1. Dataset Stats
    export_dataset_statistics(out, nodes_df, edges_df, nnif_df)
    
    # 2. Rich Topic Information (NEW: 査読での定性評価用)
    export_rich_topic_info(out, topic_model)

    # 3. Flow Breakdown (Intra vs Inter Volume)
    export_topic_flow_breakdown(out, tspin_result, topic_model)

    # 4. Clustered Heatmap
    plot_tspin_heatmap_comm_topic_clustered(out, tspin_result, nodes_df, topic_model)
    
    # 5. Asymmetry Table
    export_asymmetry_table(out, tspin_result, nodes_df, topic_model)
    
    # 6. Global SPIN (Baseline Comparison)
    with open(out / "global_tspin_result.txt", "w") as f:
        f.write("=== Global Evaluation ===\n")
        f.write("This value represents the overall polarization of the network (Eq 3.10).\n")
        f.write("Compare this with the original SPIN score to prove global consistency.\n\n")
        f.write(f"Global tSPIN Score: {tspin_result.get('tspin', 0.0):.4f}\n")

    print(f"[Research Output] Export Successfully Completed. Ready for Analysis.")