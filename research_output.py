# research_output.py
"""
Research Output Generator for SPIN / NNIF / tSPIN
Outputs publication-ready figures and tables.
Updated for Probabilistic tSPIN.
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =========================================================
# Utilities
# =========================================================

def make_output_dir(base="results") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def _get_col(df: pd.DataFrame, candidates: list) -> str:
    """
    データフレームから候補リストにある最初のカラム名を返す。
    見つからなければ None を返す。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _get_nunique_robust(df: pd.DataFrame, col: str) -> int:
    """
    カラムのユニーク数をカウントする。リスト型が含まれる場合は展開(explode)してカウントする。
    """
    if not col or col not in df.columns:
        return 0
    
    series = df[col].dropna()
    if series.empty:
        return 0
        
    # 最初の要素がリストなら explode してカウント
    first_val = series.iloc[0]
    if isinstance(first_val, list) or isinstance(first_val, np.ndarray):
        try:
            return series.explode().nunique()
        except Exception:
            # explode 失敗時はそのままカウント
            return series.astype(str).nunique()
    else:
        return series.nunique()


# =========================================================
# A. Coverage & Summary
# =========================================================

def save_coverage_summary(
    out: Path,
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    nnif_df: pd.DataFrame,
    neg_users: list,
):
    # カラム名の自動検出
    topic_col = _get_col(nodes_df, ["topic", "user_topic_labels", "user_topic_ids"])
    comm_col = _get_col(nodes_df, ["community", "lp_labels", "label"])

    num_topics = _get_nunique_robust(nodes_df, topic_col)
    
    # コミュニティ数のカウント（リスト対応）
    max_comm_id = 0
    if comm_col:
        for val in nodes_df[comm_col].dropna():
            if isinstance(val, list):
                if val: max_comm_id = max(max_comm_id, max(val))
            elif isinstance(val, (int, float)):
                max_comm_id = max(max_comm_id, int(val))
    num_comms = max_comm_id + 1

    summary = {
        "num_users_total": len(nodes_df),
        "num_negative_users": len(neg_users),
        "negative_ratio": len(neg_users) / max(len(nodes_df), 1),
        "num_edges": len(edges_df),
        "nnif_pairs_success": len(nnif_df),
        "num_topics": num_topics,
        "num_communities": num_comms,
    }

    with open(out / "coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.Series(summary).to_csv(out / "coverage_summary.csv")


# =========================================================
# B. Distribution plots
# =========================================================

def plot_nnif_distributions(out: Path, nnif_df: pd.DataFrame):
    if nnif_df.empty:
        return

    # raw
    plt.figure(figsize=(6, 4))
    plt.hist(nnif_df["nnif"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("NNIF")
    plt.ylabel("Frequency")
    plt.title("Distribution of NNIF Scores")
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out / "fig_nnif_raw.png", dpi=300)
    plt.close()

    # log
    x = nnif_df["nnif"][nnif_df["nnif"] > 0]
    if len(x) > 0:
        x_log = np.log10(x)
        plt.figure(figsize=(6, 4))
        plt.hist(x_log, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel("log10(NNIF)")
        plt.ylabel("Frequency")
        plt.title("Log-Distribution of NNIF Scores")
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out / "fig_nnif_log.png", dpi=300)
        plt.close()


# =========================================================
# C. Topic / Community Size
# =========================================================

def plot_topic_community_sizes(out: Path, nodes_df: pd.DataFrame):
    # カラム名の自動検出
    topic_col = _get_col(nodes_df, ["topic", "user_topic_labels", "user_topic_ids"])
    comm_col = _get_col(nodes_df, ["community", "lp_labels", "label"])
    
    id_col = _get_col(nodes_df, ["id", "did", "user_id"])
    if not id_col:
        nodes_df = nodes_df.reset_index()
        id_col = "index"

    # --- Topic size ---
    if topic_col:
        df_topic = nodes_df[[id_col, topic_col]].dropna()
        if not df_topic.empty and isinstance(df_topic[topic_col].iloc[0], (list, np.ndarray)):
            df_topic = df_topic.explode(topic_col)
        
        topic_counts = df_topic.groupby(topic_col)[id_col].nunique().sort_values(ascending=False)
        topic_counts.to_csv(out / "topic_user_counts.csv")

        plt.figure(figsize=(10, 5))
        topic_counts.head(30).plot(kind="bar", color="steelblue")
        plt.ylabel("# Users Involved")
        plt.xlabel("Topic ID")
        plt.title("Topic Engagement Size (Top 30)")
        plt.tight_layout()
        plt.savefig(out / "fig_topic_size.png", dpi=300)
        plt.close()
    
    # --- Community size ---
    if comm_col:
        df_comm = nodes_df[[id_col, comm_col]].dropna()
        if not df_comm.empty and isinstance(df_comm[comm_col].iloc[0], (list, np.ndarray)):
            df_comm = df_comm.explode(comm_col)

        comm_counts = df_comm.groupby(comm_col)[id_col].nunique().sort_values(ascending=False)
        comm_counts.to_csv(out / "community_user_counts.csv")

        plt.figure(figsize=(8, 4))
        comm_counts.head(20).plot(kind="bar", color="coral")
        plt.ylabel("# Users Belonging")
        plt.xlabel("Community ID")
        plt.title("Community Size Distribution")
        plt.tight_layout()
        plt.savefig(out / "fig_community_size.png", dpi=300)
        plt.close()


# =========================================================
# D. tSPIN Statistics & Visualizations (NEW)
# =========================================================

def save_tspin_topic_statistics(out: Path, summary_df: pd.DataFrame):
    """
    確率的tSPIN計算結果のサマリ(summary_df)を受け取り、
    論文用の図表（ランキング、Intra/Inter比較）を出力する。
    """
    if summary_df.empty:
        print("[WARN] tSPIN summary_df is empty.")
        return

    # 1. 保存 (CSV & LaTeX)
    # 必要なカラムだけ抽出して見やすくする
    out_cols = [
        "topic", "tspin_score", "intra_neg_ratio", "inter_neg_ratio", 
        "intra_flow", "inter_flow", "nnif_sum"
    ]
    # 存在しないカラムは無視
    use_cols = [c for c in out_cols if c in summary_df.columns]
    
    df_sorted = summary_df[use_cols].sort_values("tspin_score", ascending=False)
    
    df_sorted.to_csv(out / "tspin_topic_stats.csv", index=False)
    
    # LaTeX用 (top 20)
    try:
        df_sorted.head(20).to_latex(
            out / "tspin_topic_stats_top20.tex", 
            index=False, 
            float_format="%.3f",
            caption="Top 20 Topics by tSPIN Score"
        )
    except Exception:
        pass

    # 2. tSPIN Score Ranking Plot (Bar Chart)
    # 上位30トピックを表示
    plot_df = df_sorted.head(30).copy()
    plot_df["topic_str"] = plot_df["topic"].astype(str)
    
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["topic_str"], plot_df["tspin_score"], color="#d62728", alpha=0.8)
    plt.xlabel("Topic ID")
    plt.ylabel("tSPIN Score")
    plt.title("tSPIN Score per Topic (Heterogeneity of Conflict)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out / "fig_tspin_ranking.png", dpi=300)
    plt.close()

    # 3. Intra vs Inter Flow Contribution (Stacked Bar Chart)
    # フローの絶対量ではなく「比率」で積み上げると性質（内紛か対立か）が見やすい
    # X軸: Topic (tSPINスコア順), Y軸: Ratio (Intra + Inter = 1.0)
    
    if "intra_neg_ratio" in plot_df.columns and "inter_neg_ratio" in plot_df.columns:
        plt.figure(figsize=(12, 6))
        
        indices = np.arange(len(plot_df))
        width = 0.8
        
        # Intra (Internal Conflict) -> Blue
        p1 = plt.bar(indices, plot_df["intra_neg_ratio"], width, color="#1f77b4", label="Intra (Internal)", alpha=0.8)
        # Inter (External Conflict) -> Orange
        p2 = plt.bar(indices, plot_df["inter_neg_ratio"], width, bottom=plot_df["intra_neg_ratio"], color="#ff7f0e", label="Inter (External)", alpha=0.8)
        
        plt.xlabel("Topic ID (Sorted by tSPIN Score)")
        plt.ylabel("Conflict Composition Ratio")
        plt.title("Intra vs Inter Conflict Composition by Topic")
        plt.xticks(indices, plot_df["topic_str"], rotation=45)
        plt.legend(loc="lower right")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "fig_intra_inter_composition.png", dpi=300)
        plt.close()


def save_global_results(out: Path, tspin_result: dict):
    """
    全体のtSPINスコアや設定値を保存
    """
    lines = []
    lines.append("=== Global tSPIN Results ===")
    if "tspin" in tspin_result:
        lines.append(f"Global tSPIN: {tspin_result['tspin']:.4f}")
    if "intra_neg_ratio" in tspin_result:
        lines.append(f"Global Intra-Neg Ratio: {tspin_result['intra_neg_ratio']:.4f}")
    if "inter_neg_ratio" in tspin_result:
        lines.append(f"Global Inter-Neg Ratio: {tspin_result['inter_neg_ratio']:.4f}")
        
    with open(out / "global_results.txt", "w") as f:
        f.write("\n".join(lines))


# =========================================================
# E. Top NNIF pairs
# =========================================================

def save_top_nnif_pairs(out: Path, tspin_result: dict):
    if "top_pairs" not in tspin_result:
        return
        
    top = tspin_result["top_pairs"]
    if top.empty:
        return

    top.to_csv(out / "top_nnif_pairs.csv", index=False)
    # LaTeX出力
    try:
        top.head(20).to_latex(out / "top_nnif_pairs.tex", index=False, float_format="%.4f")
    except Exception:
        pass


# (既存の import 文の下あたりに追加)
# ...

# =========================================================
# F. Advanced Visualizations (NEW Additions)
# =========================================================

def plot_community_interaction_heatmap(out: Path, nnif_df: pd.DataFrame, nodes_df: pd.DataFrame):
    """
    コミュニティ間の「負の情報流」の強さをヒートマップで可視化する。
    nodes_df['lp_labels'] (list) を展開して集計する。
    """
    if nnif_df.empty or nodes_df.empty:
        return

    # 1. ユーザーIDとコミュニティIDの対応表を作る (Explode)
    # id, lp_labels -> id, comm_id (1対多)
    comm_col = _get_col(nodes_df, ["lp_labels", "community", "label"])
    id_col = _get_col(nodes_df, ["id", "did", "user_id"]) or "id"
    
    if not comm_col:
        return

    # リストを展開して (user, comm) のペアを作る
    user_comm = nodes_df[[id_col, comm_col]].dropna()
    
    # lp_labels がリストか確認して展開
    if not user_comm.empty and isinstance(user_comm[comm_col].iloc[0], (list, np.ndarray)):
        user_comm = user_comm.explode(comm_col)
    
    # 文字列型に統一してマージのキーにする
    user_comm[id_col] = user_comm[id_col].astype(str)
    
    # 2. NNIFデータに結合
    # nnif_df: src, dst, nnif
    df_merged = nnif_df.copy()
    df_merged["src"] = df_merged["src"].astype(str)
    df_merged["dst"] = df_merged["dst"].astype(str)
    
    # Src Community
    df_merged = df_merged.merge(user_comm, left_on="src", right_on=id_col, how="inner").rename(columns={comm_col: "src_comm"})
    # Dst Community
    df_merged = df_merged.merge(user_comm, left_on="dst", right_on=id_col, how="inner").rename(columns={comm_col: "dst_comm"})
    
    # 3. 集計 (Src Comm -> Dst Comm)
    heatmap_data = df_merged.groupby(["src_comm", "dst_comm"])["nnif"].sum().reset_index()
    
    if heatmap_data.empty:
        return

    # 4. ピボットテーブル作成
    pivot = heatmap_data.pivot(index="src_comm", columns="dst_comm", values="nnif").fillna(0)
    
    # 5. 描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".1f", linewidths=.5)
    plt.title("Negative Information Flow between Communities")
    plt.xlabel("Target Community (Receiver)")
    plt.ylabel("Source Community (Sender)")
    plt.tight_layout()
    plt.savefig(out / "fig_community_heatmap.png", dpi=300)
    plt.close()


def save_user_negative_rankings(out: Path, nnif_df: pd.DataFrame):
    """
    ユーザーごとの「負の情報の放出量(Out-degree)」と「受信量(In-degree)」をランキング出力
    """
    if nnif_df.empty:
        return

    # Out-flow (加害/拡散度)
    out_flow = nnif_df.groupby("src")["nnif"].sum().sort_values(ascending=False).head(50)
    out_flow.to_csv(out / "ranking_user_negative_out.csv", header=["total_negative_outflow"])
    
    # In-flow (被害/炎上度)
    in_flow = nnif_df.groupby("dst")["nnif"].sum().sort_values(ascending=False).head(50)
    in_flow.to_csv(out / "ranking_user_negative_in.csv", header=["total_negative_inflow"])


def save_topic_keywords(out: Path, topic_model):
    """
    BERTopicモデルから、各トピックの代表キーワードを出力する
    """
    if topic_model is None:
        return

    try:
        # トピック情報の取得
        freq_df = topic_model.get_topic_info()
        freq_df.to_csv(out / "topic_keywords_info.csv", index=False)
        
        # 見やすい形式でも保存 (Topic ID : Keywords)
        with open(out / "topic_keywords_list.txt", "w", encoding="utf-8") as f:
            for index, row in freq_df.iterrows():
                tid = row['Topic']
                if tid == -1: continue # Outlier
                
                # キーワードリストを取得 (単語, スコア)
                words = topic_model.get_topic(tid)
                if words:
                    word_str = ", ".join([w[0] for w in words[:10]]) # 上位10語
                    f.write(f"Topic {tid}: {word_str}\n")
                    
    except Exception as e:
        print(f"[WARN] Failed to save topic keywords: {e}")


def plot_topic_size_vs_conflict(out: Path, tspin_result: dict, nodes_df: pd.DataFrame):
    """
    「トピックの規模(参加人数)」と「対立度(tSPIN Score)」の相関をプロットする
    """
    if "comm_topic_matrix" not in tspin_result:
        return
    
    stats_df = tspin_result["comm_topic_matrix"] # topic, tspin_score
    if stats_df.empty:
        return
        
    # トピック規模の計算
    topic_col = _get_col(nodes_df, ["topic", "user_topic_labels", "user_topic_ids"])
    id_col = _get_col(nodes_df, ["id", "did", "user_id"]) or "index"
    
    if not topic_col:
        return

    df_topic = nodes_df[[id_col, topic_col]].dropna()
    if not df_topic.empty and isinstance(df_topic[topic_col].iloc[0], (list, np.ndarray)):
        df_topic = df_topic.explode(topic_col)
    
    size_series = df_topic.groupby(topic_col)[id_col].nunique()
    size_df = size_series.reset_index().rename(columns={id_col: "user_count", topic_col: "topic"})
    
    # 結合
    merged = pd.merge(stats_df, size_df, on="topic", how="inner")
    
    if merged.empty:
        return

    # プロット
    plt.figure(figsize=(8, 6))
    plt.scatter(merged["user_count"], merged["tspin_score"], color="purple", alpha=0.7)
    
    # IDをラベル表示
    for i, row in merged.iterrows():
        plt.annotate(str(row["topic"]), (row["user_count"], row["tspin_score"]), fontsize=9, alpha=0.7)
        
    plt.xlabel("Topic Size (Number of Users)")
    plt.ylabel("tSPIN Score (Polarization)")
    plt.title("Topic Size vs Conflict Intensity")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out / "fig_topic_size_vs_score.png", dpi=300)
    plt.close()


# =========================================================
# G. Master API (Updated)
# =========================================================

def export_all_results(
    *,
    tspin_result: dict,
    nnif_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    neg_users: list,
    topic_model=None, # ★ 追加引数
):
    out = make_output_dir()
    print(f"[Research Output] Exporting to: {out.resolve()}")

    # 1. Coverage
    try:
        save_coverage_summary(
            out,
            nodes_df=nodes_df,
            edges_df=edges_df,
            nnif_df=nnif_df,
            neg_users=neg_users,
        )
    except Exception as e:
        print(f"[Research Output] Error in coverage summary: {e}")

    # 2. Distributions
    try:
        plot_nnif_distributions(out, nnif_df)
    except Exception as e:
        print(f"[Research Output] Error in nnif plots: {e}")

    # 3. Sizes
    try:
        plot_topic_community_sizes(out, nodes_df)
    except Exception as e:
        print(f"[Research Output] Error in topic/community plots: {e}")

    # 4. tSPIN Statistics
    if "comm_topic_matrix" in tspin_result:
        try:
            save_tspin_topic_statistics(out, tspin_result["comm_topic_matrix"])
        except Exception as e:
            print(f"[Research Output] Error in tSPIN stats: {e}")

    # 5. Global Stats
    try:
        save_global_results(out, tspin_result)
    except Exception as e:
        print(f"[Research Output] Error in global results: {e}")

    # 6. Top Pairs
    try:
        save_top_nnif_pairs(out, tspin_result)
    except Exception as e:
        print(f"[Research Output] Error in top pairs: {e}")

    # --- NEW ADDITIONS ---
    
    # 7. Community Heatmap
    try:
        plot_community_interaction_heatmap(out, nnif_df, nodes_df)
    except Exception as e:
        print(f"[Research Output] Error in community heatmap: {e}")

    # 8. User Rankings
    try:
        save_user_negative_rankings(out, nnif_df)
    except Exception as e:
        print(f"[Research Output] Error in user rankings: {e}")

    # 9. Topic Keywords
    try:
        if topic_model:
            save_topic_keywords(out, topic_model)
    except Exception as e:
        print(f"[Research Output] Error in topic keywords: {e}")

    # 10. Size vs Score
    try:
        plot_topic_size_vs_conflict(out, tspin_result, nodes_df)
    except Exception as e:
        print(f"[Research Output] Error in size vs score: {e}")

    print(f"[Research Output] Done.")