# Calculate_tSPIN.py
"""
tSPIN calculation module (Probabilistic Approach)
Based on Eq(8) and Eq(9) in the paper:
F((c_i, z_k) -> (c_j, z_l)) approx sum_{u,v} NNIF(u,v) * P(c_i|u)P(z_k|u) * P(c_j|v)P(z_l|v)
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def compute_probabilistic_tspin(
    nnif_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    user_topic_df: pd.DataFrame,
    alpha: float = 0.5,
    beta: float = 0.5,
    num_topics: int = 20
) -> Dict[str, Any]:
    """
    確率的重み付けを用いてtSPINスコアを計算する関数。
    
    Args:
        nnif_df: [src, dst, nnif] を含むDataFrame (負の情報流のみをフィルタ済みであること)
        nodes_df: [id, lp_labels] を含むノード情報。lp_labelsは [1, 3] のようなリスト。
        user_topic_df: [id, topic_vector] を含む。topic_vectorは sum=1 のnp.array。
        alpha, beta: tSPINの重み係数。
        num_topics: トピックの総数。
    
    Returns:
        Dict containing global tSPIN score, topic-wise breakdown, etc.
    """
    
    # 1. ユーザIDのマッピング作成 (ID -> 行列インデックス)
    all_users = pd.concat([nnif_df["src"], nnif_df["dst"]]).unique()
    user_to_idx = {str(u): i for i, u in enumerate(all_users)}
    n_users = len(all_users)
    
    if n_users == 0:
        return {"tspin": 0.0, "comm_topic_matrix": pd.DataFrame()}

    # 2. P(Topic | User) 行列の構築: (n_users, num_topics)
    topic_matrix = np.zeros((n_users, num_topics))
    
    # user_topic_df を辞書化
    if not user_topic_df.empty and "topic_vector" in user_topic_df.columns:
        user_vec_map = dict(zip(user_topic_df["id"].astype(str), user_topic_df["topic_vector"]))
        
        for u_str, idx in user_to_idx.items():
            if u_str in user_vec_map:
                vec = user_vec_map[u_str]
                # 次元の不一致を防ぐ
                dim = min(len(vec), num_topics)
                topic_matrix[idx, :dim] = vec[:dim]
    else:
        print("[WARN] user_topic_df is empty or missing 'topic_vector'. T-SPIN will be 0.")

    # 3. P(Community | User) 行列の構築: (n_users, n_comms)
    # ラベル伝搬の結果 (lp_labels) を確率分布に変換
    # 例: lp_labels=[1, 3] -> comm[1]=0.5, comm[3]=0.5
    
    # 最大コミュニティIDを取得
    max_comm_id = 0
    if "lp_labels" in nodes_df.columns:
        for labels in nodes_df["lp_labels"]:
            if isinstance(labels, list) and labels:
                max_comm_id = max(max_comm_id, max(labels))
    
    n_comms = max_comm_id + 1
    comm_matrix = np.zeros((n_users, n_comms))
    
    if "lp_labels" in nodes_df.columns:
        user_labels_map = dict(zip(nodes_df["id"].astype(str), nodes_df["lp_labels"]))
        
        for u_str, idx in user_to_idx.items():
            labels = user_labels_map.get(u_str, [])
            if labels:
                prob = 1.0 / len(labels)
                for l in labels:
                    if l < n_comms:
                        comm_matrix[idx, l] = prob
    
    # 4. エッジごとのフロー分解と集計
    # 各トピックについて、Intra (内部) と Inter (外部) の流量を積算する
    
    topic_stats = {k: {"intra": 0.0, "inter": 0.0} for k in range(num_topics)}
    
    # ベクトル化したいがメモリ消費を抑えるため、エッジごとのループで処理
    # NNIF値は「負の情報流」として正の値が入っている前提
    
    # 高速化のためにDataFrameをnumpy化
    # src_idx, dst_idx, nnif
    edge_data = []
    for row in nnif_df.itertuples():
        s_i = user_to_idx.get(str(row.src))
        d_i = user_to_idx.get(str(row.dst))
        val = float(row.nnif)
        if s_i is not None and d_i is not None and val > 0:
            edge_data.append((s_i, d_i, val))
            
    print(f"[tSPIN] Calculating probabilistic flow for {len(edge_data)} edges...")

    for s_i, d_i, flow in edge_data:
        # このエッジの「コミュニティ一致度」 (Intra確率)
        # P(u, v same comm) = sum_c P(c|u) * P(c|v) = 内積
        comm_dot = np.dot(comm_matrix[s_i], comm_matrix[d_i])
        
        prob_intra = comm_dot
        prob_inter = 1.0 - comm_dot  # 残りは全てInterとみなす
        
        # Topicごとの関与度: P(z_k|u) * P(z_k|v)
        # (n_topics,) のベクトル演算
        topic_weights = flow * topic_matrix[s_i] * topic_matrix[d_i]
        
        # 0より大きい要素だけ加算
        nonzero_indices = np.where(topic_weights > 1e-9)[0]
        for k in nonzero_indices:
            w_k = topic_weights[k]
            topic_stats[k]["intra"] += w_k * prob_intra
            topic_stats[k]["inter"] += w_k * prob_inter

    # 5. スコア算出
    summary_rows = []
    
    for k in range(num_topics):
        intra_flow = topic_stats[k]["intra"]
        inter_flow = topic_stats[k]["inter"]
        total_flow = intra_flow + inter_flow
        
        # トピックごとの正規化（そのトピックの総流量に対する比率）
        if total_flow > 0:
            intra_ratio = intra_flow / total_flow
            inter_ratio = inter_flow / total_flow
        else:
            intra_ratio = 0.0
            inter_ratio = 0.0
            
        # 論文式(9): tSPIN = alpha * intra_ratio + beta * inter_ratio
        score = alpha * intra_ratio + beta * inter_ratio
        
        summary_rows.append({
            "topic": k,
            "nnif_sum": total_flow, # 便宜上 nnif_sum と呼ぶ
            "intra_flow": intra_flow,
            "inter_flow": inter_flow,
            "intra_neg_ratio": intra_ratio, # 互換性のため命名維持
            "inter_neg_ratio": inter_ratio,
            "tspin_score": score
        })
        
    summary_df = pd.DataFrame(summary_rows)
    
    # 全体スコア（トピックごとのフロー量で加重平均するか、単純平均か。論文では特に規定ないが加重平均が自然）
    total_network_flow = summary_df["nnif_sum"].sum()
    if total_network_flow > 0:
        global_tspin = (summary_df["tspin_score"] * summary_df["nnif_sum"]).sum() / total_network_flow
        global_intra = (summary_df["intra_neg_ratio"] * summary_df["nnif_sum"]).sum() / total_network_flow
        global_inter = (summary_df["inter_neg_ratio"] * summary_df["nnif_sum"]).sum() / total_network_flow
    else:
        global_tspin = 0.0
        global_intra = 0.0
        global_inter = 0.0

    # 上位ペア（ログ出力用などに単純にNNIF高い順）
    top_pairs = nnif_df.sort_values("nnif", ascending=False).head(20).reset_index(drop=True)
    
    # 互換性のあるDictを返す
    return {
        "tspin": global_tspin,
        "intra_neg_ratio": global_intra,
        "inter_neg_ratio": global_inter,
        "comm_topic_matrix": summary_df, # 分析用詳細
        "top_pairs": top_pairs,
        "log_distribution": np.log10(nnif_df["nnif"][nnif_df["nnif"]>0].values) if not nnif_df.empty else []
    }

# 互換性のためのエイリアス
run_tspin = compute_probabilistic_tspin