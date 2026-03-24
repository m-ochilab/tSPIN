# Calculate_tSPIN_en.py
"""
tSPIN calculation module (Probabilistic Approach) for English Data
GPU Accelerated via PyTorch Tensor Operations (torch.einsum)
Strictly adheres to Eq (3.7) - (3.11) in the thesis.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any

def compute_probabilistic_tspin(
    nnif_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    user_topic_df: pd.DataFrame,
    alpha: float = 0.3,  # 論文 4.1.3節に準拠
    beta: float = 0.7,
    num_topics: int = 20
) -> Dict[str, Any]:
    
    # テンソル演算用のデバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tSPIN] Using device: {device} for high-speed tensor aggregation.")

    # 1. ユーザIDのマッピング作成
    all_users = pd.concat([nnif_df["src"], nnif_df["dst"]]).unique()
    user_to_idx = {str(u): i for i, u in enumerate(all_users)}
    n_users = len(all_users)
    
    if n_users == 0 or nnif_df.empty:
        return {"tspin": 0.0, "comm_topic_matrix": pd.DataFrame(), "detailed_flow": []}

    # 2. P(Topic | User) 行列の構築: q_u(z_k)
    topic_matrix = np.zeros((n_users, num_topics), dtype=np.float32)
    if not user_topic_df.empty and "topic_vector" in user_topic_df.columns:
        user_vec_map = dict(zip(user_topic_df["id"].astype(str), user_topic_df["topic_vector"]))
        for u_str, idx in user_to_idx.items():
            if u_str in user_vec_map:
                vec = user_vec_map[u_str]
                dim = min(len(vec), num_topics)
                topic_matrix[idx, :dim] = vec[:dim]

    # 3. P(Community | User) 行列の構築: P(c|u)
    max_comm_id = 0
    if "lp_labels" in nodes_df.columns:
        for labels in nodes_df["lp_labels"]:
            if isinstance(labels, list) and labels:
                max_comm_id = max(max_comm_id, max(labels))
            elif isinstance(labels, (int, float)) and not pd.isna(labels):
                max_comm_id = max(max_comm_id, int(labels))
    
    n_comms = int(max_comm_id) + 1
    comm_matrix = np.zeros((n_users, n_comms), dtype=np.float32)
    
    if "lp_labels" in nodes_df.columns:
        user_labels_map = dict(zip(nodes_df["id"].astype(str), nodes_df["lp_labels"]))
        for u_str, idx in user_to_idx.items():
            labels = user_labels_map.get(u_str, [])
            if isinstance(labels, list) and labels:
                prob = 1.0 / len(labels)
                for l in labels:
                    if l < n_comms:
                        comm_matrix[idx, l] = prob
            elif isinstance(labels, (int, float)) and not pd.isna(labels):
                comm_matrix[idx, int(labels)] = 1.0

    # 行列をPyTorchテンソルに変換してGPUへ転送
    topic_matrix_t = torch.tensor(topic_matrix, device=device)
    comm_matrix_t = torch.tensor(comm_matrix, device=device)

    # 4. エッジの抽出（否定的情報流 f_uv^{neg} > 0 のみ）
    valid_edges = nnif_df[nnif_df["nnif"] > 0]
    if valid_edges.empty:
        return {"tspin": 0.0, "comm_topic_matrix": pd.DataFrame(), "detailed_flow": []}

    src_indices = valid_edges["src"].astype(str).map(user_to_idx).values
    dst_indices = valid_edges["dst"].astype(str).map(user_to_idx).values
    f_neg_vals = valid_edges["nnif"].values.astype(np.float32)

    u_idx_t = torch.tensor(src_indices, dtype=torch.long, device=device)
    v_idx_t = torch.tensor(dst_indices, dtype=torch.long, device=device)
    f_neg_t = torch.tensor(f_neg_vals, device=device)

    # 各エッジの送信者(u)と受信者(v)の属性行列を取得
    comm_u = comm_matrix_t[u_idx_t]   # (E, C)
    comm_v = comm_matrix_t[v_idx_t]   # (E, C)
    topic_u = topic_matrix_t[u_idx_t] # (E, K)
    topic_v = topic_matrix_t[v_idx_t] # (E, K)

    print("[tSPIN] Executing tensor operations (einsum) for Eq (3.8) and (3.9)...")
    
    # =========================================================
    # ★ 論文 Eq 3.8: Psi_intra (送信側トピック q_u(z_i) のみに依存)
    # =========================================================
    # 分子: sum_{u,v} P(c|u) * P(c|v) * q_u(z_i) * f_neg
    num_intra = torch.einsum('ec,ec,ek,e->ck', comm_u, comm_v, topic_u, f_neg_t)
    # 分母: sum_{u,v} P(c|u) * P(c|v) * q_u(z_i)
    den_intra = torch.einsum('ec,ec,ek->ck', comm_u, comm_v, topic_u)
    psi_intra_t = torch.where(den_intra > 0, num_intra / den_intra, torch.zeros_like(num_intra))

    # =========================================================
    # ★ 論文 Eq 3.9: Psi_inter (送受信双方のトピック q_u(z_i), q_v(z_j) に依存)
    # =========================================================
    # 分子: sum_{u,v} P(c_src|u) * P(c_tgt|v) * q_u(z_i) * q_v(z_j) * f_neg
    num_inter = torch.einsum('ei,ej,ek,el,e->ijkl', comm_u, comm_v, topic_u, topic_v, f_neg_t)
    # 分母: sum_{u,v} P(c_src|u) * P(c_tgt|v) * q_u(z_i) * q_v(z_j)
    den_inter = torch.einsum('ei,ej,ek,el->ijkl', comm_u, comm_v, topic_u, topic_v)
    psi_inter_t = torch.where(den_inter > 0, num_inter / den_inter, torch.zeros_like(num_inter))

    # GPUからCPU(NumPy)へ戻す
    num_intra_np = num_intra.cpu().numpy()
    psi_intra_np = psi_intra_t.cpu().numpy()
    num_inter_np = num_inter.cpu().numpy()
    psi_inter_np = psi_inter_t.cpu().numpy()
    den_inter_np = den_inter.cpu().numpy()

    print("[tSPIN] Building detailed flow breakdown...")
    detailed_flow = []
    
    for c_src in range(n_comms):
        for c_tgt in range(n_comms):
            for z_i in range(num_topics):
                for z_j in range(num_topics):
                    
                    if c_src == c_tgt and z_i == z_j:
                        detailed_flow.append({
                            "src_comm": c_src, "dst_comm": c_tgt,
                            "src_topic": z_i, "dst_topic": z_j,
                            "type": "intra", 
                            "score": float(psi_intra_np[c_src, z_i]), 
                            "flow_sum": float(num_intra_np[c_src, z_i])
                        })
                    
                    elif c_src != c_tgt:
                        detailed_flow.append({
                            "src_comm": c_src, "dst_comm": c_tgt,
                            "src_topic": z_i, "dst_topic": z_j,
                            "type": "inter", 
                            "score": float(psi_inter_np[c_src, c_tgt, z_i, z_j]), 
                            "flow_sum": float(num_inter_np[c_src, c_tgt, z_i, z_j])
                        })

    df_flow = pd.DataFrame(detailed_flow)
    
    # 5. tSPIN スコアの統合 (Eq 3.7)
    final_scores = []
    for c_src in range(n_comms):
        for c_tgt in range(n_comms):
            if c_src == c_tgt: continue 
            for z_i in range(num_topics):
                for z_j in range(num_topics):
                    
                    p_intra = psi_intra_np[c_src, z_i]
                    p_inter = psi_inter_np[c_src, c_tgt, z_i, z_j]
                    
                    tspin_score = alpha * p_intra + beta * p_inter
                    
                    if tspin_score > 0:
                        final_scores.append({
                            "src_comm": c_src, "dst_comm": c_tgt,
                            "src_topic": z_i, "dst_topic": z_j,
                            "psi_intra": float(p_intra),
                            "psi_inter": float(p_inter),
                            "tspin_score": float(tspin_score),
                            "flow_sum": float(num_inter_np[c_src, c_tgt, z_i, z_j]),
                            "expected_edges_D": float(den_inter_np[c_src, c_tgt, z_i, z_j]) # Eq 3.11 の D
                        })

    final_df = pd.DataFrame(final_scores)

    # 6. Global tSPIN (Eq 3.10 の厳密な加重平均)
    global_tspin = 0.0
    if not final_df.empty:
        sum_D = final_df["expected_edges_D"].sum()
        if sum_D > 0:
            global_tspin = (final_df["tspin_score"] * final_df["expected_edges_D"]).sum() / sum_D

    return {
        "tspin": global_tspin,
        "detailed_flow": df_flow.to_dict('records') if not df_flow.empty else [],
        "comm_topic_matrix": final_df,
        "top_pairs": nnif_df.sort_values("nnif", ascending=False).head(20).reset_index(drop=True),
    }

run_tspin = compute_probabilistic_tspin