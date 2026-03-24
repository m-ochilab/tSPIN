#!/usr/bin/env python3
"""
run_spin_cli_en.py
Strictly follows the methodology and parameters from the bachelor thesis for English data.
★ Update: Fetch actual repost counts to strictly follow Section 4.1.1 (Seed Selection based on High Engagement).
"""

from datetime import datetime
import pandas as pd
from elasticsearch import Elasticsearch

# --- モジュールのインポート ---
from research_matae_tokenizer_updated import (
    run_spin_with_posts_df_a,
    build_spin_network_from_sampled_edges,
    propagate_multilabel_from_seeds_threshold,
    merge_networks_by_common_users,
    _extract_post_cid,
    _terms_agg_counts_by_field # ★ リポスト数集計用に追加
)
from topic_pipeline_tokenizer_updated_en import (
    run_bertopic_on_posts,
    compute_user_topic_profiles,
    attach_topics_to_nodes,
    english_sns_tokenizer 
)
from Calculate_NNIF_tokenizer_updated_en import ( 
    build_information_flow_edges,
    build_user_timelines_negative_only, 
    build_neighbors_from_edges,
    build_user_token_sequences,
    compute_nnif_for_edges,
)
from Calculate_tSPIN_en import compute_probabilistic_tspin
from research_output_en import export_all_results_en

# GPU判定モジュールの読み込み
from negative_judge_en import load_english_negative_judge

ES_ADDR = "http://cicero.csis.oita-u.ac.jp:9200"

def get_post_texts_en(es, query_body, index="postindex-*", size=2000):
    rows = []
    res = es.search(index=index, body=query_body, size=size, request_timeout=300)
    for h in res["hits"]["hits"]:
        src = h["_source"]
        record = src.get("commit", {}).get("record", {})
        text = record.get("text", "")
        if len(text) < 10: continue 
        
        rows.append({
            "did": src.get("did"),
            "cid": _extract_post_cid(src),
            "text": text,
            "created_at": record.get("createdAt"),
        })
        
    df = pd.DataFrame(rows)
    
    # ★ 論文4.1.1節「高いエンゲージメントを持つアカウントをシードに」を厳密に実装
    if not df.empty:
        cids = df["cid"].dropna().astype(str).tolist()
        
        # 実際のリポスト数を Elasticsearch から一括集計
        repost_counts = _terms_agg_counts_by_field(
            es, index="repostindex-*", ids=cids, field="commit.record.subject.cid.keyword"
        )
        
        df["repost_count"] = df["cid"].astype(str).map(repost_counts).fillna(0).astype(int)
    else:
        df["repost_count"] = pd.Series(dtype="int64")
        
    return df

def main():
    # 処理開始前にGPUにモデルをロードしておく
    print("[System] Initializing GPU Models...")
    load_english_negative_judge()

    es = Elasticsearch(hosts=[ES_ADDR], request_timeout=1200)

    # 論文実験設定に準拠
    QUERY_A = "iran"
    QUERY_B = "america"
    START_DATE = datetime(2025, 11, 20)
    END_DATE = datetime(2026, 3, 8)
    
    def build_query(q):
        return {
            "_source": ["did", "commit.rkey", "commit.cid", "commit.record.text", "commit.record.createdAt", "commit.record.langs"],
            "query": {
                "bool": {
                    "must": [
                        {"match": {"commit.record.text": {"query": q, "operator": "and"}}},
                        {"match": {"commit.record.langs": "en"}}
                    ],
                    "filter": [{"range": {"commit.record.createdAt": {
                        "gte": START_DATE.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lte": END_DATE.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }}}],
                    "must_not": [{"exists": {"field": "commit.record.reply"}}]
                }
            }
        }

    print("\n[1] Fetching Data...")
    df_a = get_post_texts_en(es, build_query(QUERY_A))
    df_b = get_post_texts_en(es, build_query(QUERY_B))
    
    if df_a.empty or df_b.empty:
        print("[Error] Data could not be fetched. Please check ES indices or query dates.")
        return

    print("\n[2] SPIN Sampling (Network Construction)...")
    res_a = run_spin_with_posts_df_a(es, df_a, START_DATE, END_DATE)
    res_b = run_spin_with_posts_df_a(es, df_b, START_DATE, END_DATE)
    
    nodes_a, edges_a = build_spin_network_from_sampled_edges(res_a["seed_accounts"], res_a["sampled_edges"])
    nodes_b, edges_b = build_spin_network_from_sampled_edges(res_b["seed_accounts"], res_b["sampled_edges"])
    merged_nodes, merged_edges, _ = merge_networks_by_common_users(nodes_a, edges_a, nodes_b, edges_b)

    print("\n[3] Community Detection (Label Propagation)...")
    lp_nodes = propagate_multilabel_from_seeds_threshold(
        merged_nodes, merged_edges, 
        seed_ids=res_a["seed_accounts"] + res_b["seed_accounts"],
        threshold=0.3, max_iter=100
    )

    print("\n[4] Information Flow (Edge Extraction) & Fetching Timelines...")
    # 先にエッジと対象ユーザーを特定
    edges_info = build_information_flow_edges(
        es, start_dt=START_DATE, end_dt=END_DATE,
        allowed_users=set(lp_nodes["id"]),
        check_content_negativity=False 
    )
    relevant_users = set(edges_info["src"]).union(set(edges_info["dst"]))
    
    # GPUによる超高速バッチ推論（ネットワーク内の全ユーザーからネガティブ投稿だけ抽出）
    timelines = build_user_timelines_negative_only(
        es, start_dt=START_DATE, end_dt=END_DATE, allowed_users=relevant_users
    )
    
    # ★統計出力用: ノードにネガティブ判定フラグを追加（これでN/Aが直ります）
    negative_user_ids = set(timelines.keys())
    lp_nodes["is_negative"] = lp_nodes["id"].astype(str).isin(negative_user_ids)

    print("\n[5] Topic Modeling (BERTopic: SBERT on GPU) on Network Posts...")
    # ★修正: timelines (数千人分の投稿) をBERTopic用のDataFrameに変換する
    records = []
    for did, posts in timelines.items():
        for dt, txt in posts:
            records.append({"id": did, "did": did, "text": txt, "created_at": dt})
    
    target_posts = pd.DataFrame(records)
    if target_posts.empty:
        print("[Error] No negative posts found. Cannot run BERTopic.")
        return

    # 数万件のテキストを使って、高精度なトピックモデルを構築
    posts_topics, topic_model = run_bertopic_on_posts(target_posts, text_col="text")
    user_topics = compute_user_topic_profiles(posts_topics, user_col="id", probas_col="topic_probas", topic_threshold=0.05, topic_model=topic_model)
    nodes_final = attach_topics_to_nodes(lp_nodes, user_topics, user_col="id")

    print("\n[6] NNIF Calculation (Information Flow Entropy)...")
    seqs = build_user_token_sequences(timelines, tokenizer=english_sns_tokenizer, max_tokens_per_user=3000)
    neighbors = build_neighbors_from_edges(edges_info)
    nnif_df = compute_nnif_for_edges(edges_info, seqs, neighbors)

    print("\n[7] Calculating tSPIN (Eq 3.7 - 3.10)...")
    # ★修正: BERTopicが実際に見つけたトピック数を自動で取得（ノイズの -1 を除外）
    actual_num_topics = len(topic_model.get_topic_info()) - 1 
    print(f"[tSPIN] Discovered {actual_num_topics} valid topics for matrix calculation.")
    
    tspin_res = compute_probabilistic_tspin(
        nnif_df, nodes_final, user_topics,
        alpha=0.3, beta=0.7, num_topics=actual_num_topics  
    )

    print("\n[8] Exporting Robust Results for Peer Review...")
    export_all_results_en(
        tspin_result=tspin_res,
        nnif_df=nnif_df,
        nodes_df=nodes_final,
        edges_df=edges_info,
        neg_users=list(negative_user_ids), # 修正: negative_usersを正確に渡す
        topic_model=topic_model
    )
    
    print("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()