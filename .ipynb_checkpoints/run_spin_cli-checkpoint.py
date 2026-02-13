#!/usr/bin/env python3
"""
コンソールから SPIN snowball を実行するための簡易スクリプト。
byobu の中で実行して、ログと結果を眺める想定。
"""

from datetime import datetime
import yaml
from elasticsearch import Elasticsearch

from define_seed_snowball import run_spin_snowball_for_bluesky  # :contentReference[oaicite:2]{index=2}
from research_matae import run_spin_with_posts_df_a

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Tuple

import pandas as pd
import networkx as nx

import pandas as pd
import sys
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import os
from pyvis.network import Network
import streamlit.components.v1 as components
import yaml
from yaml.loader import SafeLoader

import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation as LDA
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from collections import Counter, defaultdict
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import matplotlib.colors as mcolors
from pyvis.network import Network
from transformers import pipeline
import seaborn as sns
import tempfile
from transformers import AutoTokenizer
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import re
import math
from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Tuple
import pandas as pd

from datetime import datetime, timezone
from snowball_spin import snowball_spin, make_bluesky_publication_fetcher, make_bluesky_reposter_getter
from define_seed_snowball import (
    select_seed_accounts_from_df,
    run_spin_snowball_for_bluesky,
)
from typing import Sequence, Optional

from typing import Optional, Any, List, Dict

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

from topic_pipeline_tokenizer_updated import (
    run_bertopic_on_posts,
    compute_user_topic_profiles,
    attach_topics_to_nodes,
    render_nodes_table_with_community_and_topics, 
)

from research_matae_tokenizer_updated import (
    build_spin_network_from_sampled_edges,
    propagate_multilabel_from_seeds_threshold,
    attach_negative_flag_to_nodes,
    _extract_post_cid,
    _terms_agg_counts_by_field,
    plot_comm_topic_heatmap,
    plot_nnif_log_distribution,
    merge_networks_by_common_users,
)

from Calculate_NNIF_tokenizer_updated import (
    build_information_flow_edges,
    build_user_timelines,
    build_neighbors_from_edges,
    build_user_token_sequences,
    compute_nnif_for_edges,
)

from negative_judge import load_jp_negative_lexicon

from Calculate_tSPIN import run_tspin
from Calculate_tSPIN import compute_probabilistic_tspin

from research_output import export_all_results

import requests, json

ES_ADDR = "http://cicero.csis.oita-u.ac.jp:9200"  # research_matae と揃えた設定 :contentReference[oaicite:3]{index=3}


def main() -> None:
    # --- config.yaml の読み込み（必要なら） ---
    # ここでは Bluesky アカウント情報などを使わないので、
    # 「とりあえず読み込んでおく」程度です。
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    load_jp_negative_lexicon()

    # --- Elasticsearch クライアント作成 ---
    es = Elasticsearch(
        hosts=[ES_ADDR],
        request_timeout=1200,
        retry_on_timeout=True,
        max_retries=5,
    )

    query_str_a="ロシア"
    query_str_b="ウクライナ"

    post_index="postindex-*"
    repostindex="repostindex-*"

    # --- 解析対象期間（必要に応じて変更してください） ---
    # 例: 2025-11-01 〜 2025-11-03 の投稿を対象
    start_day = datetime(2025, 11, 20)
    end_day = datetime(2026, 1, 17)

    query_body_a = {
        "_source": [
            "did",
            "commit.rkey",
            "commit.cid",            # ← これを明示
            "cid",                   # ← 念のため root 側にもフォールバック
            "commit.record.text",
            "commit.record.createdAt",
            "event.original",
            "commit.record.reply"    # ← フィールドとしては残してOK（使いたければ）
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "wildcard": {
                            # ここがポイント: query_str_a を使った部分一致
                            "commit.record.text": f"*{query_str_a}*"
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "commit.record.createdAt": {
                                "gte": start_day.isoformat(),
                                "lte": end_day.isoformat()
                            }
                        }
                    }
                ]
                # ★ リプライ除外はやめるので must_not は削除
                # "must_not": [
                #     { "exists": { "field": "commit.record.reply" } }
                # ]
            }
        }
    }

    query_body_b = {
        "_source": [
            "did",
            "commit.rkey",
            "commit.cid",            # ← これを明示
            "cid",                   # ← 念のため root 側にもフォールバック
            "commit.record.text",
            "commit.record.createdAt",
            "event.original",
            "commit.record.reply"    # ← フィールドとしては残してOK（使いたければ）
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "wildcard": {
                            # ここがポイント: query_str_a を使った部分一致
                            "commit.record.text": f"*{query_str_b}*"
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "commit.record.createdAt": {
                                "gte": start_day.isoformat(),
                                "lte": end_day.isoformat()
                            }
                        }
                    }
                ]
                # ★ リプライ除外はやめるので must_not は削除
                # "must_not": [
                #     { "exists": { "field": "commit.record.reply" } }
                # ]
            }
        }
    }

    post_df_a = get_post_texts(es=es, query_body=query_body_a, size=1000)
    post_df_b = get_post_texts(es=es, query_body=query_body_b, size=1000)
    # post_df_a は既に存在する DataFrame
    post_df_a["label"] = "A"
    post_df_b["label"] = "B"
    # st.dataframe(post_df_a[["text", "created_at", "like_count", "repost_count", "url", "uri", "rkey", "did", "cid"]], use_container_width=True)
    st.dataframe(post_df_a[["text", "created_at", "like_count", "repost_count", "url"]], use_container_width=True, column_config={"url": st.column_config.LinkColumn("url")})

    # --- SPIN snowball 実行 ---
    # logger を渡さなければ、define_seed_snowball 側で print が使われます。:contentReference[oaicite:4]{index=4}
    print("[5/12] SPIN snowball（run_spin_with_posts_df_a）実行中...")
    result = run_spin_with_posts_df_a(
        es,
        posts_df_a=post_df_a,
        start_dt=start_day,
        end_dt=end_day,
    )

    result_b = run_spin_with_posts_df_a(
        es,
        posts_df_a=post_df_b,
        start_dt=start_day,
        end_dt=end_day,
    )

    # --- 結果のサマリを標準出力に表示 ---
    print("\n==================== SPIN RESULT SUMMARY ====================")
    print(f"期間: {start_day.isoformat()} 〜 {end_day.isoformat()}")
    print("\n[Seed accounts]")
    for did in result["seed_accounts"]:
        print("  -", did)

    print(f"\nサンプルされたユーザ数: {len(result['common_list'])}")
    print(f"サンプルされたエッジ数: {len(result['sampled_edges'])}")

    # 必要なら、エッジを先頭だけ表示
    edges = result["sampled_edges"]
    print("\n[Sampled edges (first 20)]")
    for src, dst in edges[:20]:
        print(f"  {src} -> {dst}")

    print("\n=============================================================")

    # --- 1) DataFrame で生のエッジを確認 ---
    edges_df_spin = pd.DataFrame(result["sampled_edges"], columns=["src", "dst"])
    print("#### SPIN 完了")

    print("[7/12] SPINネットワーク構築（build_spin_network_from_sampled_edges）...")
    # --- 2) SPIN ネットワークを構成 ---
    nodes_df_spin, edges_df_spin_full = build_spin_network_from_sampled_edges(
        seed_accounts=result["seed_accounts"],
        sampled_edges=result["sampled_edges"],
    )

    nodes_df_spin_b, edges_df_spin_full_b = build_spin_network_from_sampled_edges(
        seed_accounts=result_b["seed_accounts"],
        sampled_edges=result_b["sampled_edges"],
    )
    print("    → Network build 完了")

    merged_nodes_df, merged_edges_df, common_users = merge_networks_by_common_users(
        nodes_a=nodes_df_spin,
        edges_a=edges_df_spin_full,
        nodes_b=nodes_df_spin_b,
        edges_b=edges_df_spin_full_b,
    )

#     # --- 3) Plotly + Streamlit で可視化 ---
#     plot_network_from_dfs(
#         nodes_df_spin,
#         edges_df_spin_full,
#         node_id_col="id",
#         node_label_col="label",  # DID（seed は★付き）
#         url_col="url",           # クリックでプロフィールを開く
#         edge_src_col="src",
#         edge_dst_col="dst",
#         edge_weight_col="weight",
#         layout="spring",         # spring layout でレイアウト
#         plot_height=700,
#         use_streamlit=True,      # ★ Streamlit に埋め込み
#     )
# else:
#     st.warning("SPIN の結果としてエッジが 0 件でした。期間やクエリ、repost_count などを確認してください。")
    all_posts_df = pd.concat(
        [post_df_a.assign(camp="A"), post_df_b.assign(camp="B")],
        ignore_index=True,
    )

    merged_user_ids = set(merged_nodes_df["id"].astype(str))

    merged_posts_df = (
        all_posts_df
        .loc[all_posts_df["did"].astype(str).isin(merged_user_ids)]
        .reset_index(drop=True)
    )

    # SPIN ネットワークにマルチラベル伝搬を適用
    # --- 3) ラベル伝搬 ---
    print("[8/12] ラベル伝搬（propagate_multilabel_from_seeds_threshold）...")
    lp_nodes_df = propagate_multilabel_from_seeds_threshold(
        merged_nodes_df,
        merged_edges_df,
        seed_ids=result["seed_accounts"],
        threshold=0.3,
        max_iter=50,
        use_weight=True,
        random_state=42,
    )
    print("    → Label propagation 完了")
    
    # --- 4) BERTopic + ユーザトピック付与 ---
    print("[9/12] BERTopic によるトピック抽出中...")
    posts_with_topics, topic_model = run_bertopic_on_posts(
        merged_posts_df,
        text_col="text",
        model_save_path=None,
    )
    print("    → BERTopic 完了")

    posts_with_topics["id"] = posts_with_topics["did"].astype(str)
    
    print("[10/12] ユーザトピックの付与...")
    user_topic_df = compute_user_topic_profiles(
        posts_with_topics,
        user_col="id",          # post_df_a の列名に合わせる
        probas_col="topic_probas",
    )
    
    nodes_with_topics = attach_topics_to_nodes(
        lp_nodes_df,
        user_topic_df,
        user_col="id",
    )
    print("    → トピック付与OK")
    
    # --- 5) ネガティブ判定（ここで is_negative を付与） ---
    print("[11/12] ネガティブ判定を実行中...")
    nodes_with_topics_and_neg = attach_negative_flag_to_nodes(
        es,
        nodes_with_topics,
        id_col="id",            # nodes_df のユーザID列名に合わせて
        start_dt=start_day,
        end_dt=end_day,
        max_posts=20,           # 「10〜20件」の上限
        post_index="postindex-*",  # "postindex-*" など
    )
    print("    → ネガティブ判定完了")
    
    print("[12/12] NNIF 計算を実行中...")
    neg_users = (
        nodes_with_topics_and_neg
        .loc[nodes_with_topics_and_neg["is_negative"], "id"]
        .astype(str)
        .tolist()
    )
    print(f"[NNIF][CALL] ネガティブユーザ数: {len(neg_users)}")
    
    # --- 6) ネガティブユーザだけで NNIF ---
    edges_info = build_information_flow_edges(
        es,
        post_index="postindex-*",
        repost_index="repostindex-*",
        start_dt=start_day,
        end_dt=end_day,
        check_content_negativity=True,
    )
    print(f"[NNIF][CALL] 負の情報フローエッジ数: {len(edges_info)}")
    
    # ★修正箇所: relevant_users をここで定義します
    # エッジ（情報の流れ）が存在するユーザーの集合です
    relevant_users = set(edges_info["src"]).union(set(edges_info["dst"]))
    print(f"[NNIF][CALL] 計算対象ユーザ数: {len(relevant_users)}")
    
    user_timelines = build_user_timelines(
        es,
        post_index="postindex-*",
        start_dt=start_day,
        end_dt=end_day,
        allowed_users=relevant_users,
        min_posts=1,
    )
    print(f"[NNIF][CALL] タイムライン構築ユーザ数: {len(user_timelines)}")
    
    user_token_seqs = build_user_token_sequences(
        user_timelines,
        max_tokens_per_user=3000,
    )
    print(f"[NNIF][CALL] トークン列を持つユーザ数: {len(user_token_seqs)}")
    
    neighbors = build_neighbors_from_edges(edges_info)
    print(f"[NNIF][CALL] 近傍情報を持つユーザ数: {len(neighbors)}")
    
    nnif_df = compute_nnif_for_edges(
        edges_info,
        user_token_seqs,
        neighbors,
        max_match_len=20,
    )

    print(f"[NNIF][CALL] NNIF 計算結果ペア数: {len(nnif_df)}")

    print("[NNIF][CALL] NNIF 計算フェーズ完了")

    print("[13/12] tSPIN 計算を実行中...")

    tspin_result = compute_probabilistic_tspin(
        nnif_df=nnif_df,
        nodes_df=nodes_with_topics, 
        user_topic_df=user_topic_df,
        alpha=0.5,
        beta=0.5
    )
    
    print("=== tSPIN RESULT ===")
    print(f"tSPIN: {tspin_result['tspin']:.4f}")
    print(f"Intra-neg ratio: {tspin_result['intra_neg_ratio']:.4f}")
    print(f"Inter-neg ratio: {tspin_result['inter_neg_ratio']:.4f}")
    
    print("\n[Top NNIF pairs]")
    print(tspin_result["top_pairs"].head(10))

    export_all_results(
        tspin_result=tspin_result,
        nnif_df=nnif_df,
        nodes_df=nodes_with_topics_and_neg,
        edges_df=edges_info,
        neg_users=list(relevant_users),
        topic_model=topic_model
    )

    print("=== SPIN CLI FINISHED ===")

    WEB_HOOK_URL = "https://saussure.csis.oita-u.ac.jp:5001/webapi/entry.cgi?api=SYNO.Chat.External&method=incoming&version=2&token=%22GELlhksoWRMB9HXplYLIV8NdJYjTrCgCrcJpW7lMF72jSsTgv5NDXj5vcjiJbm1A%22"
    CONTENT = 'Notifycation From Python. spin-cli ended.'
    requests.post(WEB_HOOK_URL, verify=False, data = 'payload=' + json.dumps({
        'text': CONTENT
    }))

def get_post_texts(es, query_body, index: str = "postindex-*", size: int = 1000) -> pd.DataFrame:
    """
    投稿取得 + CID を必ず列に入れ、CID基準で like/repost を付与して返す。

    追加仕様:
    - 文字数が 10 文字以下の text は DataFrame に入れない
    - text が重複する投稿は入れない（先に見つけたものだけ採用）
    - 条件を満たす投稿が size 件に達するまで、同じ query_body で ES からページングして取得
      （ヒットが尽きた場合は、その時点で打ち切り）
    """

    rows = []
    seen_texts = set()

    # ページング用
    from_ = 0
    page_size = size  # 必要なら 500 とか 1000 に固定してもOK

    while len(rows) < size:
        # 元の query_body を壊さないようにコピーして from/size を付ける
        local_body = dict(query_body)
        local_body["from"] = from_
        local_body["size"] = page_size

        res = es.search(
            index=index,
            body=local_body,  # ← body に query + from + size をまとめる
            track_total_hits=True,
            ignore_unavailable=True,
            allow_no_indices=True,
            expand_wildcards="open",
            request_cache=True,
        )

        hits = (res.get("hits") or {}).get("hits") or []
        if not hits:
            # これ以上ヒットがないので終了
            break

        for h in hits:
            src    = h.get("_source") or {}
            commit = src.get("commit") or {}
            record = commit.get("record") or {}

            text = record.get("text") or ""
            # ① 文字数フィルタ
            if len(text) <= 10:
                continue

            # ② 重複 text フィルタ
            if text in seen_texts:
                continue

            did  = src.get("did")
            rkey = commit.get("rkey")
            cid  = _extract_post_cid(src)  # ★ 既存の CID 抽出関数

            uri     = f"at://{did}/app.bsky.feed.post/{rkey}" if (did and rkey) else None
            web_url = f"https://bsky.app/profile/{did}/post/{rkey}" if (did and rkey) else None

            rows.append({
                "did": did,
                "rkey": rkey,
                "cid": cid,
                "text": text,
                "created_at": record.get("createdAt"),
                "uri": uri,
                "url": web_url,
            })
            seen_texts.add(text)

            # 希望件数に達したら終了
            if len(rows) >= size:
                break

        # 次ページへ
        from_ += page_size

    # DataFrame 化（列は固定で作る）
    df = pd.DataFrame(rows, columns=["did", "rkey", "cid", "text", "created_at", "uri", "url"])

    # like / repost の集計
    if not df.empty:
        cids = df["cid"].dropna().astype(str).tolist()
        
        like_counts = _terms_agg_counts_by_field(
            es, index="likeindex-*", ids=cids, field="commit.record.subject.cid.keyword"
        )
        repost_counts = _terms_agg_counts_by_field(
            es, index="repostindex-*", ids=cids, field="commit.record.subject.cid.keyword"
        )

        df["like_count"] = df["cid"].astype(str).map(like_counts).fillna(0).astype(int)
        df["repost_count"] = df["cid"].astype(str).map(repost_counts).fillna(0).astype(int)
    else:
        df["like_count"] = pd.Series(dtype="int64")
        df["repost_count"] = pd.Series(dtype="int64")

    return df

if __name__ == "__main__":
    main()