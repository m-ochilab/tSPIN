# 追加：最上部の import 付近
from __future__ import annotations
import requests
import time, math
from urllib.parse import urlencode

class BlueskyClient:
    def __init__(self, access_jwt: str, qps: float = 1.5, burst: int = 4, timeout=30):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {access_jwt}"})
        self.timeout = timeout
        self.rate = qps; self.tokens = burst; self.cap = burst; self.ts = time.monotonic()

    def _throttle(self):
        now = time.monotonic()
        self.tokens = min(self.cap, self.tokens + (now - self.ts) * self.rate)
        self.ts = now
        if self.tokens < 1:
            time.sleep((1 - self.tokens) / self.rate)
            self.ts = time.monotonic()
            self.tokens = min(self.cap, self.tokens + (now - self.ts) * self.rate)
        self.tokens -= 1

    def _get(self, path, params=None, retry=5):
        url = f"https://bsky.social{path}"
        backoff = 1.0
        for i in range(retry+1):
            self._throttle()
            r = self.s.get(url if not params else f"{url}?{urlencode(params, doseq=True)}", timeout=self.timeout)
            if 200 <= r.status_code < 300:
                return r.json()
            if r.status_code in (429,500,502,503,504) and i < retry:
                time.sleep(float(r.headers.get("Retry-After", backoff)))
                backoff = min(backoff*2, 20.0)
                continue
            r.raise_for_status()

    # 必要な API ラッパだけ（検索 / 投稿詳細 / リポスター）
    def search_posts(self, q, cursor=None, limit_total=2000):
        got = 0
        while True:
            js = self._get("/xrpc/app.bsky.feed.searchPosts", {"q": q, "limit": 100, **({"cursor": cursor} if cursor else {})})
            posts = js.get("posts", []) or []
            for p in posts:
                yield p; got += 1
                if got >= limit_total: return
            cursor = js.get("cursor")
            if not cursor or not posts: return

    def get_posts(self, uris):
        out = {}
        for i in range(0, len(uris), 25):
            qs = [("uris", u) for u in uris[i:i+25]]
            js = self._get("/xrpc/app.bsky.feed.getPosts", qs)
            for pv in js.get("posts", []) or []:
                out[pv["uri"]] = {
                    "cid": pv.get("cid"),
                    "author": (pv.get("author") or {}).get("did"),
                    "likeCount": pv.get("likeCount", 0) or 0,
                    "repostCount": pv.get("repostCount", 0) or 0,
                }
        return out

    def get_reposted_by(self, uri, cid=None, cursor=None, limit_total=500):
        got = 0
        while True:
            params = {"uri": uri, "limit": 100, **({"cid": cid} if cid else {}), **({"cursor": cursor} if cursor else {})}
            js = self._get("/xrpc/app.bsky.feed.getRepostedBy", params)
            for it in js.get("repostedBy", []) or []:
                yield it.get("did"); got += 1
                if got >= limit_total: return
            cursor = js.get("cursor")
            if not cursor: return
                
# -*- coding: utf-8 -*-
"""
analyze_fundamental.py

A small, importable foundation layer that provides:
- A pluggable "feature registry" to compose analyses as small functions
- Thin wrapper features that call your existing implementation functions
  (e.g., get_post_time_series, get_post_user_ranking, etc.) via dependency injection
- Simple utilities for date handling

Usage (in your existing app file):
----------------------------------
from analyze_fundamental import (
    FeatureSpec, run_features, register_feature, set_impl, Impl
)

# After your original function definitions exist (or at app startup):
set_impl(Impl(
    get_post_time_series=get_post_time_series,
    get_post_user_ranking=get_post_user_ranking,
    cluster_post_texts=cluster_post_texts,
    get_word_frequency_ranking=get_word_frequency_ranking,
    create_recursive_interaction_user_network_parallel=create_recursive_interaction_user_network_parallel,
    merge_graphs_on_common_nodes_multi=merge_graphs_on_common_nodes_multi,
    fetch_display_info_parallel=fetch_display_info_parallel,   # optional
    get_profile_avatar_url=get_profile_avatar_url,             # optional
))

# Then you can compose and run features
specs = [
    FeatureSpec("timeseries",   {"query": "keyword", "start": "2025-10-01", "end": "2025-10-12"}),
    FeatureSpec("user_ranking", {"query": "keyword", "start": "2025-10-01", "end": "2025-10-12", "token": None}),
]
results = run_features(es, specs)  # `es` is your Elasticsearch client
"""
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

from topic_pipeline import (
    run_bertopic_on_posts,
    compute_user_topic_profiles,
    attach_topics_to_nodes,
    render_nodes_table_with_community_and_topics,
)

from negative_judge import is_negative_user_from_texts

es_addr="http://cicero.csis.oita-u.ac.jp:9200" #Elasticsearchアドレス
es_request_timeout = 1200 #タイムアウトする時間
es_timeout = "10m" #サーバ処理時間の上限

try:
    # for typing only; not required at runtime
    from elasticsearch import Elasticsearch  # type: ignore
except Exception:  # pragma: no cover
    Elasticsearch = object  # fallback for type hints

def _log_stage(container, label):
    """ステージ開始用のロガー。終了時刻との組み合わせで使う。"""
    ts = datetime.now().strftime("%H:%M:%S")
    container.write(f"[{ts}] {label}")

def analyze_spin(config):
    
    stage_log = st.container()
    
    es = Elasticsearch(
        hosts=[es_addr],
        request_timeout=es_request_timeout,     # クライアント側のタイムアウト（秒）
        retry_on_timeout=True,                  # タイムアウト時の再試行を有効に
        max_retries=5,                          # 再試行回数
    )

    st.sidebar.header("入力パラメータ")
    option = st.sidebar.radio(
        "クエリの種類を選択してください。：",
        ["二つ"],
        index=0
    )

    # ネガティブワード辞書（とりあえず例）
    NEGATIVE_KEYWORDS: List[str] = [
        "死ね", "殺す", "バカ", "クソ", "最悪", "許さない",
        "大嫌い", "差別", "侮辱", "攻撃", "暴力", "ヘイト",
        # 必要に応じて追加
    ]
    
    query_str_a = st.sidebar.text_input("クエリA（例: Trump）", "Trump") #文字入力欄を作る
    query_str_b = st.sidebar.text_input("クエリB（例: UNO）", "UNO") #クエリB

    today = dt.date.today() #今日の日付
    yesterday = today + relativedelta(days=-1) #昨日
    twodaysago = today + relativedelta(days=-2) #一昨日
    start_date = st.sidebar.date_input("開始日", value=twodaysago) #日付入力欄を作る
    end_date = st.sidebar.date_input("終了日", value=yesterday)

    freq_choice = st.sidebar.selectbox(
        "時系列分布集計間隔",
        ["10分","30分","1時間","6時間","1日","1週間"],  # 自由に増やしてください
        index=0,
    )

    time_freq = resolve_freq_input(freq_choice)
    
    # start_date_togisen = st.sidebar.date_input("都議選開始日", value=twodaysago) #日付入力欄を作る
    # end_date_togisen = st.sidebar.date_input("都議選終了日", value=yesterday)
    language_option = st.sidebar.selectbox("対象言語", ["すべて", "英語", "日本語"])

    # 対象言語が「英語」なら "en", 「日本語」なら "ja", 「すべて」の場合は None
    if language_option == "英語":
        language_filter = "en"
    elif language_option == "日本語":
        language_filter = "ja"
    else:
        language_filter = None
    
    # 日付を ISO8601 形式に変換（時間は適宜補完）
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str   = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    start_yyyymm = start_date.strftime("%Y%m")
    end_yyyymm   = end_date.strftime("%Y%m")

    bluesky_handle =  config["bsky_user"]["handle"]
    bluesky_password = config["bsky_user"]["password"]

    st.write("クエリAの投稿内容を表示中...")
    
    # リプライ除外 + 必要フィールドだけ返す + rangeはfilterへ
    query_body_a = {
      "_source": [
        "did",
        "commit.rkey",
        "commit.cid",            # ← これを明示
        "cid",                   # ← 念のため root 側にもフォールバック
        "commit.record.text",
        "commit.record.createdAt",
        "event.original",        # ← 万一の最終手段（JSON文字列に commit.cid がいる）
        "commit.record.reply"    # 除外チェック用
      ],
      "query": {
        "bool": {
            "must": [
              { "match_phrase": { "commit.record.text": query_str_a } }
            ],
          "filter": [
            { "range": { "commit.record.createdAt": {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }}}
          ],
          "must_not": [
            { "exists": { "field": "commit.record.reply" } }
          ]
        }
      }
    }

    st.write("クエリBの投稿内容を取得中...")
    query_body_b = {
      "_source": [
        "did",
        "commit.rkey",
        "commit.cid",            # ← これを明示
        "cid",                   # ← 念のため root 側にもフォールバック
        "commit.record.text",
        "commit.record.createdAt",
        "event.original",        # ← 万一の最終手段（JSON文字列に commit.cid がいる）
        "commit.record.reply"    # 除外チェック用
      ],
      "query": {
        "bool": {
          "must": [
            { "match": { "commit.record.text" : {
                "query":query_str_b ,
                "operator" : "and"
                } }
            }
          ],
          "filter": [
            { "range": { "commit.record.createdAt": {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }}}
          ],
          "must_not": [
            { "exists": { "field": "commit.record.reply" } }
          ]
        }
      }
    }

    post_df_a = get_post_texts(es=es, query_body=query_body_a, size=1000)
    # post_df_a は既に存在する DataFrame
    post_df_a["label"] = "A"
    # st.dataframe(post_df_a[["text", "created_at", "like_count", "repost_count", "url", "uri", "rkey", "did", "cid"]], use_container_width=True)
    st.dataframe(post_df_a[["text", "created_at", "like_count", "repost_count", "url"]], use_container_width=True, column_config={"url": st.column_config.LinkColumn("url")})

    post_ts_df_a = get_post_time_series(es, query_str_a, start_date_str, end_date_str, lang_filter=language_filter)

    post_df_b = get_post_texts(es=es, query_body=query_body_b, size=1000)
    # post_df_a は既に存在する DataFrame
    post_df_b["label"] = "B"
    # st.dataframe(post_df_b[["text", "created_at", "like_count", "repost_count", "url"]], use_container_width=True)
    st.dataframe(post_df_b[["text", "created_at", "like_count", "repost_count", "url"]], use_container_width=True, column_config={"url": st.column_config.LinkColumn("url")})

    # post_ts_df_b = get_post_time_series(es, query_str_b, start_date_str, end_date_str, lang_filter=language_filter)

    # combined_post_df = pd.concat([post_df_a, post_df_b], ignore_index=True)

    plot_timeseries_from_posts(
        post_df_a,
        "created_at",
        None,
        time_freq,
        "Asia/Tokyo",
        "投稿数の時系列-クエリA",
        True,
    )
    
    plot_timeseries_from_posts(
        post_df_b,
        "created_at",
        None,
        time_freq,
        "Asia/Tokyo",
        "投稿数の時系列-クエリB",
        True,
    )

        # 重ね＋差分＋比率を表示
    _merged_ts, _, _ = plot_timeseries_compare_from_posts(
        post_df_a, post_df_b,
        created_at_col="created_at", freq=time_freq, tz="Asia/Tokyo",
        title="投稿数の時系列（A vs B：重ね＋差分＋比率）",
    )

    # tok = create_tokenizer_latest(lang="auto", allow_download=False)
    
    # # すでに post_df_a に 'cid' と 'repost_count' が入っている前提
    # nodes_df_a, edges_df_a = build_repost_network_snowball(
    #     es,
    #     post_df=post_df_a,              # ← ここから起点CIDを自動選択（repost_count最大）
    #     post_index="postindex-*",
    #     repost_index="repostindex-*",   # 重い場合は "repostindex-2025.09*,repostindex-2025.10*" など期間限定に
    #     depth=3,                        # 何ホップ進めるか
    #     top_reposters_per_post=10,      # 起点/各ポストから拾うリポスター数
    #     per_user_post_limit=20,         # 各ユーザーの直近投稿を何件見るか
    #     next_post_min_reposts=2,        # 次に辿るポストの最小リポスト数
    #     max_users=500                  # セーフティ上限
    # )
    
    # nodes_df_b, edges_df_b = build_repost_network_snowball(
    #     es,
    #     post_df=post_df_b,              # ← ここから起点CIDを自動選択（repost_count最大）
    #     post_index="postindex-*",
    #     repost_index="repostindex-*",   # 重い場合は "repostindex-2025.09*,repostindex-2025.10*" など期間限定に
    #     depth=3,                        # 何ホップ進めるか
    #     top_reposters_per_post=10,      # 起点/各ポストから拾うリポスター数
    #     per_user_post_limit=20,         # 各ユーザーの直近投稿を何件見るか
    #     next_post_min_reposts=2,        # 次に辿るポストの最小リポスト数
    #     max_users=500                  # セーフティ上限
    # )
    
    # # 確認（Streamlit）
    # st.dataframe(nodes_df_a.head(50))
    # st.dataframe(edges_df_a.head(100))
    
    # st.markdown("### リポストネットワークークエリA")
    # plot_network_from_dfs(
    #     nodes_df_a, 
    #     edges_df_a,
    #     node_size_col="repost_count",
    #     node_label_col="label",
    #     node_type_col="type",
    #     layout="spring",
    #     use_streamlit=True,
    #     show_text=False
    # )
    # st.markdown("### リポストネットワークークエリB")
    # plot_network_from_dfs(
    #     nodes_df_b, 
    #     edges_df_b,
    #     node_size_col="repost_count",
    #     node_label_col="label",
    #     node_type_col="type",
    #     layout="spring",
    #     use_streamlit=True,
    #     show_text=False
    # )
    
    # st.markdown("### 結合ネットワーク A-B")
    # # 2) ネットワークを「共通ユーザ（橋渡し）」で結合
    # merged_nodes, merged_edges, bridge_ids = merge_networks_by_common_users(
    #     nodes_df_a, edges_df_a,
    #     nodes_df_b, edges_df_b,
    #     node_id_col="id",          # デフォルトの列名のままなら省略可
    #     edge_src_col="src",
    #     edge_dst_col="dst",
    #     edge_weight_col="weight",  # 無ければ自動で 1.0 が入ります
    # )
    
    # print(f"橋渡しノード数: {len(bridge_ids)}")
    
    # # 3) 可視化（Streamlit）
    # plot_merged_network_with_highlights(
    #     merged_nodes,
    #     merged_edges,
    #     node_id_col="id",
    #     node_label_col="label",   # ラベルに DID を出しているなら "label" のままでOK
    #     url_col="url",            # 無ければ DID から自動生成されます
    #     edge_src_col="src",
    #     edge_dst_col="dst",
    #     edge_weight_col="weight",
    #     layout="spring",          # "spring" | "kamada_kawai" | "fr" | "circular"
    #     seed=42,
    #     node_size_col=None,       # 指定しなければ次数でサイズ化
    #     node_size_range=(8, 20),
    #     use_streamlit=True,
    #     title="Merged network (A ↔ B)",
    # )

    t4 = time.perf_counter()
    _log_stage(stage_log, "④ SPIN (snowball_spin) 実行中...")
    elapsed_spin = time.perf_counter() - t4
    result = run_spin_with_posts_df_a(
        es,
        posts_df_a=post_df_a,
        start_dt=start_date,
        end_dt=end_date,
    )
    _log_stage(
        stage_log,
        f"④ SPIN 完了: ユーザ {len(result['common_list'])} 件, "
        f"エッジ {len(result['sampled_edges'])} 本 （{elapsed_spin:.1f} 秒）",
    )

    # ================================
    # ★ SPIN の進捗・ログの可視化部分
    # ================================
    st.markdown("### SPIN（snowball sampling）進捗")

    # params から sampling_log を取り出す
    params = result.get("params", {})
    sampling_log = params.get("sampling_log", [])

    if sampling_log:
        total_steps = len(sampling_log)

        # 進捗バーとテキスト枠を用意
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # sampling_log 全体を一周して「どんなステップがあったか」を見せる
        # ※ SPIN の実行自体はすでに終わっているので「再生バー」のようなイメージ
        for i, step in enumerate(sampling_log):
            level = step.get("level")
            from_user = step.get("from_user")
            pub_id = step.get("pub_id")
            n_rep = step.get("n_reposters")
            chosen = step.get("chosen_reposter")

            # 0〜1で進捗を更新
            progress = (i + 1) / total_steps
            progress_bar.progress(progress)

            # 今どのステップを「再生中」かを表示
            status_text.text(
                f"Step {i+1}/{total_steps} | "
                f"Level {level} | from={from_user} | "
                f"pub={pub_id} | reposters={n_rep} | chosen={chosen}"
            )

        # レベルごとのサマリも表示
        st.markdown("#### レベル別ステップ数のサマリ")

        log_df = pd.DataFrame(sampling_log)
        level_summary = (
            log_df
            .groupby("level")
            .size()
            .reset_index(name="n_steps")
            .sort_values("level")
        )
        st.dataframe(level_summary, use_container_width=True)

    else:
        st.info(
            "sampling_log が空でした。"
            "snowball_spin() が debug=True で呼ばれているか確認してください。"
        )

    # 5) SPIN 結果を画面に表示
    st.markdown("### SPIN（snowball sampling）結果")

    # seed アカウント（DID）
    st.write("**Seed accounts (DID)**")
    st.write(result["seed_accounts"])

    # サンプルされたユーザ数・エッジ数
    st.write(f"サンプルされたユーザ数: {len(result['common_list'])}")
    st.write(f"サンプルされたエッジ数: {len(result['sampled_edges'])}")

    # エッジ一覧を DataFrame で軽く確認（上位100件だけ）
    if result["sampled_edges"]:
        # --- 1) DataFrame で生のエッジを確認 ---
        edges_df_spin = pd.DataFrame(result["sampled_edges"], columns=["src", "dst"])
        st.markdown("#### SPIN で得られたエッジ（先頭100件）")
        st.dataframe(edges_df_spin.head(100), use_container_width=True)

        # --- 2) SPIN ネットワークを構成 ---
        nodes_df_spin, edges_df_spin_full = build_spin_network_from_sampled_edges(
            seed_accounts=result["seed_accounts"],
            sampled_edges=result["sampled_edges"],
        )

        st.markdown("#### SPIN でサンプリングされたリポストネットワーク")

        # --- 3) Plotly + Streamlit で可視化 ---
        plot_network_from_dfs(
            nodes_df_spin,
            edges_df_spin_full,
            node_id_col="id",
            node_label_col="label",  # DID（seed は★付き）
            url_col="url",           # クリックでプロフィールを開く
            edge_src_col="src",
            edge_dst_col="dst",
            edge_weight_col="weight",
            layout="spring",         # spring layout でレイアウト
            plot_height=700,
            use_streamlit=True,      # ★ Streamlit に埋め込み
        )
    else:
        st.warning("SPIN の結果としてエッジが 0 件でした。期間やクエリ、repost_count などを確認してください。")

    # SPIN ネットワークにマルチラベル伝搬を適用
    lp_nodes_df = propagate_multilabel_from_seeds_threshold(
        nodes_df_spin,
        edges_df_spin_full,
        seed_ids=result["seed_accounts"],  # さっき決めた seed_account のリスト
        threshold=0.3,                     
        max_iter=50,
        use_weight=True,
        random_state=42,
    )
    
    posts_with_topics, topic_model = run_bertopic_on_posts(
        post_df_a,
        text_col="text",
        model_save_path=None,
    )

    # 2) ユーザごとのトピックプロファイル
    user_topic_df = compute_user_topic_profiles(
        posts_with_topics,
        user_col=user_col,
        probas_col="topic_probas",
        topic_threshold=topic_threshold,
        topic_model=topic_model,
    )

    # 3) ノードDFに join
    nodes_with_topics = attach_topics_to_nodes(
        nodes_df_spin,
        user_topic_df,
        user_col=user_col,
    )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def to_iso_utc_day_start(d) -> str:
    """date/datetime/str -> 'YYYY-MM-DDT00:00:00Z' (string passthrough)"""
    if isinstance(d, str):
        return d
    return f"{pd.to_datetime(d).strftime('%Y-%m-%d')}T00:00:00Z"


def to_iso_utc_day_end(d) -> str:
    """date/datetime/str -> 'YYYY-MM-DDT23:59:59Z' (string passthrough)"""
    if isinstance(d, str):
        return d
    return f"{pd.to_datetime(d).strftime('%Y-%m-%d')}T23:59:59Z"


# ------------------------------------------------------------------
# Implementation injection (dependency injection)
# ------------------------------------------------------------------

@dataclass
class Impl:
    """Holds references to your concrete implementations.
    All required unless marked Optional.
    """
    get_post_time_series: Callable[..., pd.DataFrame]
    get_post_user_ranking: Callable[..., pd.DataFrame]
    cluster_post_texts: Callable[..., Optional[pd.DataFrame]]
    get_word_frequency_ranking: Callable[..., List[Tuple[str, int]]]
    create_recursive_interaction_user_network_parallel: Callable[..., Tuple[nx.MultiDiGraph, pd.DataFrame]]
    merge_graphs_on_common_nodes_multi: Callable[[List[nx.MultiDiGraph]], nx.MultiDiGraph]

    # Optional enhancements
    fetch_display_info_parallel: Optional[Callable[..., Dict[str, Dict[str, str]]]] = None
    get_profile_avatar_url: Optional[Callable[[str], Optional[str]]] = None


_IMPL: Optional[Impl] = None


def set_impl(impl: Impl) -> None:
    """Register the actual implementations from your main code."""
    global _IMPL
    _IMPL = impl


def _require_impl() -> Impl:
    if _IMPL is None:
        raise RuntimeError("Implementation not set. Call set_impl(Impl(...)) first.")
    return _IMPL


# ------------------------------------------------------------------
# Feature registry
# ------------------------------------------------------------------

FeatureFunc = Callable[[Any, Dict[str, Any]], Dict[str, Any]]
_FEATURES: Dict[str, FeatureFunc] = {}


def register_feature(name: str):
    """Decorator to register a feature callable."""
    def deco(fn: FeatureFunc):
        _FEATURES[name] = fn
        return fn
    return deco


@dataclass
class FeatureSpec:
    """Specification: which feature to run with which parameters."""
    kind: str
    params: Dict[str, Any]


def run_features(es: "Elasticsearch", specs: List[FeatureSpec], impl: Optional[Impl] = None) -> List[Dict[str, Any]]:
    """Run a list of features in order and return their outputs.

    Args:
        es: Elasticsearch client
        specs: List of FeatureSpec
        impl: Optional Impl if you prefer per-call injection (otherwise uses global set_impl)
    """
    results: List[Dict[str, Any]] = []
    prev_impl = None
    if impl is not None:
        # Temporarily set impl for this call
        prev_impl = _IMPL
        set_impl(impl)

    try:
        for spec in specs:
            fn = _FEATURES.get(spec.kind)
            if not fn:
                results.append({"kind": spec.kind, "error": f"feature '{spec.kind}' not registered"})
                continue
            try:
                out = fn(es, spec.params)
                out["kind"] = spec.kind
                results.append(out)
            except Exception as e:
                results.append({"kind": spec.kind, "error": str(e)})
        return results
    finally:
        if impl is not None:
            # restore previous impl
            set_impl(prev_impl)  # type: ignore


# ------------------------------------------------------------------
# Built-in feature wrappers (call your existing functions via Impl)
# ------------------------------------------------------------------

import pandas as pd
from typing import List, Optional, Tuple

# def get_post_texts(es, query_body, index="postindex-*", size=1000):
#     res = es.search(index=index, body=query_body, size=size)
#     hits = res["hits"]["hits"]
#     texts = []

#     for h in hits:
#         source = h["_source"]
#         commit = source.get("commit", {})
#         record = commit.get("record", {})
#         text = record.get("text")
#         did = source.get("did")
#         rkey = commit.get("rkey")
#         cid = commit.get("cid")

#         if text and did and rkey:
#             uri = f"at://{did}/app.bsky.feed.post/{rkey}"
#             web_url = f"https://bsky.app/profile/{did}/post/{rkey}"

#             # --- Like Count ---
#             like_query = {
#                 "size": 0,
#                 "query": {
#                     "term": {
#                         "commit.record.subject.uri.keyword": uri
#                     }
#                 },
#                 "aggs": {
#                     "like_count": {
#                         "value_count": {
#                             "field": "commit.record.subject.uri.keyword"
#                         }
#                     }
#                 }
#             }
#             like_res = es.search(index="likeindex-*", body=like_query)
#             like_count = like_res.get("aggregations", {}).get("like_count", {}).get("value", 0)

#             # --- Repost Count ---
#             repost_query = {
#                 "size": 0,
#                 "query": {
#                     "term": {
#                         "commit.record.subject.uri.keyword": uri
#                     }
#                 },
#                 "aggs": {
#                     "repost_count": {
#                         "value_count": {
#                             "field": "commit.record.subject.uri.keyword"
#                         }
#                     }
#                 }
#             }
#             repost_res = es.search(index="repostindex-*", body=repost_query)
#             repost_count = repost_res.get("aggregations", {}).get("repost_count", {}).get("value", 0)

#             texts.append({
#                 "did": did,
#                 "text": text,
#                 "created_at": record.get("createdAt"),
#                 "rkey": rkey,
#                 "uri": uri,
#                 "url": web_url,
#                 "like_count": like_count,
#                 "repost_count": repost_count
#                 # "cid": cid
#             })

#     return pd.DataFrame(texts)

# import pandas as pd
# from typing import List, Dict

# --- タイムアウト対策：msearch 多発ではなく 1回の terms + terms agg で一括集計 ---

# ===============================
# CID 基準で Like / Repost を集計
# ===============================

def _extract_post_cid(src: dict) -> Optional[str]:
    """post の _source から CID を取り出す。commit.cid → root cid → event.original の順でフォールバック。"""
    if not isinstance(src, dict):
        return None
    commit = src.get("commit") or {}
    cid = commit.get("cid") or src.get("cid") or (commit.get("record") or {}).get("cid")
    if cid:
        return str(cid).strip() or None
    # まれに event.original に JSON 文字列で入っている場合
    ev = src.get("event.original") or (src.get("event") or {}).get("original")
    if isinstance(ev, str):
        try:
            obj = json.loads(ev)
            cid = (obj.get("commit") or {}).get("cid")
            if cid:
                return str(cid).strip()
        except Exception:
            pass
    return None

def _terms_agg_counts_by_field(
    es, *, index: str, ids: List[str], field: str,
    op_filter_field: Optional[str] = "commit.operation",
    op_filter_value: Optional[str] = "create",
    chunk_size: int = 5000, request_timeout: int = 60
) -> Dict[str, int]:
    uniq = [i.strip() for i in dict.fromkeys(ids) if isinstance(i, str) and i.strip()]
    out: Dict[str, int] = {}
    if not uniq:
        return out
    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i:i+chunk_size]
        must_filters = [{"terms": {field: chunk}}]
        if op_filter_field and op_filter_value:
            must_filters.append({"term": {op_filter_field: op_filter_value}})
        body = {
            "size": 0,
            "track_total_hits": False,
            "query": {"bool": {"filter": must_filters}},
            "aggs": {"by_id": {"terms": {"field": field, "size": len(chunk), "shard_size": len(chunk)+128}}},
        }
        res = es.search(
            index=index, body=body,
            ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
            request_timeout=request_timeout, request_cache=True
        )
        buckets = (res.get("aggregations") or {}).get("by_id", {}).get("buckets", []) or []
        for id_ in chunk:
            out.setdefault(id_, 0)
        for b in buckets:
            out[b["key"]] = int(b.get("doc_count", 0))
    return out

import pandas as pd

def _row_from_post(p):
    rec = (p.get("record") or {})
    author = (p.get("author") or {})
    uri = p.get("uri")
    rkey = (uri or "").split("/")[-1] if uri else None
    prof = author.get("handle") or author.get("did")
    url = f"https://bsky.app/profile/{prof}/post/{rkey}" if (prof and rkey) else None
    return {
        "did": author.get("did"),
        "rkey": rkey,
        "cid": p.get("cid") or rec.get("cid"),
        "text": rec.get("text"),
        "created_at": rec.get("createdAt"),
        "uri": uri,
        "url": url,
        "like_count": 0,
        "repost_count": 0,
    }

def get_post_texts_api(bsky: BlueskyClient, query: str, start_iso: str, end_iso: str, lang_filter: str|None, limit_total=2000):
    start = pd.to_datetime(start_iso, utc=True); end = pd.to_datetime(end_iso, utc=True)
    rows, uris = [], []
    for p in bsky.search_posts(query, limit_total=limit_total):
        ts = pd.to_datetime((p.get("record") or {}).get("createdAt"), utc=True, errors="coerce")
        if ts is pd.NaT or ts < start or ts > end: continue
        langs = (p.get("record") or {}).get("langs") or p.get("langs")
        if lang_filter and isinstance(langs, list) and langs and (lang_filter not in langs): continue
        row = _row_from_post(p); rows.append(row)
        if row["uri"]: uris.append(row["uri"])
    df = pd.DataFrame(rows, columns=["did","rkey","cid","text","created_at","uri","url","like_count","repost_count"])
    if df.empty: return df
    meta = bsky.get_posts(uris)
    df["like_count"]   = df["uri"].map(lambda u: (meta.get(u) or {}).get("likeCount", 0)).fillna(0).astype(int)
    df["repost_count"] = df["uri"].map(lambda u: (meta.get(u) or {}).get("repostCount", 0)).fillna(0).astype(int)
    df["cid"] = df["cid"].where(df["cid"].notna(), df["uri"].map(lambda u: (meta.get(u) or {}).get("cid")))
    return df

def get_post_time_series_api(bsky: BlueskyClient, query: str, start_iso: str, end_iso: str, lang_filter: str|None, freq="10min", limit_total=5000):
    # まず投稿を集め、クライアント側でバケット集計（既存の plot 関数と互換の ['time','count'] を返す）
    start = pd.to_datetime(start_iso, utc=True); end = pd.to_datetime(end_iso, utc=True)
    times = []
    for p in bsky.search_posts(query, limit_total=limit_total):
        ts = pd.to_datetime((p.get("record") or {}).get("createdAt"), utc=True, errors="coerce")
        if ts is pd.NaT or ts < start or ts > end: continue
        langs = (p.get("record") or {}).get("langs") or p.get("langs")
        if lang_filter and isinstance(langs, list) and langs and (lang_filter not in langs): continue
        times.append(ts)
    if not times: return pd.DataFrame(columns=["time","count"])
    s = pd.Series(pd.to_datetime(times, utc=True)).dt.floor(freq)
    vc = s.value_counts().sort_index()
    return pd.DataFrame({"time": vc.index.astype(str), "count": vc.values})

def resolve_freq_input(freq: Union[str, int]) -> str:
    """
    freq が int の場合は「分」扱い（例: 10 -> '10min'）
    freq が str の場合は、よく使う日本語/別名を正規化して返す（例: '1日' -> '1D', '1時間' -> '1H'）
    """
    if isinstance(freq, int):
        if freq <= 0:
            raise ValueError("freq (minutes) must be positive.")
        freq_alias = f"{freq}min"
    else:
        s = str(freq).strip().lower()
        alias_map = {
            "10m": "10min", "10min": "10min", "10分": "10min",
            "30m": "30min", "30min": "30min", "30分": "30min",
            "1h": "1H", "1hour": "1H", "1時間": "1H",
            "6h": "6H", "6時間": "6H",
            "12h": "12H", "12時間": "12H",
            "1d": "1D", "1day": "1D", "1日": "1D", "日次": "1D", "daily": "1D",
            "1w": "1W", "1week": "1W", "1週間": "1W", "週次": "1W", "weekly": "1W",
        }
        freq_alias = alias_map.get(s, s)  # 未知はそのまま（例: '15min', '2H' など）

    # 妥当性チェック（不正なら例外）
    pd.tseries.frequencies.to_offset(freq_alias)
    return freq_alias

import pandas as pd

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
            es, index="likeindex-*", ids=cids, field="commit.record.subject.cid"
        )
        repost_counts = _terms_agg_counts_by_field(
            es, index="repostindex-*", ids=cids, field="commit.record.subject.cid"
        )

        df["like_count"] = df["cid"].astype(str).map(like_counts).fillna(0).astype(int)
        df["repost_count"] = df["cid"].astype(str).map(repost_counts).fillna(0).astype(int)
    else:
        df["like_count"] = pd.Series(dtype="int64")
        df["repost_count"] = pd.Series(dtype="int64")

    return df

def _user_recent_posts(
    es,
    did: str,
    *,
    start_dt,
    end_dt,
    max_posts: int = 20,
    post_index: str = "postindex-*",
) -> pd.DataFrame:
    """
    特定ユーザ did の期間内投稿を新しい順に max_posts 件だけ取得。
    """
    body: Dict = {
        "_source": [
            "did",
            "commit.record.text",
            "commit.record.createdAt",
        ],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"did": did}},
                    {
                        "range": {
                            "commit.record.createdAt": {
                                "gte": start_dt.isoformat(),
                                "lte": end_dt.isoformat(),
                            }
                        }
                    },
                    {"term": {"commit.collection": "app.bsky.feed.post"}},
                ]
            }
        },
        "sort": [{"commit.record.createdAt": {"order": "desc"}}],
        "size": max_posts,
    }

    res = es.search(
        index=post_index,
        body=body,
        ignore_unavailable=True,
        allow_no_indices=True,
        expand_wildcards="open",
        request_timeout=120,
    )

    rows = []
    for h in (res.get("hits") or {}).get("hits") or []:
        src = h.get("_source") or {}
        commit = src.get("commit") or {}
        record = commit.get("record") or {}
        text = record.get("text") or ""
        created_at = record.get("createdAt")
        rows.append(
            {
                "did": src.get("did"),
                "text": text,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows)
    
def get_post_time_series(es, query, start, end, lang_filter=None):
    must_clauses = [
        {"match": {"commit.record.text": query}},
        {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
    ]
    if lang_filter:
        must_clauses.append({"term": {"commit.record.langs": lang_filter}})

    body = {
        "size": 0,
        "track_total_hits": True,  # ← 総ヒット数も取得
        "query": {"bool": {"must": must_clauses}},
        "aggs": {
            "posts_over_time": {
                "date_histogram": {
                    "field": "commit.record.createdAt",
                    "fixed_interval": "10m",
                    # "format": "strict_date_optional_time",  # ←必要なら有効化（key_as_stringが確実に出ます）
                    # "time_zone": "+09:00",  # ←必要に応じて（格納がUTCなら不要）
                }
            }
        }
    }

    res = es.search(index="postindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
    buckets = (res.get("aggregations", {})
                 .get("posts_over_time", {})
                 .get("buckets", []))

    # 空でも columns を固定で返す（Plotly のエラー回避）
    if not buckets:
        return pd.DataFrame(columns=["time", "count"])

    rows = []
    for b in buckets:
        ts = b.get("key_as_string")
        if not ts:
            # key (epoch ms) から ISO へ
            ts = pd.to_datetime(b.get("key"), unit="ms", utc=True).isoformat()
        rows.append({"time": ts, "count": b.get("doc_count", 0)})

    return pd.DataFrame(rows, columns=["time", "count"])

def resolve_freq_input(freq: Union[str, int]) -> str:
    """
    freq が int の場合は「分」扱い（例: 10 -> '10min'）
    freq が str の場合は、よく使う日本語/別名を正規化して返す（例: '1日' -> '1D', '1時間' -> '1H'）
    """
    if isinstance(freq, int):
        if freq <= 0:
            raise ValueError("freq (minutes) must be positive.")
        freq_alias = f"{freq}min"
    else:
        s = str(freq).strip().lower()
        alias_map = {
            "10m": "10min", "10min": "10min", "10分": "10min",
            "30m": "30min", "30min": "30min", "30分": "30min",
            "1h": "1H", "1hour": "1H", "1時間": "1H",
            "6h": "6H", "6時間": "6H",
            "12h": "12H", "12時間": "12H",
            "1d": "1D", "1day": "1D", "1日": "1D", "日次": "1D", "daily": "1D",
            "1w": "1W", "1week": "1W", "1週間": "1W", "週次": "1W", "weekly": "1W",
        }
        freq_alias = alias_map.get(s, s)  # 未知はそのまま（例: '15min', '2H' など）

    # 妥当性チェック（不正なら例外）
    pd.tseries.frequencies.to_offset(freq_alias)
    return freq_alias

def plot_timeseries_compare_from_posts(
    df_a, df_b,
    *, created_at_col="created_at", freq="10min", tz="Asia/Tokyo",
    title="A vs B（重ね＋差分）", use_streamlit=True
):
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import streamlit as st

    # 既存の集計関数で A / B を同じ粒度・同じTZに集計
    tsa = build_timeseries_df_from_posts(df_a, created_at_col, None, freq, tz, add_time_col=False)
    tsb = build_timeseries_df_from_posts(df_b, created_at_col, None, freq, tz, add_time_col=False)

    # ── 列名と型を正規化 ──
    ta = _normalize_ts_df(tsa).rename(columns={"count": "A"})
    tb = _normalize_ts_df(tsb).rename(columns={"count": "B"})

    # 時間軸をユニオン
    all_times = pd.Index(ta["time"]).union(pd.Index(tb["time"])).sort_values()
    base = pd.DataFrame({"time": all_times})

    # マージ（欠損は0）
    m = (base.merge(ta, on="time", how="left")
              .merge(tb, on="time", how="left"))
    m["A"] = m["A"].fillna(0).astype(int)
    m["B"] = m["B"].fillna(0).astype(int)
    m["diff"] = m["A"] - m["B"]
    m["sum"]  = m["A"] + m["B"]
    m["ratioA"] = np.where(m["sum"] > 0, m["A"] / m["sum"], np.nan)

    # ── 可視化：折れ線2本（A/B）＋差分帯＋比率線 ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["time"], y=m["A"], mode="lines", name="A"))
    fig.add_trace(go.Scatter(x=m["time"], y=m["B"], mode="lines", name="B"))

    # 差分帯（A-B）：上側/下側を 0 を基準に塗り分け
    fig.add_trace(go.Scatter(
        x=pd.concat([m["time"], m["time"][::-1]]),
        y=pd.concat([m["diff"].clip(lower=0), m["diff"].clip(upper=0)[::-1]]),
        fill="toself", name="A−B（差分帯）", hoverinfo="skip", opacity=0.25
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Posts (count)",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # 比率線（別図）
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=m["time"], y=m["ratioA"], mode="lines", name="A ratio"))
    fig2.update_layout(
        title="Aの占有率（A/(A+B)）",
        yaxis=dict(tickformat=".0%"),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    if use_streamlit:
        st.plotly_chart(fig, use_container_width=True, key="cmp_overlay")
        st.plotly_chart(fig2, use_container_width=True, key="cmp_ratio")

    return m, fig, fig2

def build_timeseries_df_from_posts(
    df: pd.DataFrame,
    created_at_col: str = "created_at",
    label_col: Optional[str] = None,
    freq: Union[str, int] = "10min",   # ← ここを柔軟に。例: "10min" / "1H" / "1D" / 10（分）
    tz: Optional[str] = "Asia/Tokyo",
    add_time_col: bool = True,
    label_side: str = "left",          # resample の label（'left' | 'right'）
    closed: str = "left",              # 区間の閉区間側（'left' | 'right'）
) -> pd.DataFrame:
    """
    与えられた投稿DataFrameから、created_atをtimeに変換し、freqごとの件数に集計した時系列DFを返す。
      - label_col を指定すると、ラベル別の時系列（列: [label_col,'time','count']）を返す
      - freq は '10min' / '1H' / '1D' / '1W' などの pandas オフセット、または「分」を表す整数
      - label_side/closed でバケット境界の表示・集計側を制御

    戻り値:
      単一系列: columns=['time','count']
      複数系列: columns=[label_col,'time','count']
    """
    if df is None or df.empty or created_at_col not in df.columns:
        return pd.DataFrame(columns=([label_col] if label_col else []) + ["time", "count"])

    # freq を正規化
    try:
        freq_alias = resolve_freq_input(freq)
    except Exception as e:
        raise ValueError(f"Invalid freq '{freq}': {e}")

    # created_at -> tz-aware datetime
    t = pd.to_datetime(df[created_at_col], errors="coerce", utc=True)
    if tz:
        try:
            t = t.dt.tz_convert(tz)
        except Exception:
            t = pd.to_datetime(df[created_at_col], errors="coerce", utc=True).dt.tz_convert(tz)

    # 元dfにも追加（必要なら）
    if add_time_col:
        df["time"] = t

    # 有効な行のみ
    tmp = df.copy()
    tmp["time"] = t
    tmp = tmp.dropna(subset=["time"]).sort_values("time")
    if tmp.empty:
        return pd.DataFrame(columns=([label_col] if label_col else []) + ["time", "count"])

    # 集計
    if label_col and label_col in tmp.columns:
        gr = tmp.set_index("time").groupby(tmp[label_col]).resample(freq_alias, label=label_side, closed=closed)
        out = gr.size().rename("count").reset_index()  # columns=[label_col,'time','count']
        return out
    else:
        out = (
            tmp.set_index("time")
               .resample(freq_alias, label=label_side, closed=closed)
               .size()
               .rename("count")
               .reset_index()
        )
        return out

def _normalize_ts_df(ts_df):
    """
    build_timeseries_df_from_posts の返りを ["time","count"] に正規化する。
    - pandasや分岐により "size" や 0 列名になるケースを吸収
    - 型も Datetime(UTC) と int に揃える
    """
    import pandas as pd
    import numpy as np

    if ts_df is None or len(ts_df) == 0:
        return pd.DataFrame({"time": pd.to_datetime([], utc=True), "count": []}, dtype="int64")

    df = ts_df.copy()

    # time 列を特定（通常 "time" だが保険）
    time_col = "time" if "time" in df.columns else None
    if time_col is None:
        # time らしい列を推定（datetime 型っぽい最初の列）
        for c in df.columns:
            if np.issubdtype(pd.Series(df[c]).dtype, np.datetime64):
                time_col = c
                break
        if time_col is None:
            # 強制パース（失敗時は NaT）
            time_col = df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], utc=True)

    # count 列を特定
    count_col = None
    cand = ["count", "size", 0, "n", "value"]
    for c in cand:
        if c in df.columns:
            count_col = c
            break
    if count_col is None:
        # 最後の列を count とみなす（time 以外）
        others = [c for c in df.columns if c != "time"]
        if len(others) == 0:
            # もし raw 行なら time ごとに集計
            df = df.groupby("time", as_index=False).size()
            count_col = "size"
        else:
            count_col = others[-1]

    df = df[["time", count_col]].rename(columns={count_col: "count"})
    # 欠損と型
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    # 重複行があっても同一 time 内は加算しておく
    df = df.groupby("time", as_index=False)["count"].sum().sort_values("time")
    return df
        
def plot_timeseries_from_posts(
    df: pd.DataFrame,
    created_at_col: str = "created_at",
    label_col: str | None = None,
    freq: str = "10min",
    tz: str | None = "Asia/Tokyo",
    title: str = "投稿数の時系列",
    add_time_col: bool = True,
):
    """
    build_timeseries_df_from_posts() で作った時系列を Plotly で描画。
    戻り値として集計済みDFを返す（画面描画は副作用）。
    """
    ts_df = build_timeseries_df_from_posts(
        df=df,
        created_at_col=created_at_col,
        label_col=label_col,
        freq=freq,
        tz=tz,
        add_time_col=add_time_col,
    )

    # 可視化（安全に）
    if ts_df is None or ts_df.empty or "time" not in ts_df.columns or "count" not in ts_df.columns:
        st.info(f"{title}: データがありません（created_at列や期間を確認）")
        return ts_df

    ts_df = ts_df.sort_values("time")
    if label_col and label_col in ts_df.columns:
        fig = px.line(ts_df, x="time", y="count", color=label_col, title=title)
    else:
        fig = px.line(ts_df, x="time", y="count", title=title)

    st.plotly_chart(fig, use_container_width=True)
    return ts_df

def _detect_text_col(df: pd.DataFrame, text_col: Optional[str]) -> str:
    if text_col and text_col in df.columns:
        return text_col
    candidates = [
        "text", "post_text", "body", "content", "message", "caption",
        "full_text", "commit.record.text", "record_text"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"text column not found. pass text_col explicitly. candidates tried: {candidates}")

# Basic stopwords (en + ja). 追加したいときは extra_stopwords 引数で。
STOPWORDS_EN = {
    "the","a","an","and","or","of","to","in","on","for","with","at","by","from","as","is","are","was","were",
    "be","been","being","that","this","it","its","if","else","but","not","no","so","do","did","does","doing",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","his","their","our",
    "rt","via","amp"
}
STOPWORDS_JA = {
    "の","に","は","を","た","が","で","て","と","し","れ","さ","ある","いる","も","する","から","な","こと",
    "として","い","や","れる","など","なっ","ない","この","ため","その","あの","これ","それ","あれ","よう",
    "また","ます","です","でした","ですが","ので","ね","よ","だ","なり","なる","件","さん","www","笑","w"
}
STOPWORDS_DEFAULT = STOPWORDS_EN | STOPWORDS_JA

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[一-龥々〆ヵヶぁ-んァ-ヴー]+|[A-Za-z0-9_]+")  # JA block or latin/num/underscore
HASHTAG_MENTION_RE = re.compile(r"[@#]")

def _is_japanese_token(tok: str) -> bool:
    return bool(re.search(r"[一-龥ぁ-んァ-ン]", tok))

def tokenize_simple(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    t = URL_RE.sub(" ", text)
    t = HASHTAG_MENTION_RE.sub(" ", t)
    t = t.lower()
    tokens = TOKEN_RE.findall(t)
    return tokens

# ----------------------------
# 1) 単語ランキング
# ----------------------------

def build_word_ranking_from_posts(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    top_n: int = 50,
    min_ascii_len: int = 2,
    max_token_len: int = 64,
    extra_stopwords: Optional[Iterable[str]] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    投稿DataFrameから単語出現頻度ランキングを作成。
    - デフォルトの簡易トークナイザ（日本語は連続する漢字/かな/カナを1トークン、英数は単語）
    - 英日ミックスの簡易ストップワードを同梱。extra_stopwords で追加可。

    Returns: DataFrame[["word","count"]] (+ "freq" if normalize=True)
    """
    if df is None or df.empty:
        cols = ["word", "count"] + (["freq"] if normalize else [])
        return pd.DataFrame(columns=cols)

    col = _detect_text_col(df, text_col)
    stop = set(STOPWORDS_DEFAULT)
    if extra_stopwords:
        stop |= set(extra_stopwords)

    counter = Counter()
    total_kept = 0

    for x in df[col].astype(str).tolist():
        toks = tokenize_simple(x)
        for tok in toks:
            if not tok or tok in stop:
                continue
            if len(tok) > max_token_len:
                continue
            if not _is_japanese_token(tok) and len(tok) < min_ascii_len:
                continue
            if tok.isdigit():
                continue
            counter[tok] += 1
            total_kept += 1

    items = counter.most_common(top_n if top_n and top_n > 0 else None)
    out = pd.DataFrame(items, columns=["word", "count"])
    if normalize and total_kept > 0:
        out["freq"] = out["count"] / float(total_kept)
    return out

# ----------------------------
# 2) TF-IDF ランキング（scikit-learn無し）
# ----------------------------

def build_tfidf_ranking_from_posts(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    top_n: int = 50,
    min_ascii_len: int = 2,
    max_token_len: int = 64,
    extra_stopwords: Optional[Iterable[str]] = None,
    tf_norm: str = "rel",       # "rel" (term count / doc len) or "log" (1+log tf)
    agg: str = "avg",           # "avg" | "sum" | "max" で語のスコア集計
    smooth_idf: bool = True,    # True -> idf = log((N+1)/(df+1)) + 1
) -> pd.DataFrame:
    """
    投稿DataFrameからTF-IDFの語ランキングを作成（自前実装）。
    Returns: DataFrame[["word","score","df","idf"]]
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["word", "score", "df", "idf"])

    col = _detect_text_col(df, text_col)
    stop = set(STOPWORDS_DEFAULT)
    if extra_stopwords:
        stop |= set(extra_stopwords)

    # 1) 文書ごとにトークン化＆tfを計算
    docs_tokens: List[List[str]] = []
    for x in df[col].astype(str).tolist():
        toks = tokenize_simple(x)
        filtered = []
        for tok in toks:
            if not tok or tok in stop:
                continue
            if len(tok) > max_token_len:
                continue
            if not _is_japanese_token(tok) and len(tok) < min_ascii_len:
                continue
            if tok.isdigit():
                continue
            filtered.append(tok)
        docs_tokens.append(filtered)

    N = len(docs_tokens)
    if N == 0:
        return pd.DataFrame(columns=["word", "score", "df", "idf"])

    # 2) 語彙と文書頻度 df(term) を集計
    dfreq: Counter = Counter()
    for toks in docs_tokens:
        for term in set(toks):  # 文書内で1回でも出たら +1
            dfreq[term] += 1

    # 3) idf 計算
    idf: dict = {}
    for term, dfi in dfreq.items():
        if smooth_idf:
            idf_val = math.log((N + 1) / (dfi + 1)) + 1.0
        else:
            idf_val = math.log(N / max(1, dfi))
        idf[term] = idf_val

    # 4) 各文書で tf を計算し、tf-idf を集計
    #    tf_norm: "rel" -> tf = count/len(doc), "log" -> 1+log(count)
    accum: defaultdict = defaultdict(float)
    if agg not in {"avg", "sum", "max"}:
        agg = "avg"

    for toks in docs_tokens:
        if not toks:
            continue
        counts = Counter(toks)
        doc_len = sum(counts.values())
        for term, c in counts.items():
            if tf_norm == "log":
                tf = 1.0 + math.log(float(c))
            else:
                tf = float(c) / float(doc_len) if doc_len > 0 else 0.0
            tfidf = tf * idf.get(term, 0.0)
            if agg == "sum":
                accum[term] += tfidf
            elif agg == "max":
                accum[term] = max(accum[term], tfidf)
            else:  # avg
                # 平均にしたいので一旦合計を入れておき、最後に / 文書出現数
                accum[term] += tfidf

    # 5) 平均化（必要なら）
    scores: List[Tuple[str, float]] = []
    for term, val in accum.items():
        if agg == "avg":
            scores.append((term, val / float(dfreq[term])))
        else:
            scores.append((term, val))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[: top_n if top_n and top_n > 0 else None]

    # 6) 出力DF
    out = pd.DataFrame(top, columns=["word", "score"])
    out["df"] = out["word"].map(lambda w: dfreq.get(w, 0))
    out["idf"] = out["word"].map(lambda w: idf.get(w, 0.0))
    return out

import plotly.express as px
import pandas as pd

def plot_word_ranking_bar(
    ranking_df: pd.DataFrame,
    *,
    x_col: str = "count",     # build_word_ranking_from_posts の出力に合わせる
    y_col: str = "word",
    title: str = "単語ランキング（頻度）",
    top_n: int | None = None,
    use_streamlit: bool = True,
):
    """
    単語ランキング DataFrame を横棒グラフで可視化する。
    ranking_df: columns に ["word","count"]（任意で"freq"）を想定
    """
    if ranking_df is None or ranking_df.empty or not {x_col, y_col}.issubset(ranking_df.columns):
        try:
            if use_streamlit:
                import streamlit as st
                st.info("単語ランキングの可視化対象データがありません。")
        except Exception:
            pass
        return None

    df = ranking_df[[y_col, x_col]].dropna()
    if top_n is not None and top_n > 0:
        df = df.nlargest(top_n, x_col)

    # 横棒で上位が上に来るように昇順で並べ、categoryorder を固定
    df = df.sort_values(x_col, ascending=True)
    fig = px.bar(df, x=x_col, y=y_col, orientation="h", title=title, text=x_col)
    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=df[y_col].tolist()),
        xaxis_title="Count",
        yaxis_title="Word",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    if use_streamlit:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True)
    return fig


def plot_tfidf_ranking_bar(
    tfidf_df: pd.DataFrame,
    *,
    x_col: str = "score",     # build_tfidf_ranking_from_posts の出力に合わせる
    y_col: str = "word",
    title: str = "TF-IDF ランキング",
    top_n: int | None = None,
    use_streamlit: bool = True,
):
    """
    TF-IDF ランキング DataFrame を横棒グラフで可視化する。
    小数表示は小数点以下3桁に整形。
    """
    try:
        import plotly.express as px
    except Exception:
        # plotly が無い環境でも落ちないように
        if use_streamlit:
            import streamlit as st
            st.error("plotly がインポートできません。")
        return None

    if tfidf_df is None or tfidf_df.empty or not {x_col, y_col}.issubset(tfidf_df.columns):
        if use_streamlit:
            import streamlit as st
            st.info("TF-IDF ランキングの可視化対象データがありません。")
        return None

    df = tfidf_df[[y_col, x_col]].dropna()
    if top_n is not None and top_n > 0:
        df = df.nlargest(top_n, x_col)

    # 表示用に丸めた列を用意（小数点以下3桁）
    df = df.sort_values(x_col, ascending=True).copy()
    df["__score_fmt__"] = df[x_col].round(3)

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation="h",
        title=title,
        text="__score_fmt__",  # 外側テキストは丸め値を表示
    )

    # 目盛り・ホバーの小数表示を3桁に統一
    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=df[y_col].tolist()),
        xaxis_title="TF-IDF score",
        yaxis_title="Word",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickformat=".3f"),  # 目盛り
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>score=%{x:.3f}<extra></extra>",  # ホバー
    )

    if use_streamlit:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True)
    return fig

# 最新モデル優先のトークナイザ・ファクトリ（spaCy最新パイプライン→GiNZA→SudachiPy→簡易）
# URL由来トークンの除去を強化した最新版トークナイザ
import re
from functools import lru_cache
from typing import Callable, Iterable, List, Optional, Tuple

# 既存の基本ストップワード（必要に応じてマージされます）
DEFAULT_STOP_EN = {
    "the","a","an","and","or","of","to","in","on","for","with","at","by","from","as","is","are","was","were",
    "be","been","being","that","this","it","its","if","else","but","not","no","so","do","did","does","doing",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","his","their","our","rt","via","amp"
}
DEFAULT_STOP_JA = {
    "の","に","は","を","た","が","で","て","と","し","れ","さ","ある","いる","も","する","から","な","こと",
    "として","い","や","れる","など","なっ","ない","この","ため","その","あの","これ","それ","あれ","よう",
    "また","ます","です","でした","ですが","ので","ね","よ","だ","なり","なる","さん","件","笑","ｗ","www"
}
DEFAULT_STOP = DEFAULT_STOP_EN | DEFAULT_STOP_JA

# URL/ドメイン由来の文字列を積極的に除外
URL_TOKEN_STOPWORDS = {
    # 一般
    "http","https","www","com","net","org","jp","co","io","ai","app","dev","info","me","it","tv","uk","de","fr",
    "ru","cn","kr","us","au","ca",
    # パラメータ・拡張子
    "utm","source","medium","campaign","ref","sref","fbclid","gclid","clickid","php","html","htm","json","xml",
    "pdf","jpg","jpeg","png","gif","webp","svg",
    # サービス名・ニュース系
    "google","yahoo","bing","naver","baidu","duckduckgo","news","article","articles","amp","t","co","bit","ly","bitly","毎日新聞","産経新聞","産経ニュース","ニュース","nhk","livecam","oc","朝日新聞","時事ドットコム","rss","読売新聞オンライン","jal","jtb","pr","速報","bsky","social",
    "tinyurl","youtu","youtube","insta","instagram","x","twitter","tiktok","note","medium","qiita","zenn",
    "github","gitlab","docs","drive","mail","gmail","outlook","icloud","line","blog","reddit","hatenablog"
}

# URL全体やドメインっぽいトークンの検出
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_AT_HASH_RE = re.compile(r"[@#]")
_DOMAIN_LIKE_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+$", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
_SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}|[\u3040-\u30FF\u4E00-\u9FFF]{2,}")

_JA_PARTICLES = set("のにはをとでもがかへやからまでよりって".split())
# ひらかな/カナ/句読点など
_PUNCT_JA = re.compile(r"[。、…・！（）\(\)\[\]「」『』《》【】]")

def _looks_phrase_like_ja(tok: str) -> bool:
    """助詞が複数/句読点含む/空白含む/かなのみ長大 → 文片とみなして除外"""
    if not tok or " " in tok:
        return True
    if _PUNCT_JA.search(tok):
        return True
    # 助詞数（の/に/は/を/と/で/も/が/か/へ/や/から/まで/より/って）
    part_cnt = sum(tok.count(p) for p in _JA_PARTICLES)
    if part_cnt >= 2:
        return True
    # かなのみで長過ぎる
    if re.fullmatch(r"[ぁ-んァ-ンー]{10,}", tok):
        return True
    return False

def _simple_tokenize(text: str, *, stop: set, min_len: int, max_len: int) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    t = _URL_RE.sub(" ", text)          # URL丸ごと除去
    t = _AT_HASH_RE.sub(" ", t)         # @, # 除去
    t = t.lower()
    out: List[str] = []
    for m in _SIMPLE_TOKEN_RE.finditer(t):
        w = m.group(0)
        # URL由来や短すぎ・長すぎ、数字のみを除外
        if (
            w in stop or
            w.isdigit() or
            len(w) < min_len or
            len(w) > max_len or
            _DOMAIN_LIKE_RE.match(w) or
            any(s in w for s in ("http", "www"))
        ):
            continue
        out.append(w)
    return out

@lru_cache(maxsize=1)
def _load_spacy_model(name: str):
    try:
        import spacy  # type: ignore
        return spacy.load(name, disable=["ner","parser","textcat"])
    except Exception:
        return None

@lru_cache(maxsize=1)
def _load_sudachi():
    try:
        from sudachipy import dictionary, tokenizer as sudachi_tokenizer  # type: ignore
        obj = dictionary.Dictionary().create()
        return obj, sudachi_tokenizer.Tokenizer.SplitMode.C
    except Exception:
        return None, None

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Literal

def remove_high_score_outliers(
    df: pd.DataFrame,
    score_col: str = "count",                   # 単語頻度は "count", TF-IDF は "score"
    method: Literal["auto","quantile","iqr","zscore"] = "auto",
    q: float = 0.99,                            # quantile: 上位1%を閾値候補
    iqr_k: float = 3.0,                         # iqr: Q3 + k*IQR
    z_k: float = 4.0,                           # zscore: μ + kσ
    min_samples: int = 20,                      # 統計的に扱う最小件数
    keep_at_least: int = 0,                    # 少なくとも上位K件は残す
    clip: bool = False,                         # Trueなら除外せず閾値でクリップ（ウィンズライズ）
) -> Tuple[pd.DataFrame, Dict]:
    """
    スコア（頻度/TF-IDF）が極端に大きい項目を除外 or クリップする。
    method="auto" は quantile / iqr / zscore の各閾値候補のうち最も保守的（最小）を採用。
    返り値: (処理後DataFrame, メタ情報dict)
    """
    if df is None or df.empty or score_col not in df.columns:
        return df.copy(), {"skipped": "no_data_or_score_col"}

    x = df[score_col].astype(float).dropna().to_numpy()
    n = x.size
    if n < min_samples:
        return df.copy(), {"skipped": f"too_few_samples({n}<{min_samples})"}

    thr_candidates = []
    if method in ("quantile", "auto"):
        thr_candidates.append(np.quantile(x, q))
    if method in ("iqr", "auto"):
        q1, q3 = np.quantile(x, [0.25, 0.75])
        thr_candidates.append(q3 + iqr_k * (q3 - q1))
    if method in ("zscore", "auto"):
        mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else 0.0
        if sigma > 0:
            thr_candidates.append(mu + z_k * sigma)

    thr = min(thr_candidates) if thr_candidates else np.inf

    # 上位K件は必ず残す（除外し過ぎ防止）
    if keep_at_least and n >= keep_at_least:
        kth = np.sort(x)[::-1][keep_at_least - 1]
        if thr < kth:
            thr = kth

    if clip:
        out = df.copy()
        out[score_col] = np.minimum(out[score_col].astype(float), thr)
        return out, {"threshold": float(thr), "method": method, "clip": True}

    mask = df[score_col].astype(float) <= thr
    out = df.loc[mask].reset_index(drop=True)
    removed = int((~mask).sum())
    return out, {"threshold": float(thr), "method": method, "clip": False, "removed": removed}


# ラッパー（そのまま使えるように）
def filter_word_ranking_high(wr_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """単語頻度ランキング（columns=['word','count',...]) 用の高スコア除去/クリップ"""
    return remove_high_score_outliers(wr_df, score_col="count", **kwargs)

def filter_tfidf_ranking_high(tfidf_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """TF-IDFランキング（columns=['word','score',...]) 用の高スコア除去/クリップ"""
    return remove_high_score_outliers(tfidf_df, score_col="score", **kwargs)

def remove_top_percent(
    df: pd.DataFrame,
    *,
    score_col: str = "count",                 # 単語頻度は "count"、TF-IDFは "score"
    top_pct: float = 0.05,                    # 上位何パーセントを除去するか (0<pct<1)
    tie: Literal["min","average","first","dense","max"] = "min",
    clip: bool = False,                       # True: 除去せず閾値で頭打ち（ウィンズライズ）
    group_col: Optional[str] = None,          # 例: クエリA/Bなどグループごとに処理したい場合
) -> Tuple[pd.DataFrame, Dict]:
    """
    ランキングDataFrameから“上位 top_pct%”を除去（またはクリップ）する。
    tie: rank() の method。'min' は同値をすべて同順位→上位にまとめて落としやすい。
    戻り値: (処理後DataFrame, {'threshold': 閾値, 'removed': 件数, 'clip': bool})
    """
    if df is None or df.empty or score_col not in df.columns:
        return df.copy(), {"skipped": "no_data_or_score_col"}

    if not (0.0 < top_pct < 1.0):
        raise ValueError("top_pct must be between 0 and 1 (e.g., 0.05 for top 5%).")

    def _proc(sub: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        # 上位パーセンタイル順位を計算（降順でpct=True）
        ranks_pct = sub[score_col].rank(method=tie, ascending=False, pct=True)
        # 上位 top_pct を判定（<= top_pct が“上位”）
        top_mask = ranks_pct <= top_pct

        # 閾値参考用（スコアの (1 - top_pct) 分位点）
        thr = float(sub[score_col].quantile(1 - top_pct))

        if clip:
            out = sub.copy()
            out.loc[top_mask, score_col] = thr
            return out, {"threshold": thr, "removed": 0, "clip": True}

        out = sub.loc[~top_mask].reset_index(drop=True)
        return out, {"threshold": thr, "removed": int(top_mask.sum()), "clip": False}

    if group_col and group_col in df.columns:
        parts = []
        removed_total = 0
        thresholds = {}
        for g, sub in df.groupby(group_col, group_keys=False):
            out_g, meta_g = _proc(sub)
            parts.append(out_g)
            removed_total += meta_g.get("removed", 0)
            thresholds[g] = meta_g.get("threshold")
        return pd.concat(parts, ignore_index=True), {"thresholds": thresholds, "removed": removed_total, "clip": clip}

    return _proc(df)

# 便利ラッパ（そのまま使える）
def filter_word_ranking_top_pct(wr_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """単語頻度ランキング（columns=['word','count',...]）から上位%除去/クリップ"""
    return remove_top_percent(wr_df, score_col="count", **kwargs)

def filter_tfidf_ranking_top_pct(tfidf_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """TF-IDFランキング（columns=['word','score',...]）から上位%除去/クリップ"""
    return remove_top_percent(tfidf_df, score_col="score", **kwargs)

def create_tokenizer_latest(
    *,
    lang: str = "auto",                   # 'ja' | 'en' | 'auto'
    use_lemma: bool = True,
    keep_pos_ja: Tuple[str, ...] = ("名詞","動詞","形容詞","固有名詞"),  # Sudachi 用
    keep_pos_en: Tuple[str, ...] = ("NOUN","PROPN","VERB","ADJ"),      # spaCy 用（ja/en 共通UD）
    extra_stopwords: Optional[Iterable[str]] = None,
    min_len: int = 2,
    max_len: int = 64,
    allow_download: bool = False,
    require_model: bool = False,
) -> Callable[[str], List[str]]:
    """
    最新（環境にある最新版）spaCy→GiNZA→Sudachi→簡易の順で使用。
    URL自体の除去に加え、URLに由来する一般的なトークン（google, yahoo, article, com など）や
    ドメインっぽい文字列、拡張子・追跡パラメータも除外します。
    """
    # ストップワードに URL由来の語も合流
    stop = set(DEFAULT_STOP) | set(URL_TOKEN_STOPWORDS)
    if extra_stopwords:
        stop |= set(extra_stopwords)

    # JA (spaCy)
    nlp_ja = _load_spacy_model("ja_core_news_sm") or _load_spacy_model("ja_ginza") or _load_spacy_model("ja_ginza_electra")
    if nlp_ja is None and allow_download:
        try:
            import spacy
            from spacy.cli import download
            download("ja_core_news_sm")
            nlp_ja = spacy.load("ja_core_news_sm", disable=["ner","parser","textcat"])
        except Exception:
            nlp_ja = None

    # Sudachi
    sudachi_obj, sudachi_mode = _load_sudachi()

    def _tok_ja(text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []
        t = _URL_RE.sub(" ", text)
        # spaCy-ja（UD品詞: NOUN/PROPN/VERB/ADJ）
        if nlp_ja is not None:
            doc = nlp_ja(t)
            out: List[str] = []
            for w in doc:
                if w.is_space or w.is_punct or w.like_url or w.like_email:
                    continue
                # UD品詞でフィルタ（日本語でも spaCy は英語と同じUDタグ）
                if keep_pos_en and w.pos_ not in keep_pos_en:
                    continue
                lemma = (w.lemma_ if use_lemma else w.text).lower().strip()
                if (
                    not lemma or
                    lemma in stop or
                    lemma.isdigit() or
                    len(lemma) < min_len or
                    len(lemma) > max_len or
                    _DOMAIN_LIKE_RE.match(lemma) or
                    any(s in lemma for s in ("http","www"))
                ):
                    continue
                out.append(lemma)
            return out
        # Sudachi 直接
        if sudachi_obj is not None and sudachi_mode is not None:
            toks: List[str] = []
            for m in sudachi_obj.tokenize(sudachi_mode, t):
                pos0 = m.part_of_speech()[0] if m.part_of_speech() else ""
                if keep_pos_ja and pos0 not in keep_pos_ja:
                    continue
                surface = m.dictionary_form() if use_lemma else m.surface()
                surface = surface.strip().lower()
                if (
                    not surface or
                    surface in stop or
                    surface.isdigit() or
                    len(surface) < min_len or
                    len(surface) > max_len or
                    _DOMAIN_LIKE_RE.match(surface) or
                    any(s in surface for s in ("http","www"))
                ):
                    continue
                toks.append(surface)
            return toks
        # フォールバック
        return _simple_tokenize(t, stop=stop, min_len=min_len, max_len=max_len)

    # EN (spaCy)
    nlp_en = _load_spacy_model("en_core_web_sm")
    if nlp_en is None and allow_download:
        try:
            import spacy
            from spacy.cli import download
            download("en_core_web_sm")
            nlp_en = spacy.load("en_core_web_sm", disable=["ner","parser","textcat"])
        except Exception:
            nlp_en = None

    def _tok_en(text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []
        t = _URL_RE.sub(" ", text)
        t = _AT_HASH_RE.sub(" ", t)
        if nlp_en is not None:
            doc = nlp_en(t)
            out: List[str] = []
            for w in doc:
                if w.is_space or w.is_punct or w.like_url or w.like_email:
                    continue
                if keep_pos_en and w.pos_ not in keep_pos_en:
                    continue
                lemma = (w.lemma_ if use_lemma else w.text).lower().strip()
                if (
                    not lemma or
                    lemma in stop or
                    lemma.isdigit() or
                    len(lemma) < min_len or
                    len(lemma) > max_len or
                    _DOMAIN_LIKE_RE.match(lemma) or
                    any(s in lemma for s in ("http","www"))
                ):
                    continue
                out.append(lemma)
            return out
        # フォールバック
        return _simple_tokenize(t, stop=stop, min_len=min_len, max_len=max_len)

    def _tok_auto(text: str) -> List[str]:
        return _tok_ja(text) if _CJK_RE.search(text or "") else _tok_en(text)

    if lang == "ja":
        return _tok_ja
    if lang == "en":
        return _tok_en

    have_model = (nlp_ja is not None) or (sudachi_obj is not None and sudachi_mode is not None) or (nlp_en is not None)
    if require_model and not have_model:
        raise RuntimeError("No JA/EN tokenizer model found. Install ja_core_news_sm or SudachiPy.")
        
    return _tok_auto

def _tokenize_to_set(
    text: str,
    tokenizer: Callable[[str], List[str]],
    *,
    vocab: Optional[set] = None,
    max_len: int = 12,
    reject_phrase_like: bool = True,
) -> set:
    terms = []
    for t in tokenizer(text or ""):
        if reject_phrase_like and _cooccur_looks_phrase_like_ja(t):
            continue
        if len(t) > max_len:
            continue
        if vocab is not None and t not in vocab:
            continue
        terms.append(t)
    return set(terms)  # 文書内の重複は1回に

def _cooccur_looks_phrase_like_ja(tok: str) -> bool:
    import re
    _JA_PARTICLES = ["の","に","は","を","と","で","も","が","か","へ","や","から","まで","より","って"]
    if not tok or " " in tok:
        return True
    if re.search(r"[。、…・！（）\(\)\[\]「」『』《》【】]", tok):
        return True
    if sum(tok.count(p) for p in _JA_PARTICLES) >= 2:
        return True
    if re.fullmatch(r"[ぁ-んァ-ンー]{10,}", tok):
        return True
    return False

# すでに提示した create_tokenizer_latest(...) を利用します

def _detect_text_col(df: pd.DataFrame, text_col: Optional[str]) -> str:
    if text_col and text_col in df.columns:
        return text_col
    candidates = [
        "text", "post_text", "content", "body", "caption", "full_text",
        "commit.record.text", "record_text", "message",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"text column not found. Pass text_col explicitly. Tried: {candidates}"
    )

# --- ランキング側：長すぎ/文片っぽいトークンを除去 -------------------
#============
#単語ランキング
#============
def build_word_ranking_from_posts_latest(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    *,
    lang: str = "auto",
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    top_n: int = 50,
    allow_download: bool = False,
    use_lemma: bool = True,
    keep_pos_ja: tuple[str, ...] = ("名詞","動詞","形容詞","固有名詞"),
    keep_pos_en: tuple[str, ...] = ("NOUN","PROPN","VERB","ADJ"),
    extra_stopwords: Optional[Iterable[str]] = None,
    min_len: int = 2,
    max_len: int = 12,                # ← 既定を短めに（文片の混入を抑える）
    normalize: bool = False,
    reject_phrase_like: bool = True,  # ← 追加
    require_model: bool = True,      # ← 追加
) -> pd.DataFrame:
    if df is None or df.empty:
        cols = ["word", "count"] + (["freq"] if normalize else [])
        return pd.DataFrame(columns=cols)

    col = _detect_text_col(df, text_col)
    tok = tokenizer or create_tokenizer_latest(
        lang=lang, use_lemma=use_lemma,
        keep_pos_ja=keep_pos_ja, keep_pos_en=keep_pos_en,
        extra_stopwords=extra_stopwords,
        min_len=min_len, max_len=max_len,
        allow_download=allow_download,
        require_model=require_model,
    )

    from collections import Counter
    ctr = Counter(); total = 0
    for s in df[col].dropna().astype(str):
        for t in tok(s):
            # ここで追加フィルタ
            if reject_phrase_like and _looks_phrase_like_ja(t):
                continue
            if len(t) > max_len:
                continue
            ctr[t] += 1
            total += 1

    out = pd.DataFrame(ctr.most_common(top_n if top_n else None), columns=["word","count"])
    if normalize and total > 0:
        out["freq"] = out["count"] / float(total)
    return out
#============
#tfidfランキング
#============

def build_tfidf_ranking_from_posts_latest(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    *,
    lang: str = "auto",
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    top_n: int = 50,
    allow_download: bool = False,
    use_lemma: bool = True,
    keep_pos_ja: tuple[str, ...] = ("名詞","動詞","形容詞","固有名詞"),
    keep_pos_en: tuple[str, ...] = ("NOUN","PROPN","VERB","ADJ"),
    extra_stopwords: Optional[Iterable[str]] = None,
    min_len: int = 2,
    max_len: int = 4,                # 短め（語として不自然な長文片を避ける）
    tf_norm: str = "rel",            # "rel": 相対頻度、"log": 1+log(tf)
    agg: str = "avg",                # ※ 単一文書なので実質無関係（互換のため残す）
    smooth_idf: bool = True,         # True → idf = 1.0（単一文書での推奨）
    reject_phrase_like: bool = True,
    require_model: bool = True,
    min_total_count: int = 3,        # 総出現数しきい値（低頻度語を除外）
) -> pd.DataFrame:
    """
    すべての投稿テキストを連結し、**1つの文書**として TF-IDF を計算して上位語を返す関数。
    単一文書のため idf は定数になり（smooth_idf=True なら 1.0）、実質的に **TF ランキング**となります。
    それでも既存の呼び出しと互換にし、列は ['word','score','df','idf'] を返します。

    Returns
    -------
    DataFrame(columns=['word','score','df','idf'])
      - score: tf * idf（単一文書なので idf は定数、tf は相対頻度 or log tf）
      - df:    単一文書内での出現有無（出現語は 1）
      - idf:   smooth_idf=True で 1.0（推奨）
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["word", "score", "df", "idf"])

    col = _detect_text_col(df, text_col)

    tok = tokenizer or create_tokenizer_latest(
        lang=lang,
        use_lemma=use_lemma,
        keep_pos_ja=keep_pos_ja,
        keep_pos_en=keep_pos_en,
        extra_stopwords=extra_stopwords,
        min_len=min_len,
        max_len=max_len,
        allow_download=allow_download,
        require_model=require_model,
    )

    # ---- すべてのテキストを1つの文書として結合（トークン化は各行で行い合算）----
    from collections import Counter
    total_counts = Counter()

    for s in df[col].dropna().astype(str):
        for t in tok(s):
            if reject_phrase_like and _looks_phrase_like_ja(t):
                continue
            if len(t) > max_len:
                continue
            total_counts[t] += 1

    if not total_counts:
        return pd.DataFrame(columns=["word", "score", "df", "idf"])

    # 低頻度語の除外
    allowed_vocab = {w for w, c in total_counts.items() if c >= min_total_count}
    if not allowed_vocab:
        return pd.DataFrame(columns=["word", "score", "df", "idf"])

    # 単一文書の長さ
    L = sum(total_counts[w] for w in allowed_vocab)
    if L == 0:
        return pd.DataFrame(columns=["word", "score", "df", "idf"])

    # ---- 単一文書の TF（相対 or 対数）と IDF（定数）----
    import math
    if smooth_idf:
        idf_val = 1.0       # log((1+1)/(1+1)) + 1 = 1.0
    else:
        # 単一文書で df=1 のため log(1/1)=0 になりがち。互換のため 0.0 のままにする
        idf_val = 0.0

    rows = []
    for w in allowed_vocab:
        tf_raw = total_counts[w]
        if tf_norm == "log":
            tf = 1.0 + math.log(float(tf_raw))
        else:  # "rel"
            tf = tf_raw / float(L)
        score = tf * idf_val
        rows.append((w, score, 1, idf_val))  # df は単一文書なので 1

    # スコアで並べ替え
    rows.sort(key=lambda x: x[1], reverse=True)
    if top_n and top_n > 0:
        rows = rows[:top_n]

    out = pd.DataFrame(rows, columns=["word", "score", "df", "idf"])
    return out
# =========================
# 1) 頻度トップ語での共起ランキング
# =========================
def build_cooccurrence_ranking_from_top_words(
    df: pd.DataFrame,
    *,
    text_col: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    lang: str = "auto",
    allow_download: bool = False,
    # 語彙の作り方（どちらか一方）
    word_ranking_df: Optional[pd.DataFrame] = None,  # columns=['word','count',...]
    top_n_words: int = 50,                           # word_ranking_df 未指定時に使用
    # トークンフィルタ
    max_len: int = 12,
    reject_phrase_like: bool = True,
    # 共起カットオフ/指標
    min_co: int = 2,
    metric: Literal["co_count","pmi","lift","jaccard"] = "co_count",
) -> pd.DataFrame:
    """
    与えたDataFrameから頻度上位語の語彙を作り、文書単位の共起ランキングを返す。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["w1","w2","co_count","support1","support2","pmi","lift","jaccard"])

    col = _detect_text_col(df, text_col)

    # 語彙（上位語）
    if word_ranking_df is not None and not word_ranking_df.empty and "word" in word_ranking_df.columns:
        vocab = set(word_ranking_df["word"].head(top_n_words).tolist())
    else:
        # 内部でランキングを作る（既存の最新版関数を想定）
        if tokenizer is None:
            try:
                tokenizer = create_tokenizer_latest(lang=lang, allow_download=allow_download)  # type: ignore[name-defined]
            except NameError:
                raise RuntimeError("Tokenizer is not provided and create_tokenizer_latest is not available.")
        wr = build_word_ranking_from_posts_latest(  # type: ignore[name-defined]
            df, text_col=col, tokenizer=tokenizer, top_n=max(top_n_words, 200),  # 底上げして選びやすく
            max_len=max_len,  # 文片混入を抑制
        )
        vocab = set(wr["word"].head(top_n_words).tolist())

    if tokenizer is None:
        try:
            tokenizer = create_tokenizer_latest(lang=lang, allow_download=allow_download)  # type: ignore[name-defined]
        except NameError:
            raise RuntimeError("Tokenizer is not provided and create_tokenizer_latest is not available.")

    # 文書→語セット
    docs_terms = [
        _tokenize_to_set(text, tokenizer, vocab=vocab, max_len=max_len, reject_phrase_like=reject_phrase_like)
        for text in df[col].astype(str).tolist()
    ]

    return _build_cooccurrence_core(docs_terms, min_co=min_co, metric=metric)

# =========================
# 2) TF-IDF トップ語での共起ランキング
# =========================
def build_cooccurrence_ranking_from_top_tfidf(
    df: pd.DataFrame,
    *,
    text_col: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    lang: str = "auto",
    allow_download: bool = False,
    # 語彙の作り方（どちらか一方）
    tfidf_ranking_df: Optional[pd.DataFrame] = None,  # columns=['word','score',...]
    top_n_words: int = 50,
    # トークンフィルタ
    max_len: int = 12,
    reject_phrase_like: bool = True,
    # 共起カットオフ/指標
    min_co: int = 2,
    metric: Literal["co_count","pmi","lift","jaccard"] = "co_count",
) -> pd.DataFrame:
    """
    与えたDataFrameから TF-IDF 上位語の語彙を作り、文書単位の共起ランキングを返す。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["w1","w2","co_count","support1","support2","pmi","lift","jaccard"])

    col = _detect_text_col(df, text_col)

    if tfidf_ranking_df is not None and not tfidf_ranking_df.empty and "word" in tfidf_ranking_df.columns:
        vocab = set(tfidf_ranking_df["word"].head(top_n_words).tolist())
    else:
        if tokenizer is None:
            try:
                tokenizer = create_tokenizer_latest(lang=lang, allow_download=allow_download)  # type: ignore[name-defined]
            except NameError:
                raise RuntimeError("Tokenizer is not provided and create_tokenizer_latest is not available.")
        tdf = build_tfidf_ranking_from_posts_latest(  # type: ignore[name-defined]
            df, text_col=col, tokenizer=tokenizer, top_n=max(top_n_words, 200),
            max_len=max_len,
        )
        vocab = set(tdf["word"].head(top_n_words).tolist())

    if tokenizer is None:
        try:
            tokenizer = create_tokenizer_latest(lang=lang, allow_download=allow_download)  # type: ignore[name-defined]
        except NameError:
            raise RuntimeError("Tokenizer is not provided and create_tokenizer_latest is not available.")

    docs_terms = [
        _tokenize_to_set(text, tokenizer, vocab=vocab, max_len=max_len, reject_phrase_like=reject_phrase_like)
        for text in df[col].astype(str).tolist()
    ]

    return _build_cooccurrence_core(docs_terms, min_co=min_co, metric=metric)

def _build_cooccurrence_core(docs_terms, *, min_co=2, metric="co_count"):
    """
    共起ランキングのコア計算（呼び出し元: build_cooccurrence_ranking_from_top_words / _from_top_tfidf）
    docs_terms: List[set[str]] を想定（各ドキュメント中の一意トークン集合）
    戻り値: DataFrame(columns=["w1","w2","co_count","support1","support2","pmi","lift","jaccard"]) を
            metric で降順ソートして返す
    """
    import math
    from collections import Counter
    from itertools import combinations
    import pandas as pd

    cols = ["w1", "w2", "co_count", "support1", "support2", "pmi", "lift", "jaccard"]

    if not docs_terms:
        return pd.DataFrame(columns=cols)

    # 空やNoneを除去しつつ set 化を保証
    docs_sets = []
    for s in docs_terms:
        if not s:
            continue
        docs_sets.append(set(s))

    N = len(docs_sets)
    if N == 0:
        return pd.DataFrame(columns=cols)

    support = Counter()
    pair_cnt = Counter()

    for s in docs_sets:
        for w in s:
            support[w] += 1
        # 同一文書内でのペアを数える（重複カウントなし）
        for a, b in combinations(sorted(s), 2):
            pair_cnt[(a, b)] += 1

    rows = []
    for (w1, w2), c in pair_cnt.items():
        if c < min_co:
            continue
        s1, s2 = support[w1], support[w2]
        lift = (c * N) / (s1 * s2) if s1 and s2 else 0.0
        pmi = math.log2(lift) if lift > 0 else float("-inf")
        jacc = c / (s1 + s2 - c) if (s1 + s2 - c) > 0 else 0.0
        rows.append({
            "w1": w1, "w2": w2, "co_count": c,
            "support1": s1, "support2": s2,
            "pmi": pmi, "lift": lift, "jaccard": jacc
        })

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)

    if metric not in ("co_count", "pmi", "lift", "jaccard"):
        metric = "co_count"

    # 非有限値は後段の描画で除去するが、念のためここでも落としておく
    if metric in ("pmi", "lift", "jaccard"):
        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[metric])

    return df.sort_values(metric, ascending=False).reset_index(drop=True)

def plot_cooccurrence_ranking_bar(
    cooc_df: pd.DataFrame,
    *,
    metric: str = "co_count",          # "co_count" | "pmi" | "lift" | "jaccard"
    top_n: int = 30,
    title: str | None = None,
    use_streamlit: bool = True,
    key: str | None = None,
):
    """
    共起ランキング（w1×w2）を横棒グラフで可視化する。
    入力 DataFrame は build_cooccurrence_ranking_* の出力を想定：
      columns=["w1","w2","co_count","support1","support2","pmi","lift","jaccard"]

    - metric で並び替え
    - PMI/Lift/Jaccard のときは非有限値を除外
    - Streamlit で複数描画する場合は key をユニークに
    戻り値: plotly.graph_objects.Figure or None
    """
    required_cols = {"w1","w2","co_count","support1","support2","pmi","lift","jaccard"}
    if cooc_df is None or cooc_df.empty or not required_cols.issubset(cooc_df.columns):
        try:
            if use_streamlit:
                import streamlit as st
                st.info("共起ランキングの可視化対象データがありません。")
        except Exception:
            pass
        return None

    df = cooc_df.copy()

    # metric 列の前処理（非有限を除外）
    if metric in ("pmi","lift","jaccard"):
        df = df[np.isfinite(df[metric])]
    if df.empty:
        try:
            if use_streamlit:
                import streamlit as st
                st.info("共起ランキング（有効値なし）。")
        except Exception:
            pass
        return None

    # ペア表記（w1 × w2）
    df["pair"] = df["w1"].astype(str) + " × " + df["w2"].astype(str)

    # 並び替え＆上位抽出
    df = df.sort_values(metric, ascending=False)
    if top_n and top_n > 0:
        df = df.head(top_n)
    # 横棒で上位が上になるよう昇順に並び替え
    df = df.sort_values(metric, ascending=True)

    # 軸ラベル
    metric_label = {
        "co_count": "Co-occurrence count",
        "pmi": "PMI (pointwise mutual information)",
        "lift": "Lift",
        "jaccard": "Jaccard",
    }.get(metric, metric)

    # 図
    hover_cols = ["pair","co_count","support1","support2","pmi","lift","jaccard"]
    fig = px.bar(
        df,
        x=metric,
        y="pair",
        orientation="h",
        title=title or f"Co-occurrence ranking ({metric_label})",
        text=metric,
        hover_data=hover_cols,
    )
    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=df["pair"].tolist()),
        xaxis_title=metric_label,
        yaxis_title="Word pair",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    # 表示テキストのフォーマット
    if metric in ("pmi","lift","jaccard"):
        fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
    else:
        fig.update_traces(textposition="outside", cliponaxis=False)

    if use_streamlit:
        import streamlit as st
        if key is None:
            payload = {"metric": metric, "title": title, "rows": df[["pair", metric]].to_dict("records")}
            key = "cooc-" + hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
        st.plotly_chart(fig, use_container_width=True, key=key)

    return fig

def plot_cooccurrence_ranking_both_metrics(
    cooc_df: pd.DataFrame,
    *,
    top_n: int = 30,
    use_streamlit: bool = True,
    base_key: str | None = None,
):
    """
    便利版：頻度ベース(co_count)と関連度ベース(lift)の2種類をまとめて描画する。
    Streamlit 使用時は内部で key を自動生成（base_key を付与すると衝突しにくい）。
    """
    t1 = "共起ランキング（出現回数）"
    t2 = "共起ランキング（Lift）"
    k1 = (base_key + "-count") if base_key else None
    k2 = (base_key + "-lift") if base_key else None

    fig1 = plot_cooccurrence_ranking_bar(
        cooc_df, metric="co_count", top_n=top_n, title=t1, use_streamlit=use_streamlit, key=k1
    )
    fig2 = plot_cooccurrence_ranking_bar(
        cooc_df, metric="lift", top_n=top_n, title=t2, use_streamlit=use_streamlit, key=k2
    )
    return fig1, fig2

# 既存の like/repost 集計ユーティリティを流用（CID基準）
def _terms_agg_counts_by_field(
    es, *, index: str, ids: List[str], field: str,
    op_filter_field: Optional[str] = "commit.operation",
    op_filter_value: Optional[str] = "create",
    chunk_size: int = 5000, request_timeout: int = 60
) -> Dict[str, int]:
    uniq = [i.strip() for i in dict.fromkeys(ids) if isinstance(i, str) and i.strip()]
    out: Dict[str, int] = {}
    if not uniq:
        return out
    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i:i+chunk_size]
        must_filters = [{"terms": {field: chunk}}]
        if op_filter_field and op_filter_value:
            must_filters.append({"term": {op_filter_field: op_filter_value}})
        body = {
            "size": 0,
            "track_total_hits": False,
            "query": {"bool": {"filter": must_filters}},
            "aggs": {"by_id": {"terms": {"field": field, "size": len(chunk), "shard_size": len(chunk)+128}}},
        }
        res = es.search(
            index=index, body=body,
            ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
            request_timeout=request_timeout, request_cache=True
        )
        buckets = (res.get("aggregations") or {}).get("by_id", {}).get("buckets", []) or []
        for id_ in chunk:
            out.setdefault(id_, 0)
        for b in buckets:
            out[b["key"]] = int(b.get("doc_count", 0))
    return out



def _get_reposters_by_cid(
    es, cid: str, *, repost_index: str = "repostindex-*",
    size: int = 10, request_timeout: int = 30
) -> List[Dict]:
    """
    特定の投稿（cid）をリポストしている人（did）を取得（最大 size 件）
    戻り値: [{did, created_at}]
    """
    body = {
        "size": size,
        "_source": ["did", "commit.record.createdAt", "commit.operation", "commit.record.subject.cid"],
        "sort": [{"commit.record.createdAt": "desc"}],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.operation": "create"}},
                    {"term": {"commit.record.subject.cid": cid}}
                ]
            }
        }
    }
    
    res = es.search(
        index=repost_index, body=body,
        ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
        request_timeout=request_timeout, request_cache=True
    )
    
    hits = (res.get("hits") or {}).get("hits") or []
    out = []
    for h in hits:
        src = h.get("_source") or {}
        out.append({
            "did": src.get("did"),
            "created_at": (src.get("commit") or {}).get("record", {}).get("createdAt")
        })
    # 重複 did を後勝ちでユニーク化
    uniq = {}
    for r in out:
        d = r.get("did")
        if d:
            uniq[d] = r
    return list(uniq.values())

def _get_recent_posts_by_user(
    es, did: str, *, post_index: str = "postindex-*",
    per_user_limit: int = 20, request_timeout: int = 30
) -> List[Dict]:
    """
    ユーザー(did)の直近投稿を取得（cid/rkey/createdAt/テキスト）
    戻り値: [{did, cid, rkey, created_at, text}]
    """
    body = {
        "size": per_user_limit,
        "_source": ["did", "commit.cid", "commit.rkey", "commit.record.createdAt", "commit.record.text"],
        "sort": [{"commit.record.createdAt": "desc"}],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"did": did}},
                    {"exists": {"field": "commit.cid"}}
                ],
                "must_not": [
                    {"exists": {"field": "commit.record.reply"}}  # 返信は除外（任意）
                ]
            }
        }
    }
    res = es.search(
        index=post_index, body=body,
        ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
        request_timeout=request_timeout, request_cache=True
    )
    hits = (res.get("hits") or {}).get("hits") or []
    rows = []
    for h in hits:
        src = h.get("_source") or {}
        commit = src.get("commit") or {}
        record = commit.get("record") or {}
        cid = commit.get("cid")
        rkey = commit.get("rkey")
        if not cid:
            continue
        rows.append({
            "did": did,
            "cid": str(cid),
            "rkey": rkey,
            "created_at": record.get("createdAt"),
            "text": record.get("text"),
        })
    return rows


from typing import Optional, Tuple, List, Dict, Set
import pandas as pd
from collections import Counter

def _map_cids_to_authors(
    es,
    cids: List[str],
    *,
    post_index: str = "postindex-*",
    chunk_size: int = 2000,
    request_timeout: int = 30,
) -> Dict[str, str]:
    """
    postindex から commit.cid -> did（投稿者） のマップを一括取得
    """
    uniq = [c.strip() for c in dict.fromkeys(cids) if isinstance(c, str) and c.strip()]
    out: Dict[str, str] = {}
    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i : i + chunk_size]
        body = {
            "size": len(chunk),
            "_source": ["did", "commit.cid"],
            "track_total_hits": False,
            "query": {"bool": {"filter": [{"terms": {"commit.cid": chunk}}]}},
        }
        res = es.search(
            index=post_index,
            body=body,
            ignore_unavailable=True,
            allow_no_indices=True,
            expand_wildcards="open",
            request_timeout=request_timeout,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        for h in hits:
            src = h.get("_source") or {}
            did = src.get("did")
            cid = (src.get("commit") or {}).get("cid")
            if isinstance(cid, str) and isinstance(did, str) and cid and did and cid not in out:
                out[cid] = did
    return out

def _bsky_profile_url(identifier: str | None) -> str | None:
    return f"https://bsky.app/profile/{identifier}" if isinstance(identifier, str) and identifier else None

def _api_get_reposters_by_uri(bsky: BlueskyClient, uri: str, cid: str | None = None, limit_each: int = 200) -> list[str]:
    # DID のリスト
    return list(bsky.get_reposted_by(uri=uri, cid=cid, limit_total=limit_each))

def _api_get_recent_posts_by_user(bsky: BlueskyClient, did_or_handle: str, per_user_limit: int = 50) -> list[dict]:
    return list(bsky.get_author_feed(actor=did_or_handle, limit_total=per_user_limit))

def _api_map_uris_to_authors(bsky: BlueskyClient, uris: list[str]) -> dict[str, str]:
    out = {}
    meta = bsky.get_posts(uris)
    for u, m in meta.items():
        if m.get("author"):
            out[u] = m["author"]
    return out

def build_repost_network_snowball(
    es,
    *,
    post_df: Optional[pd.DataFrame] = None,
    seed_cid: Optional[str] = None,
    post_index: str = "postindex-*",
    repost_index: str = "repostindex-*",
    # スノーボール探索パラメータ
    depth: int = 3,
    top_reposters_per_post: int = 10,
    per_user_post_limit: int = 20,
    next_post_min_reposts: int = 2,
    # 安全装置
    max_users: int = 500,
    max_edges: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ユーザーのみをノード、リポスト関係（AがBの投稿をリポスト）をエッジにしたネットワーク。
    追加: 起点投稿の作者（seed_author）を強調するため is_seed 列と特別な label（先頭に★）を付与。
    返り値:
      nodes_df: [id, type='user', label, url, in_reposts, out_reposts, repost_count, is_seed]
      edges_df: [src, dst, kind='repost', weight]
    """
    # --- 起点の決定（seed_cid） ---
    if seed_cid is None:
        if post_df is not None and "cid" in post_df.columns:
            seed_row = (post_df.sort_values("repost_count", ascending=False).head(1)
                        if "repost_count" in post_df.columns else post_df.head(1))
            if seed_row.empty or not str(seed_row.iloc[0]["cid"]).strip():
                raise ValueError("seed_cid が指定されておらず、post_df からも取得できません。")
            seed_cid = str(seed_row.iloc[0]["cid"]).strip()
        else:
            raise ValueError("seed_cid を明示するか、cid を含む post_df を渡してください。")

    # --- ユーティリティ ---
    def _get_reposters_by_cid(cid: str, size: int = 10) -> List[str]:
        body = {
            "size": size,
            "_source": ["did", "commit.operation", "commit.record.subject.cid"],
            "sort": [{"commit.record.createdAt": "desc"}],
            "query": {"bool": {"filter": [
                {"term": {"commit.operation": "create"}},
                {"term": {"commit.record.subject.cid": cid}}
            ]}}
        }
        res = es.search(
            index=repost_index, body=body,
            ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
            request_timeout=30,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        dids: List[str] = []
        for h in hits:
            src = h.get("_source") or {}
            d = src.get("did")
            if isinstance(d, str) and d:
                dids.append(d)
        # 重複除去（後勝ち）
        out = {}
        for d in dids: out[d] = True
        return list(out.keys())

    def _get_recent_posts_by_user(did: str, limit: int = 20) -> List[str]:
        body = {
            "size": limit,
            "_source": ["commit.cid", "commit.record.createdAt"],
            "sort": [{"commit.record.createdAt": "desc"}],
            "query": {"bool": {
                "filter": [
                    {"term": {"did": did}},
                    {"exists": {"field": "commit.cid"}}
                ],
                "must_not": [{"exists": {"field": "commit.record.reply"}}]
            }},
        }
        res = es.search(
            index=post_index, body=body,
            ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
            request_timeout=30,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        cids = []
        for h in hits:
            src = h.get("_source") or {}
            cid = (src.get("commit") or {}).get("cid")
            if isinstance(cid, str) and cid:
                cids.append(cid)
        return cids

    def _count_reposts_for_cids(cids: List[str]) -> Dict[str, int]:
        if not cids: return {}
        uniq = [c.strip() for c in dict.fromkeys(cids) if c]
        body = {
            "size": 0,
            "track_total_hits": False,
            "query": {"bool": {"filter": [
                {"term": {"commit.operation": "create"}},
                {"terms": {"commit.record.subject.cid": uniq}}
            ]}},
            "aggs": {"by_cid": {"terms": {"field": "commit.record.subject.cid", "size": len(uniq), "shard_size": len(uniq)+128}}},
        }
        res = es.search(
            index=repost_index, body=body,
            ignore_unavailable=True, allow_no_indices=True, expand_wildcards="open",
            request_timeout=60,
        )
        buckets = (res.get("aggregations") or {}).get("by_cid", {}).get("buckets", []) or []
        out = {c: 0 for c in uniq}
        for b in buckets:
            out[b["key"]] = int(b.get("doc_count", 0))
        return out

    # --- シード投稿の作者 did を取得 ---
    cid_to_author = _map_cids_to_authors(es, [seed_cid], post_index=post_index)
    seed_author = cid_to_author.get(seed_cid)
    if not seed_author:
        raise ValueError(f"seed_cid={seed_cid} の投稿者を特定できませんでした。")

    # --- ネットワーク構築 ---
    visited_users: Set[str] = set()
    user_frontier: Set[str] = set()

    # reposters of seed_cid → seed_author へのエッジ
    edges_ctr: Counter[tuple[str, str]] = Counter()
    reposters = _get_reposters_by_cid(seed_cid, size=top_reposters_per_post)
    for rep in reposters:
        if rep and rep != seed_author:
            edges_ctr[(rep, seed_author)] += 1
            user_frontier.add(rep)

    # BFS スノーボール
    lvl = 0
    while lvl < depth and user_frontier:
        if len(visited_users) + len(user_frontier) > max_users:
            user_frontier = set(list(user_frontier)[: max(0, max_users - len(visited_users))])

        next_frontier: Set[str] = set()
        for u in list(user_frontier):
            if u in visited_users:
                continue
            visited_users.add(u)

            # u の直近投稿
            u_post_cids = _get_recent_posts_by_user(u, limit=per_user_post_limit)
            if not u_post_cids:
                continue

            # それらのリポスト数
            counts = _count_reposts_for_cids(u_post_cids)
            # しきい値以上の投稿のリポスターを取得
            cids_focus = [c for c, k in counts.items() if k >= next_post_min_reposts]
            if not cids_focus:
                continue

            for c in cids_focus:
                reps2 = _get_reposters_by_cid(c, size=top_reposters_per_post)
                for r2 in reps2:
                    if not r2 or r2 == u:
                        continue
                    edges_ctr[(r2, u)] += 1
                    next_frontier.add(r2)
                    if len(edges_ctr) >= max_edges:
                        break
                if len(edges_ctr) >= max_edges:
                    break
            if len(edges_ctr) >= max_edges:
                break

        user_frontier = next_frontier
        lvl += 1
        if len(edges_ctr) >= max_edges:
            break

    # --- DataFrame 化（集約） ---
    edges_rows = [{"src": s, "dst": t, "kind": "repost", "weight": int(w)} for (s, t), w in edges_ctr.items()]
    edges_df = pd.DataFrame(edges_rows, columns=["src", "dst", "kind", "weight"])
    if edges_df.empty:
        base_nodes = set([seed_author]) | set(reposters)
        nodes_df = pd.DataFrame({"id": list(base_nodes), "type": "user"})
        nodes_df["url"] = nodes_df["id"].apply(_bsky_profile_url)
        # ★ 起点作者を強調
        nodes_df["is_seed"] = nodes_df["id"].eq(seed_author)
        nodes_df["label"] = np.where(nodes_df["is_seed"], "★ " + nodes_df["id"].astype(str), nodes_df["id"].astype(str))
        nodes_df["in_reposts"] = 0
        nodes_df["out_reposts"] = 0
        nodes_df["repost_count"] = nodes_df["in_reposts"]
        return nodes_df, edges_df

    # ノード集合（ユーザーのみ）
    node_ids = sorted(set(edges_df["src"]).union(set(edges_df["dst"])))
    nodes_df = pd.DataFrame({"id": node_ids})
    nodes_df["type"] = "user"
    nodes_df["url"]  = nodes_df["id"].apply(_bsky_profile_url)
    # ★ 起点作者を強調
    nodes_df["is_seed"] = nodes_df["id"].eq(seed_author)
    nodes_df["label"]   = np.where(nodes_df["is_seed"], "★ " + nodes_df["id"].astype(str), nodes_df["id"].astype(str))

    # イン/アウトの重み合計
    in_sum = edges_df.groupby("dst")["weight"].sum()
    out_sum = edges_df.groupby("src")["weight"].sum()
    nodes_df["in_reposts"]  = nodes_df["id"].map(in_sum).fillna(0).astype(int)
    nodes_df["out_reposts"] = nodes_df["id"].map(out_sum).fillna(0).astype(int)
    nodes_df["repost_count"] = nodes_df["in_reposts"]

    # --- 最大連結成分に絞る ---
    try:
        import networkx as nx
        G = nx.Graph()
        for r in nodes_df.itertuples(index=False):
            G.add_node(r.id)
        for e in edges_df.itertuples(index=False):
            G.add_edge(e.src, e.dst)
        comps = list(nx.connected_components(G))
        if comps:
            giant = max(comps, key=len)
            nodes_df = nodes_df[nodes_df["id"].isin(giant)].reset_index(drop=True)
            edges_df = edges_df[edges_df["src"].isin(giant) & edges_df["dst"].isin(giant)].reset_index(drop=True)
    except Exception:
        pass

    return nodes_df, edges_df

import math
import hashlib, json
from typing import Optional, Tuple, Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def _scale_series_to_range(
    s: pd.Series,
    out_min: float = 6.0,
    out_max: float = 18.0,
) -> pd.Series:
    """数値Seriesを[min,max]に線形スケーリング（定数/NaN多発時のフォールバック付き）"""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([ (out_min + out_max) / 2 ] * len(s), index=s.index)
    vmin, vmax = s.min(), s.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return pd.Series([ (out_min + out_max) / 2 ] * len(s), index=s.index)
    return out_min + (s - vmin) * (out_max - out_min) / (vmax - vmin)


# ← これで NaN サイズを確実に回避します（欠損は中間サイズで埋める）

def plot_network_from_dfs(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    node_id_col: str = "id",
    node_type_col: str | None = "type",
    node_label_col: str | None = "label",
    node_size_col: str | None = "repost_count",
    edge_src_col: str = "src",
    edge_dst_col: str = "dst",
    edge_weight_col: str | None = "weight",
    layout: str = "spring",
    seed: int = 42,
    node_size_range: tuple[float, float] = (6.0, 18.0),
    show_text: bool = None,
    title: str | None = "Repost Network",
    giant_component: bool = False,
    max_nodes_render: int = 2000,
    use_streamlit: bool = True,
    key: str | None = None,
    # クリックでプロフィールへ
    click_to_open_profile: bool = True,
    url_col: str = "url",
    open_auto: bool = True,
    plot_height: int = 720,
    # ★ 起点ノードの強調表示
    seed_flag_col: str = "is_seed",
    seed_color: str = "#E74C3C",        # 赤系
    default_node_color: str = "#1f77b4",# 青系
):
    import math, hashlib, json, uuid
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import networkx as nx

    def _scale_series_to_range(s: pd.Series, out_min: float, out_max: float) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        v = s.dropna()
        if v.empty or math.isclose(v.min(), v.max()):
            return pd.Series([(out_min + out_max) / 2] * len(s), index=s.index)
        scaled = out_min + (s - v.min()) * (out_max - out_min) / (v.max() - v.min())
        mid = (out_min + out_max) / 2
        return scaled.replace([np.inf, -np.inf], np.nan).fillna(mid).clip(out_min, out_max)

    def _bsky_profile_url(identifier: str | None) -> str | None:
        return f"https://bsky.app/profile/{identifier}" if isinstance(identifier, str) and identifier else None

    def _short_url(u: str | None, maxlen: int = 42) -> str:
        if not isinstance(u, str) or not u:
            return ""
        disp = u.replace("https://", "").replace("http://", "")
        return disp if len(disp) <= maxlen else disp[: maxlen - 1] + "…"

    # 入力チェック
    if node_id_col not in nodes_df.columns:
        raise ValueError(f"nodes_df に '{node_id_col}' 列がありません。")
    for c in (edge_src_col, edge_dst_col):
        if c not in edges_df.columns:
            raise ValueError(f"edges_df に '{c}' 列がありません。")

    ndf = nodes_df.copy()
    edf = edges_df.copy()

    # URL / SEED マップ
    if url_col in ndf.columns:
        id2url = dict(zip(ndf[node_id_col].astype(str), ndf[url_col].astype(str)))
    else:
        id2url = dict(zip(ndf[node_id_col].astype(str), ndf[node_id_col].astype(str).map(_bsky_profile_url)))
    id2seed = dict(zip(ndf[node_id_col].astype(str), ndf[seed_flag_col].astype(bool))) if seed_flag_col in ndf.columns else {}

    # 大規模ガード
    if len(ndf) > max_nodes_render:
        deg = pd.concat([edf[edge_src_col], edf[edge_dst_col]]).value_counts()
        keep = set(deg.head(max_nodes_render).index.astype(str))
        ndf = ndf[ndf[node_id_col].astype(str).isin(keep)]
        edf = edf[edf[edge_src_col].astype(str).isin(keep) & edf[edge_dst_col].astype(str).isin(keep)]

    # NetworkX 構築
    G = nx.Graph()
    for r in ndf.itertuples(index=False):
        nid = getattr(r, node_id_col)
        if nid is None:
            continue
        attrs = {}
        if node_type_col and node_type_col in ndf.columns: attrs["type"] = getattr(r, node_type_col)
        if node_label_col and node_label_col in ndf.columns: attrs["label"] = getattr(r, node_label_col)
        if node_size_col and node_size_col in ndf.columns: attrs["size_attr"] = getattr(r, node_size_col)
        G.add_node(str(nid), **attrs)
    for r in edf.itertuples(index=False):
        u = getattr(r, edge_src_col); v = getattr(r, edge_dst_col)
        if u is None or v is None: continue
        w = None
        if edge_weight_col and edge_weight_col in edf.columns:
            try: w = float(getattr(r, edge_weight_col))
            except Exception: w = 1.0
        G.add_edge(str(u), str(v), weight=(w if w is not None else 1.0))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        fig = go.Figure()
        if use_streamlit:
            import streamlit as st
            st.info("ネットワークに描画可能なノード/エッジがありません。")
        return fig

    if giant_component:
        comps = list(nx.connected_components(G))
        if comps:
            G = G.subgraph(max(comps, key=len)).copy()

    # レイアウト
    if layout == "spring": pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai": pos = nx.kamada_kawai_layout(G)
    elif layout == "fr": pos = nx.fruchterman_reingold_layout(G, seed=seed)
    elif layout == "circular": pos = nx.circular_layout(G)
    else: pos = nx.spring_layout(G, seed=seed)

    # 可視化データ作成
    nodes = list(G.nodes())
    xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
    types  = [G.nodes[n].get("type", None) for n in nodes]
    labels = [G.nodes[n].get("label", n) for n in nodes]
    size_attr = [G.nodes[n].get("size_attr", np.nan) for n in nodes]
    sizes = _scale_series_to_range(pd.Series(size_attr, index=nodes), *node_size_range).astype(float).tolist()

    # エッジ線
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none",
                             name="edges", opacity=0.5))

    # ノードレイヤ（hover に短縮 URL / customdata に [id,url]）
    uniq_types = sorted({t for t in types if t is not None}) if (node_type_col and node_type_col in ndf.columns) else [None]
    for t in uniq_types:
        idxs = [i for i, tv in enumerate(types) if (tv == t) or (t is None)]
        if not idxs: continue

        hovertexts, customs, colors = [], [], []
        for i in idxs:
            nid = str(nodes[i])
            url = id2url.get(nid) or _bsky_profile_url(nid)
            is_seed = bool(id2seed.get(nid, False))
            tag = "（SEED）" if is_seed else ""
            hovertexts.append(
                f"id: {nid}{tag}"
                + (f"<br>type: {types[i]}" if types[i] is not None else "")
                + f"<br>size: {sizes[i]:.1f}"
                + f"<br>url: {_short_url(url, 48)}"
            )
            customs.append([nid, url])
            colors.append(seed_color if is_seed else default_node_color)

        fig.add_trace(go.Scatter(
            x=[xs[i] for i in idxs],
            y=[ys[i] for i in idxs],
            mode="markers+text" if show_text else "markers",
            text=[labels[i] for i in idxs] if show_text else None,
            textposition="top center",
            marker=dict(size=[sizes[i] for i in idxs],
                        color=colors,
                        line=dict(width=0.5),
                        symbol=("square" if t == "post" else "circle")),
            hovertext=hovertexts,
            hoverinfo="none",
            hovertemplate="%{hovertext}<extra></extra>",
            name=("posts" if t == "post" else ("users" if t == "user" else "nodes")),
            customdata=customs,
        ))
        if t is None: break

    fig.update_layout(
        title=title or "Network",
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=plot_height,
    )

    # --- Streamlit 表示（フロントエンドでクリック→新規タブ） ---
    if use_streamlit:
        import streamlit as st
        div_id = f"plot-{uuid.uuid4().hex[:8]}"
        html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id=div_id)
        js = f"""
        <style>
          /* クリック感 */
          #{div_id} .scatterlayer .trace .points path {{ cursor: pointer; }}
        </style>
        <script>
        (function() {{
          const wait = () => {{
            const gd = document.getElementById("{div_id}");
            if (!gd || !gd.on) return setTimeout(wait, 50);
            gd.on('plotly_click', function(ev) {{
              try {{
                const pt = ev?.points?.[0];
                let cd = pt?.customdata;
                let url = Array.isArray(cd) ? (cd.length>=2 ? cd[1] : cd[0]) : cd;
                if (url && /^https?:\\/\\//.test(url)) {{
                  window.open(url, '_blank', 'noopener,noreferrer');
                }}
              }} catch(e) {{ console.error(e); }}
            }});
          }};
          wait();
        }})();
        </script>
        """
        st.components.v1.html(html + js, height=plot_height)

    return fig

from typing import Iterable, Optional, Tuple, Set, Dict
import pandas as pd

def _bsky_profile_url(identifier: Optional[str]) -> Optional[str]:
    return f"https://bsky.app/profile/{identifier}" if isinstance(identifier, str) and identifier else None

def merge_networks_by_common_users(
    nodes_a: pd.DataFrame,
    edges_a: pd.DataFrame,
    nodes_b: pd.DataFrame,
    edges_b: pd.DataFrame,
    *,
    node_id_col: str = "id",
    edge_src_col: str = "src",
    edge_dst_col: str = "dst",
    edge_weight_col: Optional[str] = "weight",
    seed_ids: Optional[Iterable[str]] = None,  # 起点（複数OK）
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    2つのユーザネットワーク（A/B）を「共通ユーザ（橋渡し）」で結合する。

    戻り値:
      merged_nodes_df, merged_edges_df, bridge_ids
        - merged_nodes_df: 列 [id, label, url, source('A'|'B'|'A|B'), is_bridge(bool), is_seed(bool), ...]
        - merged_edges_df: 列 [src, dst, weight, source('A'|'B'|'A|B'), kind]
        - bridge_ids: AとBに共通して含まれるユーザid集合
    """
    # --- 前処理: 列の存在チェック ---
    for df, name in [(nodes_a, "nodes_a"), (nodes_b, "nodes_b")]:
        if node_id_col not in df.columns:
            raise ValueError(f"{name} に列 '{node_id_col}' がありません。")
    for df, name in [(edges_a, "edges_a"), (edges_b, "edges_b")]:
        for c in (edge_src_col, edge_dst_col):
            if c not in df.columns:
                raise ValueError(f"{name} に列 '{c}' がありません。")

    # --- 文字列ID化 ---
    A_nodes = nodes_a.copy()
    B_nodes = nodes_b.copy()
    A_nodes[node_id_col] = A_nodes[node_id_col].astype(str)
    B_nodes[node_id_col] = B_nodes[node_id_col].astype(str)

    A_edges = edges_a.copy()
    B_edges = edges_b.copy()
    A_edges[edge_src_col] = A_edges[edge_src_col].astype(str)
    A_edges[edge_dst_col] = A_edges[edge_dst_col].astype(str)
    B_edges[edge_src_col] = B_edges[edge_src_col].astype(str)
    B_edges[edge_dst_col] = B_edges[edge_dst_col].astype(str)

    # --- URL 準備 ---
    if "url" not in A_nodes.columns:
        A_nodes["url"] = A_nodes[node_id_col].map(_bsky_profile_url)
    if "url" not in B_nodes.columns:
        B_nodes["url"] = B_nodes[node_id_col].map(_bsky_profile_url)

    # --- ラベル列 ---
    if "label" not in A_nodes.columns:
        A_nodes["label"] = A_nodes[node_id_col]
    if "label" not in B_nodes.columns:
        B_nodes["label"] = B_nodes[node_id_col]

    # --- 共通ユーザ（橋渡し） ---
    ids_a = set(A_nodes[node_id_col].unique().tolist())
    ids_b = set(B_nodes[node_id_col].unique().tolist())
    bridge_ids: Set[str] = ids_a & ids_b

    # --- nodes マージ（A優先で基本属性を取り、Bで欠損を補完） ---
    NA = A_nodes[[node_id_col, "label", "url"]].drop_duplicates().copy()
    NB = B_nodes[[node_id_col, "label", "url"]].drop_duplicates().copy()

    merged_nodes = pd.concat([NA.assign(source="A"), NB.assign(source="B")], ignore_index=True)
    # 同じ id が A/B 両方にある場合にまとめる
    merged_nodes = (merged_nodes
                    .sort_values(["id", "source"])  # A→B の順（任意）
                    .drop_duplicates(subset=[node_id_col], keep="first")
                    .reset_index(drop=True))
    # source を 'A|B' に再付与（橋渡し）
    merged_nodes["source"] = merged_nodes[node_id_col].apply(
        lambda x: ("A|B" if (x in bridge_ids) else ("A" if x in ids_a else "B"))
    )
    # フラグ
    seeds: Set[str] = set([str(s) for s in seed_ids]) if seed_ids is not None else set()
    merged_nodes["is_bridge"] = merged_nodes[node_id_col].isin(bridge_ids)
    merged_nodes["is_seed"] = merged_nodes[node_id_col].isin(seeds)

    # --- edges マージ（ソースを区別しつつ結合、重みは加算） ---
    def _prep_edges(E: pd.DataFrame, source_tag: str) -> pd.DataFrame:
        out = E[[edge_src_col, edge_dst_col] + ([edge_weight_col] if edge_weight_col and edge_weight_col in E.columns else [])].copy()
        if edge_weight_col not in out.columns:
            out[edge_weight_col] = 1.0
        out = out.rename(columns={edge_src_col: "src", edge_dst_col: "dst", edge_weight_col: "weight"})
        out["kind"] = "edge"
        out["source"] = source_tag
        return out

    EA = _prep_edges(A_edges, "A")
    EB = _prep_edges(B_edges, "B")
    merged_edges = pd.concat([EA, EB], ignore_index=True)

    # 重複ペアは重み加算、source は 'A|B' に（どちらにも存在する場合）
    grp = merged_edges.groupby(["src", "dst"], as_index=False).agg(
        weight=("weight", "sum"),
        kind=("kind", "first"),
        source=("source", lambda s: "A|B" if set(s) == {"A", "B"} or len(set(s)) > 1 else list(s)[0]),
    )
    merged_edges = grp

    # ノードに存在しないIDのエッジは除外
    valid_ids = set(merged_nodes[node_id_col])
    merged_edges = merged_edges[merged_edges["src"].isin(valid_ids) & merged_edges["dst"].isin(valid_ids)].reset_index(drop=True)

    # 表示順の整形
    merged_nodes = merged_nodes[[node_id_col, "label", "url", "source", "is_bridge", "is_seed"]]
    merged_nodes = merged_nodes.rename(columns={node_id_col: "id"})

    return merged_nodes, merged_edges, bridge_ids

def plot_merged_network_with_highlights(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    node_id_col: str = "id",
    node_label_col: str = "label",
    url_col: str = "url",
    edge_src_col: str = "src",
    edge_dst_col: str = "dst",
    edge_weight_col: str = "weight",
    layout: str = "spring",
    seed: int = 42,
    node_size_col: Optional[str] = None,       # 例: 'repost_count'。無ければ次数で代替
    node_size_range: tuple[float, float] = (8.0, 20.0),
    use_streamlit: bool = True,
    title: Optional[str] = "Merged network (bridged by common users)",
    key: Optional[str] = None,
):
    """
    結合ネットワークを可視化。
      - 橋渡しノード: 赤 (#d62728)
      - 起点ノード  : 橙 (#ff7f0e)
      - A由来ノード : 青 (#1f77b4)
      - B由来ノード : 緑 (#2ca02c)
      - A|B         : 橋渡し扱い（赤を優先表示）

    クリックでプロフィールを新規タブで開く（url列があれば使用、無ければ DID から生成）。
    橋渡しが1つも無い場合は Streamlit に警告を表示。
    """
    import math, json, hashlib, uuid
    import numpy as np
    import plotly.graph_objects as go
    import networkx as nx

    def _scale_series_to_range(s: pd.Series, out_min: float, out_max: float) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        v = s.dropna()
        if v.empty or math.isclose(float(v.min()), float(v.max())):
            return pd.Series([(out_min + out_max) / 2] * len(s), index=s.index)
        scaled = out_min + (s - v.min()) * (out_max - out_min) / (v.max() - v.min())
        mid = (out_min + out_max) / 2
        return scaled.replace([np.inf, -np.inf], np.nan).fillna(mid).clip(out_min, out_max)

    def _short_url(u: Optional[str], maxlen: int = 48) -> str:
        if not isinstance(u, str) or not u:
            return ""
        disp = u.replace("https://", "").replace("http://", "")
        return disp if len(disp) <= maxlen else disp[: maxlen - 1] + "…"

    # --- 入力確認 ---
    need_cols_nodes = {node_id_col, node_label_col, "source", "is_bridge", "is_seed"}
    if not need_cols_nodes.issubset(nodes_df.columns):
        raise ValueError(f"nodes_df に必要列が不足: {need_cols_nodes - set(nodes_df.columns)}")

    for c in (edge_src_col, edge_dst_col):
        if c not in edges_df.columns:
            raise ValueError(f"edges_df に '{c}' 列がありません。")

    ndf = nodes_df.copy()
    edf = edges_df.copy()

    # URL 補完
    if url_col not in ndf.columns:
        ndf[url_col] = ndf[node_id_col].map(_bsky_profile_url)

    # クリック用に id→url マップ
    id2url: Dict[str, str] = dict(zip(ndf[node_id_col].astype(str), ndf[url_col].astype(str)))

    # --- グラフ構築 ---
    G = nx.Graph()
    for r in ndf.itertuples(index=False):
        nid = getattr(r, node_id_col)
        if nid is None: continue
        G.add_node(
            str(nid),
            label=getattr(r, node_label_col),
            source=getattr(r, "source"),
            is_bridge=bool(getattr(r, "is_bridge")),
            is_seed=bool(getattr(r, "is_seed")),
        )
    for r in edf.itertuples(index=False):
        u = getattr(r, edge_src_col); v = getattr(r, edge_dst_col)
        if u is None or v is None: continue
        w = float(getattr(r, edge_weight_col)) if (edge_weight_col in edf.columns) else 1.0
        G.add_edge(str(u), str(v), weight=w)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        fig = go.Figure()
        if use_streamlit:
            import streamlit as st
            st.info("ネットワークに描画可能なノード/エッジがありません。")
        return fig

    # 橋渡しノード警告
    has_bridge = any(G.nodes[n].get("is_bridge", False) for n in G.nodes())
    if use_streamlit and not has_bridge:
        try:
            import streamlit as st
            st.warning("橋渡しとなるノードが見つかりませんでした。")
        except Exception:
            pass

    # レイアウト
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(G, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    nodes = list(G.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]

    # サイズ：指定列 or 次数
    if node_size_col and node_size_col in ndf.columns:
        size_map = dict(zip(ndf[node_id_col].astype(str), pd.to_numeric(ndf[node_size_col], errors="coerce")))
        raw_sizes = [size_map.get(n, np.nan) for n in nodes]
    else:
        deg = dict(G.degree())
        raw_sizes = [deg.get(n, 0.0) for n in nodes]
    sizes = _scale_series_to_range(pd.Series(raw_sizes, index=nodes), *node_size_range).astype(float).tolist()

    # エッジ座標
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none",
                             name="edges", opacity=0.45))

    # 色割り当て
    COLOR_BRIDGE = "#d62728"   # 赤
    COLOR_SEED   = "#ff7f0e"   # 橙
    COLOR_A      = "#1f77b4"   # 青
    COLOR_B      = "#2ca02c"   # 緑
    COLOR_OTHER  = "#7f7f7f"

    # レイヤ別に描画（橋渡し＞起点＞A＞B の順で上書きされないよう個別トレース）
    def _pick_color(n: str) -> str:
        nd = G.nodes[n]
        if nd.get("is_bridge"): return COLOR_BRIDGE
        if nd.get("is_seed"):   return COLOR_SEED
        src = nd.get("source")
        if src == "A": return COLOR_A
        if src == "B": return COLOR_B
        if src == "A|B": return COLOR_BRIDGE  # 念のため
        return COLOR_OTHER

    # hover / customdata
    hovertexts = []
    customdata = []
    labels = []
    for i, n in enumerate(nodes):
        lab = G.nodes[n].get("label", n)
        src = G.nodes[n].get("source")
        is_b = G.nodes[n].get("is_bridge")
        is_s = G.nodes[n].get("is_seed")
        url = id2url.get(n) or _bsky_profile_url(n)
        labels.append(lab)
        hovertexts.append(
            f"id: {n}"
            f"<br>label: {lab}"
            f"<br>source: {src}"
            f"<br>bridge: {is_b}, seed: {is_s}"
            f"<br>url: {_short_url(url)}"
        )
        customdata.append([n, url])

    # トレースを1本にまとめ、マーカー色を個別指定
    colors = [_pick_color(n) for n in nodes]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        # text=labels,
        textposition="top center",
        marker=dict(size=sizes, line=dict(width=0.5), color=colors),
        hovertext=hovertexts, hoverinfo="none", hovertemplate="%{hovertext}<extra></extra>",
        name="nodes",
        customdata=customdata,
    ))

    fig.update_layout(
        title=title or "Merged Network",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )

    # --- Streamlit 描画（ノードクリックで新規タブを開く） ---
    if use_streamlit:
        import streamlit as st
        div_id = f"plot-{uuid.uuid4().hex[:8]}"
        html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id=div_id)
        js = f"""
        <script>
        (function() {{
          const wait = () => {{
            const gd = document.getElementById("{div_id}");
            if (!gd || !gd.on) {{ return setTimeout(wait, 50); }}
            gd.on('plotly_click', function(ev) {{
              try {{
                const pt = ev.points && ev.points[0];
                let cd = pt && pt.customdata;
                let url = null;
                if (Array.isArray(cd)) {{
                  url = cd.length >= 2 ? cd[1] : cd[0];
                }} else {{
                  url = cd;
                }}
                if (url) window.open(url, '_blank');
              }} catch (e) {{ console.error(e); }}
            }});
          }};
          wait();
        }})();
        </script>
        """
        # ヘルプ凡例
        st.caption("色: 🔴橋渡し / 🟧起点 / 🟦A / 🟩B")
        if not has_bridge:
            st.warning("橋渡しとなるノードが見つかりませんでした。")
        st.components.v1.html(html + js, height=720)

    return fig

def run_spin_with_posts_df_a(es, posts_df_a, start_dt, end_dt):
    """
    research_matae で作った posts_df_a を使って
    seed_account を決め、その seed で SPIN を回す。
    """

    log_container = st.container()

    def spin_logger(msg: str) -> None:
        # 時刻付きで表示
        ts = datetime.now().strftime("%H:%M:%S")
        log_container.write(f"[{ts}] {msg}")
        
    # 1. DataFrame から seed_accounts を決める
    seed_accounts = select_seed_accounts_from_df(
        posts_df_a,
        author_field="did",           # posts_df_a の列名に合わせて変更
        repost_count_field="repost_count",
        min_reposts_for_seed=0,
        top_n_seed=10,                 # まずは1アカウントだけ
    )

    # 2. SPIN を実行（seed_accounts_override に渡すのがポイント）
    result = run_spin_snowball_for_bluesky(
        es,
        post_index="postindex-*",
        repost_index="repostindex-*",
        start_day=start_dt,
        end_day=end_dt,
        text_query_for_seed=None,         # seed は DF ベースなので不要
        min_reposts_for_seed=0,           # override するのでここも実質無視される
        top_n_seed=len(seed_accounts),
        number_of_publications=10,
        number_of_recursions=5,
        require_reposts_min=1,
        seed=42,
        seed_accounts_override=seed_accounts,  # ★ここで上書き
    )

    return result

import pandas as pd

def build_spin_network_from_sampled_edges(
    seed_accounts: list[str],
    sampled_edges: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    SPIN の出力（seed_accounts, sampled_edges）から、
    plot_network_from_dfs で扱えるユーザネットワークを構成する。

    戻り値:
      nodes_df: [id, type, label, url, in_reposts, out_reposts, repost_count, is_seed]
      edges_df: [src, dst, kind, weight]
    """
    # 文字列に統一
    seed_accounts = [str(a) for a in seed_accounts]

    # --- エッジが 0 件の場合の安全なフォールバック ---
    if not sampled_edges:
        nodes_df = pd.DataFrame({"id": seed_accounts})
        nodes_df["type"] = "user"
        nodes_df["url"] = nodes_df["id"].map(_bsky_profile_url)
        nodes_df["is_seed"] = True
        nodes_df["in_reposts"] = 0
        nodes_df["out_reposts"] = 0
        nodes_df["repost_count"] = 0
        # ラベルは seed を★付きにしておく
        nodes_df["label"] = nodes_df["id"].apply(lambda did: f"★ {did}")
        edges_df = pd.DataFrame(columns=["src", "dst", "kind", "weight"])
        return nodes_df, edges_df

    # --- エッジ DataFrame ---
    edges_df = pd.DataFrame(sampled_edges, columns=["src", "dst"])
    edges_df["src"] = edges_df["src"].astype(str)
    edges_df["dst"] = edges_df["dst"].astype(str)
    edges_df["kind"] = "repost"
    edges_df["weight"] = 1

    # 同じ (src, dst) のエッジが複数あれば weight に集約
    edges_df = (
        edges_df
        .groupby(["src", "dst", "kind"], as_index=False)["weight"]
        .sum()
    )

    # --- ノード集合 ---
    node_ids = sorted(set(edges_df["src"]).union(edges_df["dst"]))
    nodes_df = pd.DataFrame({"id": node_ids})
    nodes_df["id"] = nodes_df["id"].astype(str)
    nodes_df["type"] = "user"

    # プロフィール URL（クリックで Bluesky プロフィールへ飛べるように）
    nodes_df["url"] = nodes_df["id"].map(_bsky_profile_url)

    # seed フラグ
    seed_set = set(seed_accounts)
    nodes_df["is_seed"] = nodes_df["id"].isin(seed_set)

    # in / out のリポスト数をカウント
    in_re = (
        edges_df
        .groupby("dst")["weight"]
        .sum()
        .rename("in_reposts")
    )
    out_re = (
        edges_df
        .groupby("src")["weight"]
        .sum()
        .rename("out_reposts")
    )

    nodes_df = nodes_df.merge(
        in_re, how="left", left_on="id", right_index=True
    )
    nodes_df = nodes_df.merge(
        out_re, how="left", left_on="id", right_index=True
    )

    nodes_df["in_reposts"] = nodes_df["in_reposts"].fillna(0).astype(int)
    nodes_df["out_reposts"] = nodes_df["out_reposts"].fillna(0).astype(int)
    nodes_df["repost_count"] = nodes_df["in_reposts"] + nodes_df["out_reposts"]

    # ラベル: seed は★付き、それ以外は DID をそのまま
    def _make_label(row):
        did = row["id"]
        return f"★ {did}" if row["is_seed"] else did

    nodes_df["label"] = nodes_df.apply(_make_label, axis=1)

    return nodes_df, edges_df

def propagate_multilabel_from_seeds_threshold(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    seed_ids: Sequence[str],
    *,
    threshold: float = 0.5,
    max_iter: int = 50,
    use_weight: bool = True,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    seed_ids に 1,2,... のラベル番号を振り、
    「しきい値以上のコミュニティに属する」マルチラベル伝搬を行う。

    - 各ノードは複数ラベル（コミュニティ）に所属しうる。
    - ノード u の近傍 v からの投票を、エッジ重みで集計し、
      w_c / total_w >= threshold を満たすラベル c を全部採用する。

    追加される列:
      - mlp_labels    : List[int]   → 所属ラベル番号のリスト
      - mlp_label_str : str         → "1,3" のような文字列
      - mlp_in_comm_k : bool        → ラベル k に属するかどうか (k=1..K)
    """
    rng = random.Random(random_state)

    # --- 無向 + 重み付きグラフを構築 ---
    G = nx.Graph()
    for row in edges_df.itertuples(index=False):
        src = getattr(row, "src", None)
        dst = getattr(row, "dst", None)
        if not src or not dst:
            continue

        w = getattr(row, "weight", 1.0)
        if w is None:
            w = 1.0

        if G.has_edge(src, dst):
            G[src][dst]["weight"] += float(w)
        else:
            G.add_edge(src, dst, weight=float(w))

    # 孤立ノードも含める
    for nid in nodes_df["id"]:
        if nid not in G:
            G.add_node(nid)

    # --- seed にラベル番号を振る: seed_ids[i] -> ラベル i+1 ---
    seed_ids = [str(s) for s in seed_ids]
    seed_label_map = {sid: (i + 1) for i, sid in enumerate(seed_ids)}
    num_labels = len(seed_ids)

    # 各ノードのラベル集合: Dict[node_id, Set[int]]
    labels: dict[str, set[int]] = {}
    for n in G.nodes():
        if n in seed_label_map:
            labels[n] = {seed_label_map[n]}  # seed は 1つラベルを持ってスタート
        else:
            labels[n] = set()               # 非 seed は空集合

    non_seed_nodes = [n for n in G.nodes() if n not in seed_label_map]

    # --- ラベル伝搬ループ ---
    for it in range(max_iter):
        changed = False
        rng.shuffle(non_seed_nodes)  # 順番をランダム化

        new_labels = {n: set(lbls) for n, lbls in labels.items()}  # コピー

        for u in non_seed_nodes:
            # 近傍のラベルを集計
            vote_counter: Counter[int] = Counter()
            total_w = 0.0

            for v, data in G[u].items():
                nbr_labels = labels.get(v, set())
                if not nbr_labels:
                    continue

                w = data.get("weight", 1.0) or 1.0
                if use_weight:
                    total_w += w
                    for c in nbr_labels:
                        vote_counter[c] += w
                else:
                    total_w += 1.0
                    for c in nbr_labels:
                        vote_counter[c] += 1.0

            if not vote_counter or total_w <= 0.0:
                # 近傍にラベル付きノードがいない → ラベル変化なし
                continue

            # しきい値以上のラベルだけ採用
            new_set: set[int] = set()
            for c, w_c in vote_counter.items():
                if (w_c / total_w) >= threshold:
                    new_set.add(c)

            if new_set != labels[u]:
                new_labels[u] = new_set
                changed = True

        labels = new_labels

        if not changed:
            # print(f"[MLP] converged at iter={it}")
            break

    # --- DataFrame に書き戻し ---
    out = nodes_df.copy()

    def _labels_for_node(nid: str) -> list[int]:
        s = labels.get(nid, set())
        return sorted(s)

    out["mlp_labels"] = out["id"].astype(str).map(_labels_for_node)
    out["mlp_label_str"] = out["mlp_labels"].apply(
        lambda xs: ",".join(str(x) for x in xs) if xs else ""
    )

    # 各コミュニティに属しているかどうかのブール列
    for k in range(1, num_labels + 1):
        col = f"mlp_in_comm_{k}"
        out[col] = out["mlp_labels"].apply(lambda s: k in s)

    return out

@register_feature("timeseries")
def feature_timeseries(es: "Elasticsearch", p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
        p = {query:str, start:any, end:any, lang:Optional['en'|'ja']}
    Output:
        {"df": pd.DataFrame}  # columns=['time','count']
    """
    impl = _require_impl()
    q   = p["query"]
    st_ = to_iso_utc_day_start(p["start"])
    en_ = to_iso_utc_day_end(p["end"])
    lang = p.get("lang")
    df = impl.get_post_time_series(es, q, st_, en_, lang_filter=lang)
    return {"df": df}

from collections import Counter
from typing import Optional, Collection, Tuple, Dict, List
import pandas as pd

def _iter_reposts_in_period(
    es,
    *,
    repost_index: str = "repostindex-*",
    start_dt=None,
    end_dt=None,
    batch_size: int = 5000,
) -> List[Tuple[str, str]]:
    """
    期間内の repost ドキュメントから
    (subject_cid, reposter_did) のリストを取得するヘルパー。

    ※ Bluesky の ES マッピングを前提:
      - commit.collection == "app.bsky.feed.repost"
      - commit.record.subject.cid に元ポストの cid
      - did フィールドにリポストしたユーザの DID
    """
    query: Dict = {
        "size": batch_size,
        "_source": ["did", "commit.record.subject.cid", "commit.record.createdAt"],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.collection": "app.bsky.feed.repost"}},
                ]
            }
        },
        "sort": [{"commit.record.createdAt": "asc"}],
    }

    # 期間指定があれば range フィルタを追加
    if start_dt is not None and end_dt is not None:
        query["query"]["bool"]["filter"].append(
            {
                "range": {
                    "commit.record.createdAt": {
                        "gte": start_dt.isoformat(),
                        "lte": end_dt.isoformat(),
                    }
                }
            }
        )

    # 単純な search_after ループ
    results: List[Tuple[str, str]] = []
    search_after = None

    while True:
        if search_after is not None:
            query["search_after"] = search_after

        res = es.search(
            index=repost_index,
            body=query,
            ignore_unavailable=True,
            allow_no_indices=True,
            expand_wildcards="open",
            request_timeout=120,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        if not hits:
            break

        for h in hits:
            src = h.get("_source") or {}
            did = src.get("did")
            commit = src.get("commit") or {}
            record = commit.get("record") or {}
            subj = record.get("subject") or {}
            cid = subj.get("cid")

            if isinstance(did, str) and isinstance(cid, str) and did and cid:
                # (元ポストのCID, リポストしたユーザ)
                results.append((cid, did))

        search_after = hits[-1].get("sort")
        if search_after is None:
            break

    return results

def _iter_replies_in_period(
    es,
    *,
    post_index: str = "postindex-*",
    start_dt=None,
    end_dt=None,
    batch_size: int = 5000,
) -> List[Tuple[str, str]]:
    """
    期間内の reply 投稿から
    (parent_cid, replier_did) のリストを取得するヘルパー。

    ※ Bluesky の ES マッピングを前提:
      - commit.collection == "app.bsky.feed.post"
      - commit.record.reply.parent.cid に「返信先」の CID
      - did フィールドに返信したユーザの DID
    """
    query: Dict = {
        "size": batch_size,
        "_source": [
            "did",
            "commit.record.reply.parent.cid",
            "commit.record.createdAt",
            "commit.collection",
        ],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.collection": "app.bsky.feed.post"}},
                    {"exists": {"field": "commit.record.reply"}},
                ]
            }
        },
        "sort": [{"commit.record.createdAt": "asc"}],
    }

    if start_dt is not None and end_dt is not None:
        query["query"]["bool"]["filter"].append(
            {
                "range": {
                    "commit.record.createdAt": {
                        "gte": start_dt.isoformat(),
                        "lte": end_dt.isoformat(),
                    }
                }
            }
        )

    results: List[Tuple[str, str]] = []
    search_after = None

    while True:
        if search_after is not None:
            query["search_after"] = search_after

        res = es.search(
            index=post_index,
            body=query,
            ignore_unavailable=True,
            allow_no_indices=True,
            expand_wildcards="open",
            request_timeout=120,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        if not hits:
            break

        for h in hits:
            src = h.get("_source") or {}
            did = src.get("did")
            commit = src.get("commit") or {}
            record = commit.get("record") or {}
            rep_info = record.get("reply") or {}
            parent = rep_info.get("parent") or {}
            parent_cid = parent.get("cid")

            if isinstance(did, str) and isinstance(parent_cid, str) and did and parent_cid:
                # (親ポストCID, 返信したユーザ)
                results.append((parent_cid, did))

        search_after = hits[-1].get("sort")
        if search_after is None:
            break

    return results


def _iter_quotes_in_period(
    es,
    *,
    post_index: str = "postindex-*",
    start_dt=None,
    end_dt=None,
    batch_size: int = 5000,
) -> List[Tuple[str, str]]:
    """
    期間内の quote 投稿から
    (quoted_cid, quoter_did) のリストを取得するヘルパー。

    ※ Bluesky の ES マッピングに依存します。
      ここでは例として:
        commit.record.embed.record.cid に引用先ポストの CID が入っている想定。

      実際の mapping によって field 名は適宜修正してください。
    """
    query: Dict = {
        "size": batch_size,
        "_source": [
            "did",
            "commit.record.embed.record.cid",
            "commit.record.createdAt",
            "commit.collection",
        ],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.collection": "app.bsky.feed.post"}},
                    {"exists": {"field": "commit.record.embed.record.cid"}},
                ]
            }
        },
        "sort": [{"commit.record.createdAt": "asc"}],
    }

    if start_dt is not None and end_dt is not None:
        query["query"]["bool"]["filter"].append(
            {
                "range": {
                    "commit.record.createdAt": {
                        "gte": start_dt.isoformat(),
                        "lte": end_dt.isoformat(),
                    }
                }
            }
        )

    results: List[Tuple[str, str]] = []
    search_after = None

    while True:
        if search_after is not None:
            query["search_after"] = search_after

        res = es.search(
            index=post_index,
            body=query,
            ignore_unavailable=True,
            allow_no_indices=True,
            expand_wildcards="open",
            request_timeout=120,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        if not hits:
            break

        for h in hits:
            src = h.get("_source") or {}
            did = src.get("did")
            commit = src.get("commit") or {}
            record = commit.get("record") or {}
            embed = record.get("embed") or {}
            rec = embed.get("record") or {}
            quoted_cid = rec.get("cid")

            if isinstance(did, str) and isinstance(quoted_cid, str) and did and quoted_cid:
                # (引用元CID, 引用したユーザ)
                results.append((quoted_cid, did))

        search_after = hits[-1].get("sort")
        if search_after is None:
            break

    return results


def _iter_mentions_in_period(
    es,
    *,
    post_index: str = "postindex-*",
    start_dt=None,
    end_dt=None,
    batch_size: int = 5000,
) -> List[Tuple[str, str]]:
    """
    期間内の投稿から (author_did, mentioned_did) を返すヘルパー。

    ※ これも ES のマッピングにかなり依存します。
      - もし commit.record.mentions[].did のような配列があればそれを利用。
      - そうでなければ、text から @handle を正規表現で抜き出し、
        別途 handle->DID の辞書を用意して解決する必要があります。

    ここでは「mentions[].did がある場合」の素直な実装にしてあります。
    """
    query: Dict = {
        "size": batch_size,
        "_source": [
            "did",
            "commit.record.mentions",
            "commit.record.text",
            "commit.record.createdAt",
            "commit.collection",
        ],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.collection": "app.bsky.feed.post"}},
                    {"exists": {"field": "commit.record.mentions"}},
                ]
            }
        },
        "sort": [{"commit.record.createdAt": "asc"}],
    }

    if start_dt is not None and end_dt is not None:
        query["query"]["bool"]["filter"].append(
            {
                "range": {
                    "commit.record.createdAt": {
                        "gte": start_dt.isoformat(),
                        "lte": end_dt.isoformat(),
                    }
                }
            }
        )

    results: List[Tuple[str, str]] = []
    search_after = None

    while True:
        if search_after is not None:
            query["search_after"] = search_after

        res = es.search(
            index=post_index,
            body=query,
            ignore_unavailable=True,
            allow_no_indices=True,
            expand_wildcards="open",
            request_timeout=120,
        )
        hits = (res.get("hits") or {}).get("hits") or []
        if not hits:
            break

        for h in hits:
            src = h.get("_source") or {}
            author = src.get("did")
            commit = src.get("commit") or {}
            record = commit.get("record") or {}
            mentions = record.get("mentions") or []

            if not isinstance(author, str) or not author:
                continue

            # mentions が [{ "did": "...", ... }, ...] 形式を想定
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                mdid = m.get("did")
                if isinstance(mdid, str) and mdid and mdid != author:
                    # (書いた人, メンションされた人)
                    results.append((author, mdid))

        search_after = hits[-1].get("sort")
        if search_after is None:
            break

    return results


def build_information_flow_edges(
    es,
    *,
    post_index: str = "postindex-*",
    repost_index: str = "repostindex-*",
    start_dt=None,
    end_dt=None,
    allowed_users: Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    SPIN / NNIF 用の「情報フロー・ネットワーク」のエッジ集合を構成する。

    - repost : 元投稿者 -> リポストしたユーザ
    - reply  : 元投稿者 -> 返信したユーザ
    - quote  : 元投稿者 -> 引用したユーザ
    - mention: 投稿したユーザ -> メンションされたユーザ

    allowed_users が指定されていれば、その集合に含まれないユーザ間エッジは捨てる。

    戻り値:
      edges_df: [src, dst, kind, weight]
    """
    # ---------------------------------------------------------
    # 0) allowed_users をセットにしておく（フィルタ用）
    # ---------------------------------------------------------
    if allowed_users is not None:
        allowed_set = {str(u) for u in allowed_users}
    else:
        allowed_set = None

    # ---------------------------------------------------------
    # 1) repost: (subject_cid, reposter_did) -> (author_did, reposter_did)
    # ---------------------------------------------------------
    repost_pairs = _iter_reposts_in_period(
        es,
        repost_index=repost_index,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    # すべての subject_cid をまとめて投稿者 DID に引く
    subject_cids = [cid for cid, _ in repost_pairs]
    cid_to_author = _map_cids_to_authors(es, subject_cids, post_index=post_index)

    edge_counter: Counter[Tuple[str, str, str]] = Counter()

    for cid, reposter in repost_pairs:
        author = cid_to_author.get(cid)
        if not author:
            continue
        src = str(author)
        dst = str(reposter)
        if src == dst:
            continue
        if allowed_set is not None and (src not in allowed_set or dst not in allowed_set):
            continue
        edge_counter[(src, dst, "repost")] += 1

    # ---------------------------------------------------------
    # 2) reply: (parent_cid, replier_did) -> (parent_author_did, replier_did)
    # ---------------------------------------------------------
    reply_pairs = _iter_replies_in_period(
        es,
        post_index=post_index,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    parent_cids = [cid for cid, _ in reply_pairs]
    cid_to_author_reply = _map_cids_to_authors(es, parent_cids, post_index=post_index)

    for cid, replier in reply_pairs:
        author = cid_to_author_reply.get(cid)
        if not author:
            continue
        src = str(author)
        dst = str(replier)
        if src == dst:
            continue
        if allowed_set is not None and (src not in allowed_set or dst not in allowed_set):
            continue
        edge_counter[(src, dst, "reply")] += 1

    # ---------------------------------------------------------
    # 3) quote: (quoted_cid, quoter_did) -> (quoted_author_did, quoter_did)
    # ---------------------------------------------------------
    quote_pairs = _iter_quotes_in_period(
        es,
        post_index=post_index,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    quoted_cids = [cid for cid, _ in quote_pairs]
    cid_to_author_quote = _map_cids_to_authors(es, quoted_cids, post_index=post_index)

    for cid, quoter in quote_pairs:
        author = cid_to_author_quote.get(cid)
        if not author:
            continue
        src = str(author)
        dst = str(quoter)
        if src == dst:
            continue
        if allowed_set is not None and (src not in allowed_set or dst not in allowed_set):
            continue
        edge_counter[(src, dst, "quote")] += 1

    # ---------------------------------------------------------
    # 4) mention: (author_did, mentioned_did)
    # ---------------------------------------------------------
    mention_pairs = _iter_mentions_in_period(
        es,
        post_index=post_index,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    for author, mentioned in mention_pairs:
        src = str(author)
        dst = str(mentioned)
        if src == dst:
            continue
        if allowed_set is not None and (src not in allowed_set or dst not in allowed_set):
            continue
        edge_counter[(src, dst, "mention")] += 1

    # ---------------------------------------------------------
    # 5) Counter -> edges_df へ変換
    # ---------------------------------------------------------
    rows = [
        {"src": s, "dst": t, "kind": k, "weight": int(w)}
        for (s, t, k), w in edge_counter.items()
    ]
    edges_df = pd.DataFrame(rows, columns=["src", "dst", "kind", "weight"])
    return edges_df


@register_feature("user_ranking")
def feature_user_ranking(es: "Elasticsearch", p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
        p = {query:str, start:any, end:any, token: Optional[str]}
    Output:
        {"df": pd.DataFrame}  # did, post_count, (handle, displayName), profile_url, profile_image
    """
    impl = _require_impl()
    q   = p["query"]
    st_ = to_iso_utc_day_start(p["start"])
    en_ = to_iso_utc_day_end(p["end"])
    token = p.get("token")

    df = impl.get_post_user_ranking(es, q, st_, en_)

    # Optional enrichments
    if token and impl.fetch_display_info_parallel is not None:
        did_list = df.get("did", pd.Series([])).tolist()
        try:
            did_map = impl.fetch_display_info_parallel(did_list, token)
        except Exception:
            did_map = {}
        df["handle"] = df["did"].map(lambda d: did_map.get(d, {}).get("handle", d))
        df["displayName"] = df["did"].map(lambda d: did_map.get(d, {}).get("displayName", ""))
    else:
        if "did" in df.columns:
            df["handle"] = df["did"]
        df["displayName"] = df.get("displayName", pd.Series([""] * len(df)))

    # profile url / avatar (avatar depends on your impl, may require global es)
    if "did" in df.columns:
        df["profile_url"] = df["did"].apply(lambda did: f"https://bsky.app/profile/{did}")
        if impl.get_profile_avatar_url is not None:
            try:
                df["profile_image"] = df["did"].apply(lambda did: impl.get_profile_avatar_url(did))
            except Exception:
                df["profile_image"] = None
        else:
            df["profile_image"] = None

    # sort if post_count exists
    if "post_count" in df.columns:
        df = df.sort_values("post_count", ascending=False)

    return {"df": df}

# research_matae.py に追記

from typing import List, Dict
import pandas as pd

def _user_recent_posts(
    es,
    did: str,
    *,
    start_dt,
    end_dt,
    max_posts: int = 20,
    post_index: str = "postindex-*",
) -> pd.DataFrame:
    """
    特定ユーザ did の投稿を期間内から新しい順に max_posts 件だけ取得。
    """
    body: Dict = {
        "_source": [
            "did",
            "commit.record.text",
            "commit.record.createdAt",
        ],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"did": did}},
                    {
                        "range": {
                            "commit.record.createdAt": {
                                "gte": start_dt.isoformat(),
                                "lte": end_dt.isoformat(),
                            }
                        }
                    },
                    {"term": {"commit.collection": "app.bsky.feed.post"}},
                ]
            }
        },
        "sort": [{"commit.record.createdAt": {"order": "desc"}}],
        "size": max_posts,
    }

    res = es.search(
        index=post_index,
        body=body,
        ignore_unavailable=True,
        allow_no_indices=True,
        expand_wildcards="open",
        request_timeout=120,
    )

    rows = []
    for h in (res.get("hits") or {}).get("hits") or []:
        src = h.get("_source") or {}
        commit = src.get("commit") or {}
        record = commit.get("record") or {}
        text = record.get("text") or ""
        created_at = record.get("createdAt")
        rows.append(
            {
                "did": src.get("did"),
                "text": text,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows)

def is_negative_user(
    es,
    did: str,
    *,
    start_dt,
    end_dt,
    negative_words: List[str] = None,
    max_posts: int = 20,
    post_index: str = "postindex-*",
) -> bool:
    """
    1ユーザについて、
    期間内の最近 max_posts 件の投稿のどれか1つでも
    ネガティブワードが含まれていれば True を返す。
    """
    if negative_words is None:
        negative_words = NEGATIVE_KEYWORDS

    df = _user_recent_posts(
        es,
        did,
        start_dt=start_dt,
        end_dt=end_dt,
        max_posts=max_posts,
        post_index=post_index,
    )
    if df.empty:
        return False

    neg_words = [w for w in negative_words if w]  # 念のため空文字除去

    for text in df["text"].astype(str):
        for w in neg_words:
            if w in text:
                return True
    return False

def attach_negative_flag_to_nodes(
    es,
    nodes_df: pd.DataFrame,
    *,
    id_col: str = "id",  # nodes_df 側のユーザID列
    start_dt,
    end_dt,
    max_posts: int = 20,
    post_index: str = "postindex-*",
) -> pd.DataFrame:
    """
    SPIN ノードDFに is_negative 列を追加する。
    各ユーザについて 10〜20 件投稿を取り、1件でもネガなら True。
    """
    nodes_df = nodes_df.copy()
    flags: List[bool] = []

    for did in nodes_df[id_col].astype(str):
        df_posts = _user_recent_posts(
            es,
            did,
            start_dt=start_dt,
            end_dt=end_dt,
            max_posts=max_posts,
            post_index=post_index,
        )
        if df_posts.empty:
            flags.append(False)
            continue

        texts = df_posts["text"].astype(str).tolist()
        is_neg = is_negative_user_from_texts(texts)
        flags.append(is_neg)

    nodes_df["is_negative"] = flags
    return nodes_df

def plot_nnif_log_distribution(log_dist, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(log_dist, bins=30)
    plt.xlabel("log10(NNIF sum)")
    plt.ylabel("Frequency")
    plt.title("Distribution of NNIF")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_comm_topic_heatmap(agg_df, out_path):
    pivot = agg_df.pivot_table(
        index=["src_community", "src_topic"],
        columns=["dst_community", "dst_topic"],
        values="nnif_sum",
        fill_value=0,
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        square=False,
    )
    plt.title("Topic-aware NNIF matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

@register_feature("word_ranking")
def feature_word_ranking(es: "Elasticsearch", p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
        p = {query:str, start:any, end:any, lang:Optional['en'|'ja'], top_n:int=25}
    Output:
        {"ranking": List[Tuple[word, count]]}
    """
    impl = _require_impl()
    q   = p["query"]
    st_ = to_iso_utc_day_start(p["start"])
    en_ = to_iso_utc_day_end(p["end"])
    lang = p.get("lang")
    top_n = int(p.get("top_n", 25))
    ranking = impl.get_word_frequency_ranking(es, q, st_, en_, lang_filter=lang, top_n=top_n)
    return {"ranking": ranking}


@register_feature("cluster_texts")
def feature_cluster_texts(es: "Elasticsearch", p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
        p = {query:str, start:any, end:any, n_clusters:int=5}
    Output:
        {"df": pd.DataFrame}  # columns=['x','y','cluster','text','cluster_str']
    """
    impl = _require_impl()
    q   = p["query"]
    st_ = to_iso_utc_day_start(p["start"])
    en_ = to_iso_utc_day_end(p["end"])
    k   = int(p.get("n_clusters", 5))
    df  = impl.cluster_post_texts(es, q, st_, en_, n_clusters=k)
    if df is None:
        return {"df": pd.DataFrame()}
    if "cluster" in df.columns:
        df["cluster_str"] = df["cluster"].astype(str)
    return {"df": df}

@register_feature("snowball_interactions")
def feature_snowball_interactions(es: "Elasticsearch", p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      p = {
        posts_df: pd.DataFrame, start:any, end:any,
        iterations:int=8, users_per_iteration:int=12, label:str=None
      }
    Output:
      {"graph": nx.MultiDiGraph, "posts": pd.DataFrame}
    """
    impl = _require_impl()
    posts_df = p["posts_df"]
    st_ = p["start"]
    en_ = p["end"]
    it = int(p.get("iterations", 8))
    k  = int(p.get("users_per_iteration", 12))
    label = p.get("label")

    result = impl.create_recursive_interaction_user_network_parallel(
        es=es,
        start_date=st_,
        end_date=en_,
        iterations=it,
        users_per_iteration=k,
        posts_df=posts_df,
        label=label
    )

    # Some existing implementations may return None on error
    if result is None:
        return {"graph": nx.MultiDiGraph(), "posts": pd.DataFrame()}

    G, df = result
    return {"graph": G, "posts": df}


@register_feature("merge_graphs")
def feature_merge_graphs(es: "Elasticsearch", p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
        p = {graphs: List[nx.MultiDiGraph]}
    Output:
        {"graph": nx.MultiDiGraph}
    """
    impl = _require_impl()
    graphs = p["graphs"]
    if not graphs:
        raise ValueError("graphs is empty")
    if len(graphs) == 1:
        return {"graph": graphs[0]}
    merged = impl.merge_graphs_on_common_nodes_multi(graphs)
    return {"graph": merged}


__all__ = [
    "Impl",
    "set_impl",
    "FeatureSpec",
    "run_features",
    "register_feature",
    "feature_timeseries",
    "feature_user_ranking",
    "feature_word_ranking",
    "feature_cluster_texts",
    "feature_snowball_interactions",
    "feature_merge_graphs",
    "to_iso_utc_day_start",
    "to_iso_utc_day_end",
]
