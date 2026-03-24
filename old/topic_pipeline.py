# topic_pipeline.py
"""
BERTopic を用いて、SNS投稿から
「ユーザごとのトピックプロファイル」を推定し、
ノードDF（コミュニティラベル付き）に結合するためのユーティリティ集。

想定：
- posts_df: user_id, text を持つ DataFrame
- nodes_df: user_id, lp_labels(ラベル伝搬コミュニティ) を持つ DataFrame
"""

from typing import Optional, Any, List, Dict

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
# =========================
# 1. BERTopic モデルの構築
# =========================
def build_bsky_bertopic_model() -> BERTopic:
    """
    Bluesky など短文SNS向けの BERTopic モデルを構築する。

    - 多言語埋め込み（日本語対応）
    - UMAP + HDBSCAN を明示指定
    - CountVectorizer で BoW を制御
    - calculate_probabilities=True で各投稿のトピック分布を出す
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 高性能日本語 + 多言語モデル
    embedding_model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        device=device,
    )

    # UMAP改良
    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,     # ★ 次元数UP
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN改良
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=15,    # ★ SNS向けに小さめ
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="leaf",  # 小クラスタ検出に強い
        prediction_data=True,
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,
        bm25_weighting=True
    )

    vectorizer_model = CountVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),     # ★ 重要
        tokenizer=lambda x: list(x),  # ★ 簡易トークナイザ（後でMeCabに差し替え可）
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True,
    )

    return topic_model

# ==================================
# 2. 投稿に対して BERTopic を実行する
# ==================================

def run_bertopic_on_posts(
    posts_df: pd.DataFrame,
    text_col: str = "text",
    model_save_path: Optional[str] = None,
) -> tuple[pd.DataFrame, BERTopic]:
    """
    posts_df に対して BERTopic を学習し、
    各投稿に topic_id と topic_probas を付与する。

    Parameters
    ----------
    posts_df : pd.DataFrame
        user_id, text の列を持つ DataFrame を想定。
    text_col : str
        投稿本文の列名。
    model_save_path : str | None
        学習済みモデルを保存したい場合のパス。

    Returns
    -------
    posts_df_out : pd.DataFrame
        topic_id, topic_probas 列が追加された DataFrame。
    topic_model : BERTopic
        学習済み BERTopic モデル。
    """
    topic_model = build_bsky_bertopic_model()

    docs = posts_df[text_col].astype(str).tolist()

    # fit + transform（トピックIDと確率分布を同時に取得）
    topics, probs = topic_model.fit_transform(docs)

    posts_df_out = posts_df.copy()
    posts_df_out["topic_id"] = topics
    posts_df_out["topic_probas"] = list(probs)  # 1行ごとに np.ndarray or list[float]

    # 任意でモデル保存
    if model_save_path is not None:
        topic_model.save(
            model_save_path,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    return posts_df_out, topic_model


def load_bsky_bertopic_model(model_path: str) -> BERTopic:
    """
    既存の BERTopic モデルをロードするヘルパー。
    （新規学習ではなく、ストリーミングデータに使うとき用）
    """
    topic_model = BERTopic.load(model_path)
    return topic_model


def assign_topics_to_new_posts(
    topic_model: BERTopic,
    new_posts_df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    学習済み BERTopic モデルを使って、新規投稿に topic_id, topic_probas を付与する。

    Parameters
    ----------
    topic_model : BERTopic
        すでに fit 済みの BERTopic モデル。
    new_posts_df : pd.DataFrame
        user_id, text を持つ DataFrame。
    text_col : str
        投稿本文の列名。

    Returns
    -------
    new_posts_df_out : pd.DataFrame
        topic_id, topic_probas 列を追加した DataFrame。
    """
    docs = new_posts_df[text_col].astype(str).tolist()
    topics, probs = topic_model.transform(docs)

    new_posts_df_out = new_posts_df.copy()
    new_posts_df_out["topic_id"] = topics
    new_posts_df_out["topic_probas"] = list(probs)

    return new_posts_df_out


# ======================================
# 3. ユーザごとのトピックプロファイルを計算
# ======================================

def compute_user_topic_profiles(
    posts_df: pd.DataFrame,
    user_col: str = "id",
    probas_col: str = "topic_probas",
    topic_threshold: float = 0.2,
    topic_model: Optional[BERTopic] = None,
) -> pd.DataFrame:
    """
    BERTopic のトピック分布から、ユーザごとのトピックプロファイルを計算する。

    流れ：
    - user_id ごとに、投稿の topic_probas を平均
    - 平均ベクトルの成分が閾値以上のトピックだけを採用

    Returns
    -------
    user_topic_df : pd.DataFrame
        各行が1ユーザ。
        user_col, topic_vector, user_topic_ids, user_topic_labels などを持つ。
    """
    # topic_probas が入っている行だけ利用
    valid_posts = posts_df.dropna(subset=[probas_col]).copy()

    # 各行のベクトルを np.ndarray に統一
    def _to_array(x):
        if isinstance(x, np.ndarray):
            return x
        return np.array(x, dtype=float)

    valid_posts["topic_probas_arr"] = valid_posts[probas_col].apply(_to_array)

    if valid_posts.empty:
        # 投稿が全くないケースのガード
        return pd.DataFrame(columns=[user_col, "topic_vector", "user_topic_ids", "user_topic_labels"])

    # 一つサンプルを取り出してトピック数を決定
    sample_vec = valid_posts["topic_probas_arr"].iloc[0]
    n_topics = len(sample_vec)

    user2sum: Dict[Any, np.ndarray] = {}
    user2cnt: Dict[Any, int] = {}

    # ユーザごとに確率ベクトルを加算
    for row in valid_posts.itertuples():
        uid = getattr(row, user_col)
        vec = row.topic_probas_arr

        if uid not in user2sum:
            user2sum[uid] = np.zeros(n_topics, dtype=float)
            user2cnt[uid] = 0

        user2sum[uid] += vec
        user2cnt[uid] += 1

    records: List[Dict[str, Any]] = []
    for uid, svec in user2sum.items():
        cnt = user2cnt[uid]
        mean_vec = svec / max(cnt, 1)

        # 閾値以上のトピックID
        topic_ids = np.where(mean_vec >= topic_threshold)[0].tolist()

        # トピック名（ラベル）はここでは簡易に "Topic {id}" にする
        topic_labels = None
        if topic_model is not None:
            labels = [f"Topic {tid}" for tid in topic_ids]
            topic_labels = labels

        records.append(
            {
                user_col: uid,
                "topic_vector": mean_vec,
                "user_topic_ids": topic_ids,
                "user_topic_labels": topic_labels,
            }
        )

    user_topic_df = pd.DataFrame(records)
    return user_topic_df


# ================================
# 4. ノードDFへトピック情報を付与
# ================================

def attach_topics_to_nodes(
    nodes_df: pd.DataFrame,
    user_topic_df: pd.DataFrame,
    user_col: str = "id",
) -> pd.DataFrame:
    """
    ノードDF（ユーザDF）にユーザごとのトピック情報をマージする。

    Parameters
    ----------
    nodes_df : pd.DataFrame
        user_id, lp_labels などを持つ DF。
    user_topic_df : pd.DataFrame
        compute_user_topic_profiles() で作成した DF。
    user_col : str
        ユーザID列名。

    Returns
    -------
    merged_df : pd.DataFrame
        nodes_df に topic_vector, user_topic_ids, user_topic_labels が追加された DF。
    """
    merged_df = nodes_df.merge(
        user_topic_df[[user_col, "topic_vector", "user_topic_ids", "user_topic_labels"]],
        on=user_col,
        how="left",
    )
    return merged_df


# ================================
# 5. Streamlit での表示用関数
# ================================

def render_nodes_table_with_community_and_topics(nodes_df: pd.DataFrame) -> None:
    """
    ノード（ユーザ）ごとのコミュニティラベルとトピックを Streamlit で表示する。
    """
    st.subheader("ノード属性（ラベル伝搬コミュニティ × BERTopic トピック）")

    # 表示したい列を選択（存在する列だけ）
    display_cols = []
    for col in ["id", "lp_labels", "user_topic_ids", "user_topic_labels"]:
        if col in nodes_df.columns:
            display_cols.append(col)

    st.dataframe(nodes_df[display_cols])