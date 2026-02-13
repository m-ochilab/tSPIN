# spin_runner.py
# -----------------
# Bluesky + Elasticsearch 向けの SPIN 用 seed 選択 & Snowball 実行ユーティリティ
#
# - 期間内の投稿から「リポストが多い著者」を seed accounts として選ぶ
# - その seed をもとに snowball_spin() を実行して、情報拡散ネットワークをサンプリング
#
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Sequence, Callable

from elasticsearch import Elasticsearch
from snowball_spin import snowball_spin  # 既存の Algorithm 1 実装を利用

# =========================================================
# 1. seed accounts を選ぶ関数
# =========================================================

def select_seed_accounts_by_max_reposts(
    es: Elasticsearch,
    *,
    post_index: str = "postindex-*",
    start_day: datetime,
    end_day: datetime,
    min_reposts_for_seed: int = 5,
    top_n_seed: int = 20,
    author_field: str = "did",
    created_at_field: str = "commit.record.createdAt",
    repost_count_field: str = "repost_count",
    text_query: Optional[str] = None,
    text_field: str = "commit.record.text",
) -> List[str]:
    """
    期間内の投稿から「リポスト数の多い著者」を seed accounts として選ぶ。

    前提:
      - 投稿ドキュメントに `repost_count_field` (例: 再投稿数) が格納されている。
        格納されていない場合は、事前集計してコピーしておくか、
        この関数をカスタマイズして likeindex/repostindex を参照するように変更する。

    戻り値:
      seed_accounts: 著者ID（Blueskyなら DID）のリスト
    """
    must_filters = [
        {
            "range": {
                created_at_field: {
                    "gte": start_day.isoformat(),
                    "lte": end_day.isoformat(),
                }
            }
        }
    ]
    must_clauses = []

    if text_query:
        must_clauses.append({"match": {text_field: {"query": text_query, "operator": "and"}}})

    body = {
        "size": 0,
        "query": {
            # "bool": {
            #     "must": must_clauses,
            #     "filter": must_filters,
            # }
            "bool": {
              "must": [
                { "match": { "commit.record.text" : {
                    "query":text_query,
                    "operator" : "and"
                    } }
                }
              ],
              "filter": [
                { "range": { "commit.record.createdAt": {
                    "gte": start_day.isoformat(),
                    "lte": end_day.isoformat(),
                }}}
              ],
              "must_not": [
                { "exists": { "field": "commit.record.reply" } }
              ]
            }
        },
        "aggs": {
            "by_author": {
                "terms": {
                    "field": author_field,
                    "size": max(top_n_seed * 5, 50),  # 余裕をもって多めに
                },
                "aggs": {
                    "max_reposts": {
                        "max": {"field": repost_count_field}
                    }
                },
            }
        },
    }

    res = es.search(
        index=post_index,
        body=body,
        request_timeout=600,
    )

    buckets = (
        res.get("aggregations", {})
           .get("by_author", {})
           .get("buckets", [])
    ) or []

    # max_reposts が一定以上の著者を抽出し、降順ソート
    candidates: List[Tuple[str, float]] = []
    for b in buckets:
        author = b.get("key")
        max_r = (b.get("max_reposts") or {}).get("value")
        if author is None or max_r is None:
            continue
        if max_r < min_reposts_for_seed:
            continue
        candidates.append((author, float(max_r)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    seeds = [a for (a, _) in candidates[:top_n_seed]]
    return seeds

import pandas as pd
from typing import Optional, List

def select_seed_accounts_from_df(
    df: pd.DataFrame,
    *,
    author_field: str = "did",
    repost_count_field: str = "repost_count",
    top_n_seed: int = 1,
    min_reposts_for_seed: int = 0,
) -> List[str]:
    """
    research_matae から渡ってきた DataFrame から seed_account を決める。

    - 各著者ごとに repost_count の最大値をとる
    - min_reposts_for_seed 以上の著者だけ残す
    - 大きい順に並べて上位 top_n_seed を返す
    """
    if df.empty:
        return []

    if author_field not in df.columns or repost_count_field not in df.columns:
        raise ValueError(f"DataFrame に {author_field}, {repost_count_field} が必要です。")

    # 数値に変換（文字列混じりでも落ちないように）
    tmp = df[[author_field, repost_count_field]].copy()
    tmp[repost_count_field] = pd.to_numeric(
        tmp[repost_count_field], errors="coerce"
    )
    tmp = tmp.dropna(subset=[repost_count_field])

    # 著者ごとの最大リポスト数
    by_author = (
        tmp.groupby(author_field)[repost_count_field]
           .max()
    )

    # しきい値でフィルタ
    by_author = by_author[by_author >= min_reposts_for_seed]

    if by_author.empty:
        return []

    # 大きい順に並べて上位を seed に
    by_author = by_author.sort_values(ascending=False)
    seed_accounts = list(by_author.head(top_n_seed).index)
    return seed_accounts


# =========================================================
# 2. Snowball-SPIN 用の ES フェッチャ／リポスター取得関数
# =========================================================

Publication = Dict[str, Any]

def make_bluesky_publication_fetcher(
    es: Elasticsearch,
    *,
    post_index: str = "postindex-*",
    author_field: str = "did",
    created_at_field: str = "commit.record.createdAt",
    repost_count_field: str = "repost_count",
    size_limit: int = 2000,
) -> Callable[[str, datetime, datetime], List[Publication]]:
    """
    snowball_spin() に渡すための fetch_publications(author, start, end) を生成。

    - author_field: 著者ID（Blueskyなら DID）が入っているフィールド
    - created_at_field: 投稿日時
    - repost_count_field: 再投稿数（無い場合は 0 で埋めるか、後でフィルタ条件を緩める）
    """

    def _fetch(author: str, start_dt: datetime, end_dt: datetime) -> List[Publication]:
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {author_field: author}},
                        {
                            "range": {
                                created_at_field: {
                                    "gte": start_dt.isoformat(),
                                    "lte": end_dt.isoformat(),
                                }
                            }
                        },
                    ]
                }
            },
            "sort": [{created_at_field: {"order": "desc"}}],
            "size": size_limit,
            "_source": True,
        }

        res = es.search(
            index=post_index,
            body=body,
            request_timeout=600,
        )

        pubs: List[Publication] = []
        for h in res.get("hits", {}).get("hits", []):
            src = h.get("_source", {}) or {}
            commit = src.get("commit") or {}
            record = commit.get("record") or {}

            cid = commit.get("cid") or src.get("cid") or record.get("cid")
            repost_count = src.get(repost_count_field, 0)

            pubs.append(
                {
                    "id": h.get("_id"),          # ES上のID（必要なら）
                    "cid": cid,                  # Blueskyのpost CID（repostindexと結びつける）
                    "author": src.get(author_field),
                    "created_at": record.get("createdAt"),
                    "repost_count": repost_count,
                }
            )
        return pubs

    return _fetch

def make_bluesky_reposter_getter(
    es: Elasticsearch,
    *,
    repost_index: str = "repostindex-*",
    subject_cid_field: str = "commit.record.subject.cid",
    reposter_field: str = "did",
    batch_size: int = 2000,
) -> Callable[[Publication], List[str]]:
    """
    snowball_spin() に渡す get_reposters(publication) を生成。

    - publication["cid"] をキーにして、repostindex-* からリポスター (did) を取得する前提。
    """

    def _get_reposters(pub: Publication) -> List[str]:
        cid = pub.get("cid")
        if not cid:
            return []

        body = {
            "size": batch_size,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {subject_cid_field: cid}},
                    ]
                }
            },
            "_source": [reposter_field],
        }

        res = es.search(
            index=repost_index,
            body=body,
            request_timeout=600,
            ignore_unavailable=True,
            allow_no_indices=True,
        )

        users: List[str] = []
        for h in res.get("hits", {}).get("hits", []):
            src = h.get("_source", {}) or {}
            u = src.get(reposter_field)
            if u:
                users.append(u)
        return users

    return _get_reposters

# =========================================================
# 3. まとめて実行するヘルパー関数
# =========================================================

import time  # ★ 追加

def run_spin_snowball_for_bluesky(
    es: Elasticsearch,
    *,
    post_index: str = "postindex-*",
    repost_index: str = "repostindex-*",
    start_day: datetime,
    end_day: datetime,
    text_query_for_seed: Optional[str] = None,
    min_reposts_for_seed: int = 5,
    top_n_seed: int = 20,
    number_of_publications: int = 8,
    number_of_recursions: int = 2,
    require_reposts_min: int = 1,
    seed: Optional[int] = 10,
    seed_accounts_override: Optional[Sequence[str]] = None,  # 既存
    logger: Optional[Callable[[str], None]] = None,          # ★ 追加
) -> Dict[str, Any]:
    """
    1. seed accounts を自動選択
    2. snowball_spin() を実行
    までを一気にやる関数。

    戻り値:
      {
        "seed_accounts": [...],       # seedとして使った DID
        "common_list": [...],         # snowball で収集したユーザ（重複あり）
        "sampled_edges": [...],       # (author, reposter) のエッジ
        "params": {...},              # snowball_spin のパラメータ＋タイミング
      }
    """

    # ---------------------------------
    # ロガーのセットアップ
    # ---------------------------------
    if logger is None:
        def _log(msg: str) -> None:
            print(msg)
    else:
        _log = logger

    timings: Dict[str, float] = {}

    _log(
        f"[SPIN] run_spin_snowball_for_bluesky START "
        f"({start_day.isoformat()} → {end_day.isoformat()})"
    )

    # ---------------------------------
    # 1) seed accounts を決める
    # ---------------------------------
    t0 = time.perf_counter()

    if seed_accounts_override is not None:
        seed_accounts = list(seed_accounts_override)
        _log(f"[SPIN] Using override seed accounts: {len(seed_accounts)} users")
    else:
        _log(
            "[SPIN] Selecting seed accounts via Elasticsearch "
            f"(min_reposts_for_seed={min_reposts_for_seed}, top_n_seed={top_n_seed})..."
        )
        seed_accounts = select_seed_accounts_by_max_reposts(
            es,
            post_index=post_index,
            start_day=start_day,
            end_day=end_day,
            min_reposts_for_seed=min_reposts_for_seed,
            top_n_seed=top_n_seed,
            text_query=text_query_for_seed,
            author_field="did",
            created_at_field="commit.record.createdAt",
            repost_count_field="repost_count",
            text_field="commit.record.text",
        )
        _log(f"[SPIN] Seed selection done. Found {len(seed_accounts)} accounts.")

    timings["seed_selection_sec"] = time.perf_counter() - t0

    if not seed_accounts:
        _log("[SPIN] ERROR: seed accounts not found.")
        raise RuntimeError("seed accounts が見つかりませんでした。")

    # ---------------------------------
    # 2) フェッチャ／リポスター取得関数を作る
    # ---------------------------------
    t1 = time.perf_counter()

    fetch_publications = make_bluesky_publication_fetcher(
        es,
        post_index=post_index,
        author_field="did",
        created_at_field="commit.record.createdAt",
        repost_count_field="repost_count",
    )

    get_reposters = make_bluesky_reposter_getter(
        es,
        repost_index=repost_index,
        subject_cid_field="commit.record.subject.cid",
        reposter_field="did",
    )

    timings["make_fetchers_sec"] = time.perf_counter() - t1
    _log(
        f"[SPIN] Fetchers ready "
        f"(elapsed={timings['make_fetchers_sec']:.2f} sec)."
    )

    # ---------------------------------
    # 3) Snowball-SPIN 本体
    # ---------------------------------
    t2 = time.perf_counter()

    _log(
        "[SPIN] Running snowball_spin "
        f"(number_of_publications={number_of_publications}, "
        f"number_of_recursions={number_of_recursions}, "
        f"require_reposts_min={require_reposts_min}, "
        f"seed={seed})..."
    )

    common_list, sampled_edges, params = snowball_spin(
        accounts=seed_accounts,
        start_day=start_day,
        end_day=end_day,
        fetch_publications=fetch_publications,
        get_reposters=get_reposters,
        number_of_publications=number_of_publications,
        number_of_recursions=number_of_recursions,
        require_reposts_min=require_reposts_min,
        seed=seed,
        debug=True,
    )

    timings["snowball_spin_sec"] = time.perf_counter() - t2

    _log(
        f"[SPIN] snowball_spin finished in {timings['snowball_spin_sec']:.2f} sec. "
        f"Sampled users={len(common_list)}, edges={len(sampled_edges)}."
    )

    # ---------------------------------
    # 4) params にメタ情報としてタイミングを保存
    # ---------------------------------
    if isinstance(params, dict):
        meta = params.setdefault("meta", {})
        meta["timing_run_spin_snowball_for_bluesky"] = timings
        meta["n_seed_accounts"] = len(seed_accounts)

    _log("[SPIN] run_spin_snowball_for_bluesky DONE.")

    return {
        "seed_accounts": seed_accounts,
        "common_list": common_list,
        "sampled_edges": sampled_edges,
        "params": params,
    }

