
"""
snowball_spin.py
----------------
Faithful implementation of SPIN (Muñoz et al., EPJ Data Science 2024) Algorithm 1
with **explicit date range inputs** (start_day, end_day).
"""

from __future__ import annotations
from datetime import datetime
from typing import Callable, Iterable, List, Dict, Any, Sequence, Tuple, Optional
import random

Publication = Dict[str, Any]

def snowball_spin(
    *,
    accounts: Sequence[str],
    start_day: datetime,
    end_day: datetime,
    fetch_publications: Callable[[str, datetime, datetime], Iterable[Publication]],
    get_reposters: Callable[[Publication], Sequence[str]],
    number_of_publications: int = 8,
    number_of_recursions: int = 2,
    pub_id_key: str = "id",
    require_reposts_min: int = 1,  # ★ ここからは repost_count フィールドではなく len(reposters) に効く
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    debug: bool = False,
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, Any]]:
    """
    SPIN (Muñoz et al., 2024) Algorithm 1 に相当する本体。
    postindex-* に repost_count フィールドが無くても動くように、
    「実際に得られたリポスター数 len(reposters) >= require_reposts_min」
    でフィルタする。
    """
    _rng = rng or random.Random(seed)

    common_list: List[str] = []
    sampled_edges: List[Tuple[str, str]] = []

    # 各ステップのサンプリングログを記録
    sampling_log: List[Dict[str, Any]] = []

    def _log_step(level: int, from_user: str, pub: Publication,
                  reposters: Sequence[str], chosen: Optional[str]) -> None:
        if not debug:
            return
        sampling_log.append(
            {
                "level": level,
                "from_user": from_user,
                "pub_id": pub.get(pub_id_key),
                "pub_author": pub.get("author"),
                # postindex-* に repost_count がないので、ここはあれば使う・なければ None
                "pub_repost_count_field": pub.get("repost_count"),
                "n_reposters": len(reposters),
                "chosen_reposter": chosen,
                "created_at": pub.get("created_at"),
            }
        )
        print(
            f"[SPIN] level={level} from={from_user} pub={pub.get(pub_id_key)} "
            f"n_reposters={len(reposters)} chosen={chosen}"
        )

    def _sample_publications(author: str) -> List[Publication]:
        """
        ある著者の期間内の投稿を取得して、
        最大 number_of_publications 件だけランダムサンプリングする。
        ※ここでは repost_count によるフィルタは一切しない
        """
        pubs = list(fetch_publications(author, start_day, end_day))
        if debug:
            print(f"[SPIN] author={author}: total {len(pubs)} publications fetched")
        if not pubs:
            return []
        k = min(number_of_publications, len(pubs))
        return _rng.sample(pubs, k)

    # Level 0: seed accounts からスタート
    level = 0
    for account in accounts:
        pubs = _sample_publications(account)
        for pub in pubs:
            reposters = list(get_reposters(pub))

            # ★ ここで「実際のリポスター数」によるフィルタ
            if len(reposters) < require_reposts_min:
                if debug:
                    _log_step(level, account, pub, reposters, None)
                continue

            # 論文と同様、1ユーザだけランダムに選ぶ
            reposter = _rng.choice(reposters)
            common_list.append(reposter)
            sampled_edges.append((account, reposter))
            _log_step(level, account, pub, reposters, reposter)

    # Recursion levels
    for r in range(1, number_of_recursions + 1):
        level = r
        newly_retrieved = list(common_list)  # Alg.1 に合わせてコピーを使う
        if debug:
            print(f"[SPIN] ===== Recursion level {r}: {len(newly_retrieved)} users =====")
        for user in newly_retrieved:
            pubs = _sample_publications(user)
            for pub in pubs:
                reposters = list(get_reposters(pub))

                if len(reposters) < require_reposts_min:
                    if debug:
                        _log_step(level, user, pub, reposters, None)
                    continue

                reposter = _rng.choice(reposters)
                common_list.append(reposter)
                sampled_edges.append((user, reposter))
                _log_step(level, user, pub, reposters, reposter)

    params = dict(
        start_day=start_day.isoformat(),
        end_day=end_day.isoformat(),
        number_of_publications=number_of_publications,
        number_of_recursions=number_of_recursions,
        require_reposts_min=require_reposts_min,
        pub_id_key=pub_id_key,
        seed=seed,
        accounts=list(accounts),
        sampling_log=sampling_log,
    )
    return common_list, sampled_edges, params

# ---------- Optional Elasticsearch helpers ----------
def make_bluesky_publication_fetcher(
    es: Elasticsearch,
    *,
    post_index: str = "postindex-*",
    author_field: str = "did",
    created_at_field: str = "commit.record.createdAt",
    size_limit: int = 2000,
) -> Callable[[str, datetime, datetime], List[Publication]]:
    """
    snowball_spin() に渡す fetch_publications(author, start, end) を生成。

    - ESの postindex-* から、「その著者が期間内に書いた app.bsky.feed.post」を取得する。
    - pub["cid"] には commit.cid を入れる（repostindex 側の subject.cid と結びつける）。
    """
    def _fetch(author: str, start_dt: datetime, end_dt: datetime) -> List[Publication]:
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {author_field: author}},
                        {"term": {"commit.collection": "app.bsky.feed.post"}},
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

            # ★ ここが重要: post の CID は commit.cid から取る
            cid = commit.get("cid")

            pubs.append(
                {
                    "id": h.get("_id"),
                    "cid": cid,
                    "author": src.get(author_field),
                    "created_at": record.get("createdAt"),
                    # repost_count は ES には無いのでダミーで 0 を入れておく（使わない）
                    "repost_count": 0,
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
    size_limit: int = 2000,
) -> Callable[[Publication], List[str]]:
    """
    snowball_spin() に渡す get_reposters(publication) を生成。

    - publication["cid"] をキーにして、repostindex-* の
      commit.record.subject.cid と一致するものを探し、
      その did をリポスターとして返す。
    """
    def _get_reposters(pub: Publication) -> List[str]:
        cid = pub.get("cid")
        if not cid:
            return []

        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {subject_cid_field: cid}},
                        {"term": {"commit.collection": "app.bsky.feed.repost"}},
                    ]
                }
            },
            "size": size_limit,
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
