# Calculate_NNIF.py
# NNIF 計算用の「情報フロー・ネットワーク」構築関数をまとめたモジュール
# - repost / reply / quote / mention からユーザ間エッジを作る
# - ここでは NNIF 本体の計算はまだ置かず、「エッジ構築」までを担当

from collections import Counter
from typing import Optional, Collection, Tuple, Dict, List, Set

import pandas as pd

# 既に research_matae.py にある CID -> 著者 DID 変換ヘルパーを再利用
# （片方向の import なので循環参照は起こらない）
from research_matae_tokenizer_updated import _map_cids_to_authors
from negative_judge import is_negative_text

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
                # (元ポストCID, リポストしたユーザ DID)
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
                # (親ポストCID, 返信したユーザ DID)
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

    ※ Bluesky の ES マッピングに依存:
      例として:
        commit.record.embed.record.cid に引用先ポストの CID が入っている想定。
      実際の mapping に合わせて field 名は調整してください。
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
                # (引用元CID, 引用したユーザ DID)
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

    ※ これも ES のマッピングに依存:
      - commit.record.mentions[].did があればそれを利用。
      - なければ text から @handle をパースし、別途 handle->DID を解決する必要あり。
    ここでは「mentions[].did がある」前提のシンプル版。
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
                    # (書いたユーザ DID, メンションされたユーザ DID)
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
    check_content_negativity: bool = False,  # ★論文の手法に合わせるための新フラグ
) -> pd.DataFrame:
    """
    SPIN / NNIF 用の「情報フロー・ネットワーク」のエッジ集合を構成する。
    check_content_negativity=True の場合、インタラクション元の投稿がネガティブな場合のみエッジを採用する。
    """
    if allowed_users is not None:
        allowed_set = {str(u) for u in allowed_users}
    else:
        allowed_set = None

    # 各種インタラクションの取得
    print("[IF] Fetching raw interaction pairs...")
    repost_pairs = _iter_reposts_in_period(es, repost_index=repost_index, start_dt=start_dt, end_dt=end_dt)
    reply_pairs = _iter_replies_in_period(es, post_index=post_index, start_dt=start_dt, end_dt=end_dt)
    quote_pairs = _iter_quotes_in_period(es, post_index=post_index, start_dt=start_dt, end_dt=end_dt)
    mention_pairs = _iter_mentions_in_period(es, post_index=post_index, start_dt=start_dt, end_dt=end_dt)

    # フィルタリング用のCIDセットを作成
    valid_cids = set()
    
    if check_content_negativity:
        print("[IF] Checking content negativity for all CIDs...")
        # Repost, Reply, Quote の対象CIDを収集
        all_cids = [p[0] for p in repost_pairs] + [p[0] for p in reply_pairs] + [p[0] for p in quote_pairs]
        # テキストを取得して判定
        valid_cids = get_negative_cids(es, all_cids, post_index=post_index)
        print(f"[IF] Found {len(valid_cids)} negative CIDs out of {len(set(all_cids))} unique interactions.")
    else:
        # チェックしない場合は全てのCIDを有効とする（セットに含めなくてもロジックでスキップさせる）
        pass

    edge_counter: Counter[Tuple[str, str, str]] = Counter()

    # --- 1) Repost ---
    target_reposts = []
    for cid, reposter in repost_pairs:
        if check_content_negativity and cid not in valid_cids:
            continue
        target_reposts.append(cid)
        
    if target_reposts:
        cid_to_author = _map_cids_to_authors(es, target_reposts, post_index=post_index)
        for cid, reposter in repost_pairs:
            if check_content_negativity and cid not in valid_cids: continue
            
            author = cid_to_author.get(cid)
            if not author: continue
            
            src, dst = str(author), str(reposter)
            if src == dst: continue
            if allowed_set and (src not in allowed_set or dst not in allowed_set): continue
            
            edge_counter[(src, dst, "repost")] += 1

    # --- 2) Reply ---
    target_replies = []
    for cid, replier in reply_pairs:
        if check_content_negativity and cid not in valid_cids:
            continue
        target_replies.append(cid)
        
    if target_replies:
        cid_to_author_reply = _map_cids_to_authors(es, target_replies, post_index=post_index)
        for cid, replier in reply_pairs:
            if check_content_negativity and cid not in valid_cids: continue
            
            author = cid_to_author_reply.get(cid)
            if not author: continue
            
            src, dst = str(author), str(replier)
            if src == dst: continue
            if allowed_set and (src not in allowed_set or dst not in allowed_set): continue
            
            edge_counter[(src, dst, "reply")] += 1

    # --- 3) Quote ---
    target_quotes = []
    for cid, quoter in quote_pairs:
        if check_content_negativity and cid not in valid_cids:
            continue
        target_quotes.append(cid)
        
    if target_quotes:
        cid_to_author_quote = _map_cids_to_authors(es, target_quotes, post_index=post_index)
        for cid, quoter in quote_pairs:
            if check_content_negativity and cid not in valid_cids: continue
            
            author = cid_to_author_quote.get(cid)
            if not author: continue
            
            src, dst = str(author), str(quoter)
            if src == dst: continue
            if allowed_set and (src not in allowed_set or dst not in allowed_set): continue
            
            edge_counter[(src, dst, "quote")] += 1

    # --- 4) Mention (テキスト判定が難しいため、全体のネガティブチェックがONならスキップするか、別途判定が必要) ---
    # ここでは、メンションはテキストそのものに含まれるため、投稿自体がネガティブなら採用するロジックにするのが理想だが、
    # 簡易化のため「チェックONならメンションはエッジに含めない」または「スルー」とする。
    # 論文では「リポストから相互作用ネットワークを構築」とあるため、メンションは優先度が低い。
    if not check_content_negativity:
        for author, mentioned in mention_pairs:
            src, dst = str(author), str(mentioned)
            if src == dst: continue
            if allowed_set and (src not in allowed_set or dst not in allowed_set): continue
            edge_counter[(src, dst, "mention")] += 1

    # --- DataFrame化 ---
    rows = [
        {"src": s, "dst": t, "kind": k, "weight": int(w)}
        for (s, t, k), w in edge_counter.items()
    ]
    edges_df = pd.DataFrame(rows, columns=["src", "dst", "kind", "weight"])
    return edges_df

# Calculate_NNIF.py の続きに追記

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Collection
import pandas as pd


def build_user_timelines(
    es,
    *,
    post_index: str = "postindex-*",
    start_dt=None,
    end_dt=None,
    allowed_users: Optional[Collection[str]] = None,
    min_posts: int = 1,
    batch_size: int = 5000,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    ユーザごとに「時系列の投稿列（タイムライン）」を構築する。

    戻り値:
      user_timelines:
        {
          user_did: [
            (created_at_iso, text),
            (created_at_iso, text),
            ...
          ],
          ...
        }

    注意:
      - created_at は ISO8601 文字列のまま返す（ISO なら文字列ソートで時間順になる）
      - h(T|S) 計算では「順序」が重要なので、必ず時系列順にソートする
      - allowed_users を指定すると、そのユーザだけを対象にする
    """
    if allowed_users is None:
        raise ValueError("build_user_timelines: allowed_users は None ではいけません。")

    allowed_set = {str(u) for u in allowed_users}

    # ユーザごとの投稿を一時的に貯める
    tmp_posts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # -----------------------------------------
    # 全投稿からまとめて拾う方式
    #   - 期間 + collection=post で絞り込み
    #   - did が allowed_users に含まれているものだけ使う
    # -----------------------------------------
    query: Dict = {
        "size": batch_size,
        "_source": [
            "did",
            "commit.collection",
            "commit.record.text",
            "commit.record.createdAt",
        ],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.collection": "app.bsky.feed.post"}},
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
            if not isinstance(did, str):
                continue
            if did not in allowed_set:
                # SPIN 対象ユーザ以外は無視
                continue

            commit = src.get("commit") or {}
            record = commit.get("record") or {}
            text = record.get("text")
            created_at = record.get("createdAt")

            if not isinstance(text, str) or not isinstance(created_at, str):
                continue

            # created_at は ISO8601 文字列をそのまま保持
            tmp_posts[did].append((created_at, text))

        search_after = hits[-1].get("sort")
        if search_after is None:
            break

    # -----------------------------------------
    # ユーザごとに created_at でソート & min_posts でフィルタ
    # -----------------------------------------
    user_timelines: Dict[str, List[Tuple[str, str]]] = {}

    for did, posts in tmp_posts.items():
        # created_at 文字列でソート（ISO8601前提）
        posts_sorted = sorted(posts, key=lambda x: x[0])
        if len(posts_sorted) >= min_posts:
            user_timelines[did] = posts_sorted

    return user_timelines


def build_neighbors_from_edges(
    edges_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    情報フロー・ネットワーク（edges_df）から、
    ユーザごとの「近傍ノード集合」を構築する。

    - edges_df は build_information_flow_edges() の戻り値を想定
      列: ["src", "dst", "kind", "weight"]

    ここでは無向グラフとして近傍を定義し、
    src->dst のエッジがあれば
      - neighbors[src] に dst
      - neighbors[dst] に src
    の両方を追加する。

    戻り値:
      neighbors:
        {
          user_did: [neighbor_did1, neighbor_did2, ...],
          ...
        }
    """
    neighbors_set: Dict[str, set] = defaultdict(set)

    for _, row in edges_df.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        if not src or not dst or src == dst:
            continue

        neighbors_set[src].add(dst)
        neighbors_set[dst].add(src)

    # set -> list 変換して返す（順序は一応ソートしておく）
    neighbors: Dict[str, List[str]] = {
        node: sorted(list(nbrs)) for node, nbrs in neighbors_set.items()
    }
    return neighbors

# ==== ここから Calculate_NNIF.py に追記 ==============================

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Collection
import pandas as pd


# ---------------------------------------------------------
# 1) ユーザごとのトークン列を準備する
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1) ユーザごとのトークン列を準備する
# ---------------------------------------------------------

from typing import Callable

# NOTE:
#   ここは NNIF / h(T|S) の入力になる「トークン列」を作る最重要部分です。
#   空白 split は日本語・短文SNS・絵文字/URL/ハッシュタグに弱いので、
#   research_matae.py の create_tokenizer_latest()（spaCy→GiNZA→Sudachi→簡易の順）を利用します。
#
#   - 日本語: spaCy(JA) or SudachiPy があれば形態素レベルで分割
#   - 英語/ラテン文字: spaCy(EN) があれば lemma/品詞で分割
#   - どれも無ければ正規表現ベースでフォールバック（環境依存を最小化）
#
#   ※ max_tokens_per_user を設定すると、後段の計算量を確実に抑えられます。

try:
    # 既にこのファイルで research_matae を import しているため、追加 import でも循環参照は起きません
    from research_matae import create_tokenizer_latest  # type: ignore
except Exception:
    create_tokenizer_latest = None  # type: ignore


def build_user_token_sequences(
    user_timelines: Dict[str, List[Tuple[str, str]]],
    max_tokens_per_user: Optional[int] = None,
    *,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    build_user_timelines() の出力（user -> [(created_at, text), ...]）から、
    NNIF / h(T|S) 計算で使うためのトークン列を構築する。

    戻り値:
      user_token_seqs:
        {
          user_did: (
            tokens: [tok1, tok2, ...],
            times:  [t1,   t2,   ...],  # 各トークンに対応する created_at (ISO8601文字列)
          ),
          ...
        }

    max_tokens_per_user:
      各ユーザごとのトークン数の上限。
      None の場合は制限なし。計算コストが大きい場合は 2000〜5000 程度に制限推奨。

    tokenizer:
      文字列 -> トークン列 を返す関数。
      None の場合は create_tokenizer_latest(lang="auto") を使う。
    """
    if tokenizer is None:
        if create_tokenizer_latest is not None:
            # NNIF は「ユーザ間で同じ表現が繰り返される」ほど一致しやすいので lemma は有効なことが多い
            tokenizer = create_tokenizer_latest(
                lang="auto",
                use_lemma=True,
                min_len=1,      # 短文SNSでは 1 文字も残したいケースがある（例: "米", "EU"）
                max_len=64,
                allow_download=False,
            )
        else:
            # research_matae が import できない場合の最終フォールバック
            tokenizer = lambda s: (s or "").split()

    user_token_seqs: Dict[str, Tuple[List[str], List[str]]] = {}

    for did, posts in user_timelines.items():
        tokens: List[str] = []
        times: List[str] = []

        for created_at, text in posts:
            toks = tokenizer(text)
            for tok in toks:
                tokens.append(tok)
                times.append(created_at)
                if max_tokens_per_user is not None and len(tokens) >= max_tokens_per_user:
                    break
            if max_tokens_per_user is not None and len(tokens) >= max_tokens_per_user:
                break

        if tokens:
            user_token_seqs[did] = (tokens, times)

    return user_token_seqs

# ---------------------------------------------------------
# 2) 時間同期クロスエントロピー h(T|S)
# ---------------------------------------------------------

def _longest_match_length(
    target_tokens: List[str],
    source_tokens: List[str],
    start_idx: int,
    max_src_idx: int,
    max_match_len: int,
) -> int:
    """
    target_tokens[start_idx:] の先頭から、
    source_tokens[0:max_src_idx+1] のどこかに現れる
    「最長の連続一致部分列の長さ」を返す。

    max_match_len で最大長を制限し、計算量の爆発を抑える。

    返り値:
      L_i >= 1 を保証する（全くマッチがなくても 1 とする）。
    """
    best = 0
    # target 側でこれ以上伸ばせない長さ
    max_l = min(max_match_len, len(target_tokens) - start_idx)

    for l in range(1, max_l + 1):
        subseq = target_tokens[start_idx:start_idx + l]
        found = False

        limit_j = max_src_idx - l + 1
        if limit_j < 0:
            break

        # source_tokens のプレフィックス内を総当たりで探索（素朴実装）
        for j in range(0, limit_j + 1):
            if source_tokens[j:j + l] == subseq:
                found = True
                break

        if found:
            best = l
        else:
            # l 長のマッチが見つからなかったら、これ以上伸ばしても無理なので break
            break

    if best == 0:
        best = 1
    return best

def get_h(
    target: str,
    source: str,
    user_token_seqs: Dict[str, Tuple[List[str], List[str]]],
    h_cache: Dict[Tuple[str, str], float],
    max_match_len: int,
) -> float:
    """
    h(target | source) をキャッシュ付きで返す
    """
    key = (target, source)
    if key in h_cache:
        return h_cache[key]

    if target not in user_token_seqs or source not in user_token_seqs:
        h_cache[key] = float("inf")
        return h_cache[key]

    tgt_tokens, tgt_times = user_token_seqs[target]
    src_tokens, src_times = user_token_seqs[source]

    h_val = timesync_cross_entropy_from_tokens(
        target_tokens=tgt_tokens,
        target_times=tgt_times,
        source_tokens=src_tokens,
        source_times=src_times,
        max_match_len=max_match_len,
    )

    h_cache[key] = h_val
    return h_val


def timesync_cross_entropy_from_tokens(
    target_tokens: List[str],
    target_times: List[str],
    source_tokens: List[str],
    source_times: List[str],
    max_match_len: int = 20,
) -> float:
    """
    時間同期クロスエントロピー h(T|S) を、トークン列レベルで推定する。

    引数:
      target_tokens, target_times:
        ターゲット T のトークン列と、各トークンに対応する createdAt (ISO文字列)。
      source_tokens, source_times:
        ソース S のトークン列と、各トークンに対応する createdAt (ISO文字列)。
      max_match_len:
        Λ_i を探索するときの最大マッチ長。大きくすると精度は上がるが計算量が増える。

    返り値:
      h(T|S) の推定値（小さいほど「S->T の情報フローが強い」）。
      片方が空の場合などは、無限大 (float('inf')) を返す。
    """
    N_T = len(target_tokens)
    N_S = len(source_tokens)

    if N_T == 0 or N_S == 0:
        return float("inf")

    # 時刻に関する前提:
    # - target_times, source_times ともに ISO8601 形式
    # - 文字列比較で時系列順が保たれる（build_user_timelines の実装を前提）

    # まず「各 target トークン i について、どこまでの source を使えるか」を前計算
    # prefix_end_for_i[i] = target i の時刻までに出現した source トークンの最大インデックス
    prefix_end_for_i: List[int] = []
    prefix_end = -1
    s_idx = 0

    for i in range(N_T):
        t_time = target_times[i]
        # source_times は昇順と仮定
        while s_idx < N_S and source_times[s_idx] <= t_time:
            prefix_end = s_idx
            s_idx += 1
        prefix_end_for_i.append(prefix_end)

    # Λ_i を合計
    sum_L = 0
    for i in range(N_T):
        max_src_idx = prefix_end_for_i[i]
        if max_src_idx < 0:
            # この時点までに source が一度も投稿していない
            L_i = 1
        else:
            L_i = _longest_match_length(
                target_tokens=target_tokens,
                source_tokens=source_tokens,
                start_idx=i,
                max_src_idx=max_src_idx,
                max_match_len=max_match_len,
            )
        sum_L += L_i

    if sum_L <= 0:
        return float("inf")

    # 論文の形を簡略化した近似:
    #   h(T|S) ≈ (N_T * log2(N_S)) / sum_i Λ_i
    # log2(0) を避けるために max(N_S, 2) とする
    h = (N_T * math.log2(max(N_S, 2))) / float(sum_L)
    return h

# ---------------------------------------------------------
# 3) NNIF(S, T) と、エッジ全体への適用
# ---------------------------------------------------------

def compute_nnif_for_pair(
    source: str,
    target: str,
    user_token_seqs: Dict[str, Tuple[List[str], List[str]]],
    neighbors: Dict[str, List[str]],
    h_cache: Optional[Dict[Tuple[str, str], float]] = None,
    max_match_len: int = 20,
) -> Optional[float]:
    """
    1 つのノードペア (S, T) について NNIF(S, T) を計算する。

    引数:
      source, target:
        ユーザ DID （S, T）。
      user_token_seqs:
        build_user_token_sequences の出力。
      neighbors:
        build_neighbors_from_edges の出力（無向近傍）。
      h_cache:
        (target, source) -> h(target|source) をキャッシュする辞書。
        複数ペアを計算する場合、外で dict を作って渡すと高速化できる。
      max_match_len:
        h(T|S) 計算時の最大マッチ長（timesync_cross_entropy_from_tokens の引数）。

    戻り値:
      NNIF(S, T) の値（float）。
      必要な情報が足りない場合（トークン列なしなど）は None。
    """
    print(f"[NNIF][PAIR] start {source} -> {target}")

    if h_cache is None:
        h_cache = {}

    # h(T|S), h(S|T)
    h_TS = get_h(target, source, user_token_seqs, h_cache, max_match_len)
    h_ST = get_h(source, target, user_token_seqs, h_cache, max_match_len)

    print(f"[NNIF][PAIR] h({target}|{source})={h_TS}")
    print(f"[NNIF][PAIR] h({source}|{target})={h_ST}")

    if math.isinf(h_TS) or math.isinf(h_ST):
        print("[NNIF][PAIR] inf が含まれるため return None")
        return None

    denom_T = 0.0
    for x in neighbors.get(target, []):
        hx = get_h(target, x, user_token_seqs, h_cache, max_match_len)
        if not math.isinf(hx):
            denom_T += hx

    denom_S = 0.0
    for x in neighbors.get(source, []):
        hx = get_h(source, x, user_token_seqs, h_cache, max_match_len)
        if not math.isinf(hx):
            denom_S += hx

    print(f"[NNIF][PAIR] denom_T={denom_T}, denom_S={denom_S}")

    if denom_T <= 0.0 or denom_S <= 0.0:
        print("[NNIF][PAIR] 分母ゼロで return None")
        return None

    nnif_val = (h_TS / denom_T) - (h_ST / denom_S)
    print(f"[NNIF][PAIR] NNIF={nnif_val}")

    return nnif_val

def compute_nnif_for_edges(
    edges_df: pd.DataFrame,
    user_token_seqs: Dict[str, Tuple[List[str], List[str]]],
    neighbors: Dict[str, List[str]],
    max_match_len: int = 20,
) -> pd.DataFrame:

    pairs = set()
    for _, row in edges_df.iterrows():
        s = str(row["src"])
        t = str(row["dst"])
        if s and t and s != t:
            pairs.add((s, t))

    print(f"[NNIF][CORE] NNIF 計算開始: ペア数={len(pairs)}")

    h_cache: Dict[Tuple[str, str], float] = {}
    rows = []

    for i, (s, t) in enumerate(pairs, 1):
        if i == 1 or i % 10 == 0:
            print(f"[NNIF][CORE] 進捗 {i}/{len(pairs)} : {s} -> {t}")

        nnif_val = compute_nnif_for_pair(
            source=s,
            target=t,
            user_token_seqs=user_token_seqs,
            neighbors=neighbors,
            h_cache=h_cache,
            max_match_len=max_match_len,
        )

        if nnif_val is None:
            print(f"[NNIF][CORE] skip {s}->{t} (計算不可)")
            continue

        rows.append({"src": s, "dst": t, "nnif": float(nnif_val)})

    print(f"[NNIF][CORE] NNIF 計算完了: 成功={len(rows)} ペア")

    return pd.DataFrame(rows, columns=["src", "dst", "nnif"])

def _map_cids_to_text(
    es,
    cids: List[str],
    *,
    post_index: str = "postindex-*",
    chunk_size: int = 1000,
    request_timeout: int = 30,
) -> Dict[str, str]:
    """
    CIDのリストからテキスト内容を一括取得するヘルパー関数
    """
    uniq = list(set([c for c in cids if c]))
    out: Dict[str, str] = {}
    
    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i : i + chunk_size]
        body = {
            "size": len(chunk),
            "_source": ["commit.cid", "commit.record.text"],
            "track_total_hits": False,
            "query": {"bool": {"filter": [{"terms": {"commit.cid": chunk}}]}},
        }
        try:
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
                cid = (src.get("commit") or {}).get("cid")
                text = (src.get("commit") or {}).get("record", {}).get("text")
                if cid and text:
                    out[cid] = text
        except Exception as e:
            print(f"[WARN] Failed to fetch texts for chunk: {e}")
            continue
            
    return out

def get_negative_cids(es, cids: List[str], post_index: str = "postindex-*") -> Set[str]:
    """
    CIDリストの中で、テキストがネガティブ判定されるものの集合を返す
    """
    cid_text_map = _map_cids_to_text(es, cids, post_index=post_index)
    neg_cids = set()
    for cid, text in cid_text_map.items():
        if is_negative_text(text):
            neg_cids.add(cid)
    return neg_cids
