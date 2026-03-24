# Calculate_NNIF_tokenizer_updated_en.py
import math
import pandas as pd
import numpy as np
import multiprocessing
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Collection, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit # ★ 高速化のためのNumba

from research_matae_tokenizer_updated import _map_cids_to_authors
from negative_judge_en import check_strong_keyword, load_english_negative_judge

# =========================================================
# ★ Numba (JIT) による超高速アルゴリズム群
# Numbaは文字列を扱えないため、事前に整数(ID)化された配列を受け取ります。
# =========================================================
@njit(fastmath=True)
def _longest_match_length_numba(target_tokens, source_tokens, start_idx, max_src_idx, max_match_len):
    best = 0
    max_l = min(max_match_len, len(target_tokens) - start_idx)
    for l in range(1, max_l + 1):
        found = False
        limit_j = max_src_idx - l + 1
        if limit_j < 0: break
        for j in range(0, limit_j + 1):
            # 配列の比較ループ
            match = True
            for k in range(l):
                if source_tokens[j+k] != target_tokens[start_idx+k]:
                    match = False
                    break
            if match:
                found = True
                break
        if found: 
            best = l
        else: 
            break
    if best == 0: best = 1
    return best

@njit(fastmath=True)
def timesync_cross_entropy_numba(target_tokens, target_times, source_tokens, source_times, max_match_len=20):
    N_T = len(target_tokens)
    N_S = len(source_tokens)
    if N_T == 0 or N_S == 0: return np.inf
    
    sum_L = 0
    s_idx = 0
    
    for i in range(N_T):
        t_time = target_times[i]
        # タイムシンクのインデックス検索
        while s_idx < N_S and source_times[s_idx] <= t_time:
            s_idx += 1
        prefix_end = s_idx - 1
        
        if prefix_end < 0:
            sum_L += 1
        else:
            sum_L += _longest_match_length_numba(target_tokens, source_tokens, i, prefix_end, max_match_len)
            
    if sum_L <= 0: return np.inf
    h = (N_T * math.log2(max(N_S, 2))) / float(sum_L)
    return h

# =========================================================
# 並列処理用ワーカー
# =========================================================
def _compute_nnif_worker(source, target, user_token_seqs, target_neighbors, source_neighbors, max_match_len):
    def _local_get_h(t, s):
        if t not in user_token_seqs or s not in user_token_seqs:
            return float("inf")
        tgt_tokens, tgt_times = user_token_seqs[t]
        src_tokens, src_times = user_token_seqs[s]
        # ★ Numba関数を呼び出す
        return float(timesync_cross_entropy_numba(tgt_tokens, tgt_times, src_tokens, src_times, max_match_len))

    h_TS = _local_get_h(target, source)
    h_ST = _local_get_h(source, target)

    if math.isinf(h_TS) or math.isinf(h_ST):
        return source, target, None

    denom_T = 0.0
    for x in target_neighbors:
        hx = _local_get_h(target, x)
        if not math.isinf(hx): denom_T += hx

    denom_S = 0.0
    for x in source_neighbors:
        hx = _local_get_h(source, x)
        if not math.isinf(hx): denom_S += hx

    if denom_T <= 0.0 or denom_S <= 0.0:
        return source, target, None

    nnif_val = (h_TS / denom_T) - (h_ST / denom_S)
    return source, target, float(nnif_val)

# =========================================================
# メイン処理 (マルチコア並列化)
# =========================================================
def compute_nnif_for_edges(edges_df: pd.DataFrame, user_token_seqs: Dict[str, Tuple[np.ndarray, np.ndarray]], neighbors: Dict[str, List[str]], max_match_len: int = 20) -> pd.DataFrame:
    pairs = set()
    for _, row in edges_df.iterrows():
        s = str(row["src"])
        t = str(row["dst"])
        if s and t and s != t: pairs.add((s, t))

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"[NNIF][CORE] NNIF Calculation started with CPU Multiprocessing ({num_cores} cores): {len(pairs)} pairs")
    rows = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for s, t in pairs:
            tgt_nbrs = neighbors.get(t, [])
            src_nbrs = neighbors.get(s, [])
            futures.append(
                executor.submit(_compute_nnif_worker, s, t, user_token_seqs, tgt_nbrs, src_nbrs, max_match_len)
            )
            
        completed = 0
        for future in as_completed(futures):
            s, t, nnif_val = future.result()
            completed += 1
            if completed % 1000 == 0 or completed == len(pairs):
                print(f"[NNIF][CORE] Progress: {completed}/{len(pairs)}")
            if nnif_val is not None:
                rows.append({"src": s, "dst": t, "nnif": nnif_val})

    print(f"[NNIF][CORE] NNIF Calculation completed: Success={len(rows)} pairs")
    return pd.DataFrame(rows, columns=["src", "dst", "nnif"])


# =========================================================
# データ構築関数
# =========================================================
def build_user_token_sequences(user_timelines, max_tokens_per_user=None, *, tokenizer=None):
    if tokenizer is None: tokenizer = lambda s: (s or "").split()
    
    # ★ 高速化: Numba用に単語を整数IDに変換するための辞書
    vocab = {}
    def get_id(w):
        if w not in vocab: vocab[w] = len(vocab)
        return vocab[w]

    user_token_seqs = {}
    for did, posts in user_timelines.items():
        tokens, times = [], []
        for created_at, text in posts:
            toks = tokenizer(text)
            # 時間を float (epoch time) に変換してNumbaで比較できるようにする
            ts_float = pd.to_datetime(created_at).timestamp()
            
            for tok in toks:
                tokens.append(get_id(tok))
                times.append(ts_float)
                if max_tokens_per_user and len(tokens) >= max_tokens_per_user: break
            if max_tokens_per_user and len(tokens) >= max_tokens_per_user: break
            
        if tokens: 
            # numpyの型付き配列にして保存
            user_token_seqs[did] = (np.array(tokens, dtype=np.int32), np.array(times, dtype=np.float64))
            
    print(f"[NNIF Data] Token dictionary size: {len(vocab)} unique words.")
    return user_token_seqs

def build_user_timelines_negative_only(es, *, post_index="postindex-*", start_dt=None, end_dt=None, allowed_users=None, min_posts=1, batch_size=5000):
    if allowed_users is None: raise ValueError("allowed_users は必須です。")
    allowed_set = {str(u) for u in allowed_users}
    tmp_posts = defaultdict(list)
    query = {
        "size": batch_size,
        "_source": ["did", "commit.collection", "commit.record.text", "commit.record.createdAt"],
        "query": {"bool": {"filter": [{"term": {"commit.collection": "app.bsky.feed.post"}}]}},
        "sort": [{"commit.record.createdAt": "asc"}],
    }
    if start_dt and end_dt:
        query["query"]["bool"]["filter"].append(
            {"range": {"commit.record.createdAt": {"gte": start_dt.isoformat(), "lte": end_dt.isoformat()}}}
        )

    search_after = None
    print("[NNIF Data] Fetching posts from Elasticsearch...")
    while True:
        if search_after is not None: query["search_after"] = search_after
        res = es.search(index=post_index, body=query, request_timeout=120)
        hits = (res.get("hits") or {}).get("hits") or []
        if not hits: break

        for h in hits:
            src = h.get("_source") or {}
            did = src.get("did")
            if not isinstance(did, str) or did not in allowed_set: continue
            text = src.get("commit", {}).get("record", {}).get("text", "")
            created_at = src.get("commit", {}).get("record", {}).get("createdAt", "")
            if not text or not created_at: continue
            tmp_posts[did].append((created_at, text))

        search_after = hits[-1].get("sort")
        if search_after is None: break

    print("[NNIF AI] Starting high-speed batch sentiment inference on GPU...")
    analyzer = load_english_negative_judge()
    user_timelines = defaultdict(list)
    texts_for_ai = []
    metadata_for_ai = [] 

    for did, posts in tmp_posts.items():
        for created_at, text in posts:
            if check_strong_keyword(text):
                user_timelines[did].append((created_at, text))
            else:
                texts_for_ai.append(text)
                metadata_for_ai.append((did, created_at, text))

    if texts_for_ai:
        print(f"[NNIF AI] Passing {len(texts_for_ai)} posts to RoBERTa model...")
        ai_results = analyzer.predict_batch(texts_for_ai, batch_size=256) # バッチサイズを256に増やしてGPU効率UP
        for is_neg, (did, created_at, text) in zip(ai_results, metadata_for_ai):
            if is_neg: user_timelines[did].append((created_at, text))

    final_timelines = {}
    for did, posts in user_timelines.items():
        posts_sorted = sorted(posts, key=lambda x: x[0])
        if len(posts_sorted) >= min_posts:
            final_timelines[did] = posts_sorted

    return final_timelines

def build_neighbors_from_edges(edges_df: pd.DataFrame) -> Dict[str, List[str]]:
    neighbors_set = defaultdict(set)
    for _, row in edges_df.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        if not src or not dst or src == dst: continue
        neighbors_set[src].add(dst)
        neighbors_set[dst].add(src)
    return {node: sorted(list(nbrs)) for node, nbrs in neighbors_set.items()}

def _iter_reposts_in_period(es, *, repost_index="repostindex-*", start_dt=None, end_dt=None, batch_size=5000):
    query = {
        "size": batch_size,
        "_source": ["did", "commit.record.subject.cid", "commit.record.createdAt"],
        "query": {"bool": {"filter": [{"term": {"commit.collection": "app.bsky.feed.repost"}}]}},
        "sort": [{"commit.record.createdAt": "asc"}],
    }
    if start_dt and end_dt:
        query["query"]["bool"]["filter"].append({"range": {"commit.record.createdAt": {"gte": start_dt.isoformat(), "lte": end_dt.isoformat()}}})
    results = []
    search_after = None
    while True:
        if search_after: query["search_after"] = search_after
        res = es.search(index=repost_index, body=query, request_timeout=120)
        hits = (res.get("hits") or {}).get("hits") or []
        if not hits: break
        for h in hits:
            src = h.get("_source") or {}
            did = src.get("did")
            cid = (src.get("commit", {}).get("record", {}).get("subject", {})).get("cid")
            if did and cid: results.append((cid, did))
        search_after = hits[-1].get("sort")
        if not search_after: break
    return results

def build_information_flow_edges(es, *, post_index="postindex-*", repost_index="repostindex-*", start_dt=None, end_dt=None, allowed_users=None, check_content_negativity=False):
    allowed_set = {str(u) for u in allowed_users} if allowed_users else None
    print("[IF] Fetching raw interaction pairs...")
    
    # ★ リポスト (repost) のみを取得するようにスリム化！
    repost_pairs = _iter_reposts_in_period(es, repost_index=repost_index, start_dt=start_dt, end_dt=end_dt)
    target_reposts = [p[0] for p in repost_pairs]
    edge_counter = Counter()
    
    if target_reposts:
        cid_to_author = _map_cids_to_authors(es, target_reposts, post_index=post_index)
        for cid, reposter in repost_pairs:
            author = cid_to_author.get(cid)
            if not author: continue
            src, dst = str(author), str(reposter)
            if src == dst: continue
            if allowed_set and (src not in allowed_set or dst not in allowed_set): continue
            edge_counter[(src, dst, "repost")] += 1

    rows = [{"src": s, "dst": t, "kind": k, "weight": int(w)} for (s, t, k), w in edge_counter.items()]
    return pd.DataFrame(rows, columns=["src", "dst", "kind", "weight"])