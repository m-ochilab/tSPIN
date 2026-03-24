# topic_pipeline_tokenizer_updated_en.py
import re
import numpy as np
import pandas as pd
import torch
import spacy
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from typing import Optional, Any, List, Dict

CUSTOM_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "its", "if", "else", 
    "but", "not", "no", "so", "do", "did", "does", "doing", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "their", "our",
    "rt", "via", "amp", "http", "https", "www", "com", "co"
}

try:
    nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ---------------------------------------------------------
# ★ 高速化ポイント: 事前にDataFrame全体を並列処理(n_process=-1)でトークン化する
# ---------------------------------------------------------
def pre_tokenize_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    print("[BERTopic] Starting high-speed parallel tokenization with SpaCy...")
    texts = df[text_col].astype(str).tolist()
    
    # 事前ノイズ除去
    clean_texts = [re.sub(r"https?://\S+|www\.\S+|@\w+|#", " ", t) for t in texts]
    
    processed_texts = []
    # マルチコア(n_process=-1)で一気に処理
    for doc in nlp_en.pipe(clean_texts, n_process=-1, batch_size=1000):
        tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_punct and not token.is_space and not token.is_digit
            and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
            and len(token.lemma_) >= 2 
            and token.lemma_.lower() not in CUSTOM_STOPWORDS
        ]
        processed_texts.append(" ".join(tokens))
        
    df_out = df.copy()
    df_out["pre_tokenized_text"] = processed_texts
    return df_out

def build_english_bertopic_model() -> BERTopic:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    
    # ★ 高速化ポイント: SpaCy関数を呼ばず、空白区切りの簡易Tokenizerに変更（事前に処理済みのため）
    vectorizer_model = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=2)
    
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True,
    )

def run_bertopic_on_posts(posts_df: pd.DataFrame, text_col: str = "text", sample_size: int = 150000) -> tuple[pd.DataFrame, BERTopic]:
    # 1. 事前トークン化（超高速）
    posts_df_processed = pre_tokenize_dataframe(posts_df, text_col)
    
    # 2. モデル構築
    topic_model = build_english_bertopic_model()
    docs = posts_df_processed["pre_tokenized_text"].tolist() 

    # ★ 修正: データが膨大な場合、サンプリングして学習し、全体をTransform(推論)する
    if len(docs) > sample_size:
        print(f"\n[BERTopic] Dataset is massive ({len(docs)} posts).")
        print(f"[BERTopic] Training UMAP/HDBSCAN on a random sample of {sample_size} posts...")
        
        # 15万件をランダム抽出してモデルを学習（Fit）
        train_df = posts_df_processed.sample(n=sample_size, random_state=42)
        train_docs = train_df["pre_tokenized_text"].tolist()
        topic_model.fit(train_docs)
        
        # 学習した空間モデルを使って、全400万件のデータを推論・分類（Transform）
        print(f"[BERTopic] Training complete! Now transforming all {len(docs)} posts...")
        topics, probs = topic_model.transform(docs)
    else:
        # データが少なければ通常通り一括処理
        topics, probs = topic_model.fit_transform(docs)

    posts_df_processed["topic_id"] = topics
    posts_df_processed["topic_probas"] = list(probs)

    return posts_df_processed, topic_model

# compute_user_topic_profiles と attach_topics_to_nodes は変更なしのため省略(元のまま)
def compute_user_topic_profiles(posts_df: pd.DataFrame, user_col: str = "id", probas_col: str = "topic_probas", topic_threshold: float = 0.1, topic_model: Optional[BERTopic] = None) -> pd.DataFrame:
    valid_posts = posts_df.dropna(subset=[probas_col]).copy()
    def _to_array(x):
        if isinstance(x, np.ndarray): return x
        return np.array(x, dtype=float)
    valid_posts["topic_probas_arr"] = valid_posts[probas_col].apply(_to_array)
    if valid_posts.empty: return pd.DataFrame(columns=[user_col, "topic_vector", "user_topic_ids", "user_topic_labels"])
    sample_vec = valid_posts["topic_probas_arr"].iloc[0]
    n_topics = len(sample_vec)
    user2sum, user2cnt = {}, {}
    for row in valid_posts.itertuples():
        uid = getattr(row, user_col)
        vec = row.topic_probas_arr
        if uid not in user2sum:
            user2sum[uid] = np.zeros(n_topics, dtype=float)
            user2cnt[uid] = 0
        user2sum[uid] += vec
        user2cnt[uid] += 1
    records = []
    for uid, svec in user2sum.items():
        mean_vec = svec / max(user2cnt[uid], 1)
        topic_ids = np.where(mean_vec >= topic_threshold)[0].tolist()
        topic_labels = [f"Topic {tid}" for tid in topic_ids] if topic_model else None
        records.append({user_col: uid, "topic_vector": mean_vec, "user_topic_ids": topic_ids, "user_topic_labels": topic_labels})
    return pd.DataFrame(records)

def attach_topics_to_nodes(nodes_df: pd.DataFrame, user_topic_df: pd.DataFrame, user_col: str = "id") -> pd.DataFrame:
    return nodes_df.merge(user_topic_df[[user_col, "topic_vector", "user_topic_ids", "user_topic_labels"]], on=user_col, how="left")

def english_sns_tokenizer(text: str) -> list[str]:
    # 互換性のため残す
    return text.split()