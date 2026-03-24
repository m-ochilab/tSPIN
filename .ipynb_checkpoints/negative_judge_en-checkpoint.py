import re
import torch
import numpy as np
from typing import List, Set
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# =========================================================
# 設定: モデルと閾値
# =========================================================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NEGATIVE_THRESHOLD = 0.4

# =========================================================
# 補助的: ルールベース判定用 (Strong Negative)
# =========================================================
STRONG_NEG_EN: Set[str] = {
    "kill", "die", "death", "murder", "slaughter", "fuck", "shit", "bitch",
    "asshole", "idiot", "moron", "scum", "trash", "liar", "traitor", "nazi",
    "fascist", "racist", "terrorist", "genocide", "hypocrite", "disgusting",
    "bullshit", "warmonger", "destroy", "eliminate", "enemy", "evil"
}

class EnglishSentimentAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnglishSentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        print(f"[SentimentAI] Loading model to GPU: {MODEL_NAME} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval() # 評価モードに固定
            print(f"[SentimentAI] Model loaded successfully on {self.device}.")
        except Exception as e:
            print(f"[SentimentAI] Error loading model: {e}")
            self.model = None

    def _preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def predict_batch(self, texts: List[str], batch_size: int = 128) -> List[bool]:
        """
        ★ GPU並列化: テキストのリストを一括で推論し、大幅に高速化します。
        VRAMに余裕があれば batch_size を 256 や 512 に増やすとさらに速くなります。
        """
        if not self.model or not texts:
            return [False] * len(texts)
            
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_processed = [self._preprocess(t) for t in batch_texts]
            
            # 複数テキストを一括でTensor変換
            encoded_input = self.tokenizer(
                batch_processed, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            # GPUへ転送
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                output = self.model(**encoded_input)
            
            scores = output.logits.cpu().numpy()
            scores = softmax(scores, axis=1) # [batch_size, 3]
            
            # 各テキストの判定
            for score in scores:
                neg_score = score[0]
                pos_score = score[2]
                is_neg = (neg_score >= NEGATIVE_THRESHOLD and neg_score > pos_score)
                results.append(bool(is_neg))
                
        return results

# グローバルインスタンス
_analyzer = None

def load_english_negative_judge():
    global _analyzer
    if _analyzer is None:
        _analyzer = EnglishSentimentAnalyzer()
    return _analyzer

def _normalize(text: str) -> str:
    if not isinstance(text, str): return ""
    return text.lower().replace("\n", " ")

def check_strong_keyword(text: str) -> bool:
    """正規表現と辞書による超高速な事前チェック"""
    t_norm = _normalize(text)
    if not t_norm: return False
    tokens = set(re.findall(r"\b[a-z]+\b", t_norm))
    return not tokens.isdisjoint(STRONG_NEG_EN)

def is_negative_user_from_texts(texts: List[str]) -> bool:
    """単一ユーザーのテキストリスト判定(非推奨: 速度が落ちるためバッチ用関数を使用)"""
    for t in texts:
        if check_strong_keyword(t): return True
    analyzer = load_english_negative_judge()
    results = analyzer.predict_batch(texts, batch_size=32)
    return any(results)