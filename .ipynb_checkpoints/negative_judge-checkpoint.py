import unicodedata
from pathlib import Path
from typing import List, Set

# --- 設定 ---
# 辞書ファイルのパス（build_neg_dictionary.py で生成されたもの）
JP_NEG_DICT_PATH = Path("data/jp_negative_words.txt")

# 手動で定義する「超強ネガティブ語」（これが入っていたら即判定）
STRONG_NEG_JA: Set[str] = {
    "死ね", "殺す", "ぶっ殺す", "クソ", "ゴミ", "クズ",
    "無能", "低能", "バカ", "馬鹿", "カス", "消えろ",
    "頭おかしい", "老害", "死ねば",
}

# メモリ上にロードする辞書セット
JP_NEG_WORDS: Set[str] = set()


def load_jp_negative_lexicon():
    """
    起動時に一度だけ呼び出し、辞書をメモリにロードする関数。
    """
    global JP_NEG_WORDS
    if not JP_NEG_DICT_PATH.exists():
        print(f"WARNING: {JP_NEG_DICT_PATH} がありません。build_neg_dictionary.py を実行してください。")
        return

    with JP_NEG_DICT_PATH.open("r", encoding="utf-8") as f:
        # 空行を除去してセットに格納
        JP_NEG_WORDS = {w.strip() for w in f if w.strip()}
    
    print(f"[Lexicon] Loaded {len(JP_NEG_WORDS)} Japanese negative words.")


def _normalize(text: str) -> str:
    """テキスト正規化（NFKC、改行削除）"""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    return t.replace("\n", " ").replace("\t", " ")


def is_negative_text(text: str) -> bool:
    """
    1つのテキストがネガティブか判定する。
    
    判定基準:
      1. STRONG_NEG_JA に含まれる単語が 1つでも あれば True (即死)
      2. JP_NEG_WORDS (一般ネガ辞書) に含まれる単語が 2つ以上 あれば True
    """
    t = _normalize(text)
    if not t:
        return False

    # 1. 強ネガチェック (優先度高・処理高速)
    for w in STRONG_NEG_JA:
        if w in t:
            # デバッグ用にどの単語で引っかかったか知りたい場合はここで print(w)
            return True

    # 2. 一般ネガ辞書チェック
    # 辞書サイズが大きい場合のループ最適化
    # 全探索でも短文なら高速ですが、ヒット数が閾値を超えたら即returnして計算量を抑えます。
    hit_count = 0
    
    for w in JP_NEG_WORDS:
        if w in t:
            hit_count += 1
            if hit_count >= 2: # 閾値：2語以上でネガティブ認定
                return True

    return False


def is_negative_user_from_texts(texts: List[str]) -> bool:
    """
    ユーザーの投稿リスト（複数）を受け取り、
    1つでもネガティブ判定される投稿があれば True を返す。
    """
    if not JP_NEG_WORDS:
        # 辞書が空の場合はロードを試みる（安全策）
        load_jp_negative_lexicon()

    for t in texts:
        if is_negative_text(t):
            return True
    return False


# --- 動作確認用 ---
if __name__ == "__main__":
    # 1. 辞書ロード
    load_jp_negative_lexicon()

    # 2. テストデータ
    samples = [
        "あいつは本当にクズだな",       # STRONG_NEGヒット
        "昨日は体調が悪くて、気分も落ち込んで最悪だった", # 一般ネガ(悪い, 落ち込む, 最悪) -> Hit
        "今日は最高の天気ですね！",      # ポジティブ -> False
        "ちょっと疲れたけど頑張る",      # ネガ1語(疲れた)だけ -> False (閾値2のため)
    ]

    print("-" * 30)
    for s in samples:
        result = is_negative_text(s)
        print(f"判定: {'[黒]' if result else '[白]'} : {s}")