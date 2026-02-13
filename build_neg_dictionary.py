import requests
import pandas as pd
from pathlib import Path

# --- 設定 ---
TOHOKU_DICT_URL = "https://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/wago.121808.pn"
PN_CSV_PATH = Path("pn.csv.m3.120408.trim") # ローカルにある名詞辞書
OUTPUT_PATH = Path("data/jp_negative_words.txt") # 出力先


def fetch_tohoku_wago_negatives() -> set[str]:
    """東北大学の日本語評価極性辞書（用言編）からネガティブ語を取得"""
    print(f"[1/3] Downloading & Parsing Tohoku Dictionary from {TOHOKU_DICT_URL} ...")
    neg_words = set()
    try:
        response = requests.get(TOHOKU_DICT_URL)
        response.raise_for_status()
        
        # 文字コード自動判別（重要）
        response.encoding = response.apparent_encoding
        content = response.text
        
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            
            label = parts[0] # 例: "ネガ（経験）"
            word = parts[1].strip()
            
            if "ネガ" in label and word:
                neg_words.add(word)
                
        print(f"      -> Found {len(neg_words)} negative words (verbs/adjs).")
        return neg_words

    except Exception as e:
        print(f"Error fetching Tohoku dictionary: {e}")
        return set()


def load_pn_csv_negatives() -> set[str]:
    """手元の pn.csv.m3.120408.trim（名詞編）からネガティブ語を取得"""
    print(f"[2/3] Loading local dictionary from {PN_CSV_PATH} ...")
    neg_words = set()

    if not PN_CSV_PATH.exists():
        print(f"WARNING: {PN_CSV_PATH} が見つかりません。名詞辞書はスキップします。")
        return set()

    try:
        # 自前でパース（pandasを使わず軽量に処理）
        with open(PN_CSV_PATH, encoding="utf-8-sig") as f: # BOM付き対応
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                
                word = parts[0].strip('"')
                label = parts[1]
                
                # 'n' (negative) のみを抽出
                if label == 'n':
                    neg_words.add(word)

        print(f"      -> Found {len(neg_words)} negative words (nouns).")
        return neg_words

    except Exception as e:
        print(f"Error reading pn.csv: {e}")
        return set()


def main():
    # 1. 用言辞書（Web）を取得
    wago_set = fetch_tohoku_wago_negatives()
    
    # 2. 名詞辞書（Local）を取得
    noun_set = load_pn_csv_negatives()
    
    # 3. 統合（和集合）
    combined_set = wago_set.union(noun_set)
    
    # 4. 保存
    print(f"[3/3] Saving merged dictionary to {OUTPUT_PATH} ...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for w in sorted(combined_set):
            f.write(w + "\n")
            
    print(f"DONE! Total {len(combined_set)} unique negative words saved.")


if __name__ == "__main__":
    main()