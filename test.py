import streamlit as st
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
import urllib
import requests
import json


# このスクリプトと同じディレクトリの icons/logo.png をアイコンにしたい例
icon_path = os.path.join(os.path.dirname(__file__), "icons", "logo.png")

st.set_page_config(
    page_title="Blue Sky Analysis App",
    page_icon=icon_path
)

# Elasticsearch の接続先（環境に合わせて変更してください）
es = Elasticsearch("http://localhost:9200")

# サイドバー入力：クエリと期間
st.sidebar.header("入力パラメータ")
query_str = st.sidebar.text_input("クエリ（例: Trump）", "Trump")
today = dt.date.today()
yesterday = today + relativedelta(days=-1)
twodaysago = today + relativedelta(days=-2)
start_date = st.sidebar.date_input("開始日", value=twodaysago)
end_date = st.sidebar.date_input("終了日", value=yesterday)
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

# CSSスタイルの例
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Hello World !</p>', unsafe_allow_html=True)

# ユーザ情報を取得する関数 (Elasticsearchのインデックス名を調整)
def get_user_info(did):
    try:
        res = es.get(index="userindex-*", id=did) # ユーザ情報を格納するインデックスに対して、DIDで検索
        user_info = res['_source']['commit']['record']
        return user_info
    except Exception as e:
        return {"displayName": did, "avatar": None} # エラー時はデフォルト値を返す

# プロファイル情報を取得する関数
def get_profile_info(did):
    try:
        res = es.get(index="profileindex-*", id=did)
        profile_info = res['_source']['commit']['record']
        return profile_info.get("displayName", did)  # displayName が存在しない場合は DID を返す
    except Exception as e:
        return did  # エラー時は DID を返す

# ユーザーのプロフィール画像のURLを取得する関数
def get_profile_avatar_url(did):
    try:
        res = es.get(index="profileindex-*", id=did)
        profile_info = res['_source']['commit']['record']
        return profile_info.get("avatar", None)
    except Exception as e:
        return None

# 関数: 認証トークンを取得
def get_auth_token(handle: str, password: str):
    login_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = {
        "identifier": handle,  # 例: yourname.bsky.social
        "password": password
    }
    try:
        resp = requests.post(login_url, json=payload)
        resp.raise_for_status()  # HTTPエラーを例外として発生
        return resp.json()["accessJwt"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"ログイン失敗 (リクエストエラー): {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"ログイン失敗 (JSONデコードエラー): {e}")
    except KeyError as e:
        raise Exception(f"ログイン失敗 (キーエラー): {e}")

# 関数: アバターURLを取得
def get_avatar_url(handle: str, token: str):
    url = f"https://bsky.social/xrpc/app.bsky.actor.getProfile?actor={handle}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json().get("avatar")
    except requests.exceptions.RequestException as e:
        print(f"プロフィール取得失敗 (リクエストエラー): {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"プロフィール取得失敗 (JSONデコードエラー): {e}")
        return None

#  Streamlit アプリパスワード入力
bluesky_handle =  "bcoredsclass0001.bsky.social"
bluesky_password = "masanao_ochi"

#########################
# 1. 投稿数の時系列グラフ（postindex）
#########################
def get_post_time_series(query, start, end, lang_filter=None):
    # 基本の must 条件
    must_clauses = [
        {"match": {"commit.record.text": query}},
        {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
    ]
    # lang_filter が設定されていれば、"langs" フィールドに対する term フィルタを追加
    if lang_filter:
        must_clauses.append({"term": {"commit.record.langs": lang_filter}})
    
    body = {
        "size": 0,
        "query": {
            "bool": {
                "must": must_clauses
            }
        },
        "aggs": {
            "posts_over_time": {
                "date_histogram": {
                    "field": "commit.record.createdAt",
                    "fixed_interval": "10m"
                }
            }
        }
    }

#    st.write(body)
    res = es.search(index="postindex-*", body=body)
    buckets = res["aggregations"]["posts_over_time"]["buckets"]
    df = pd.DataFrame([{"time": b["key_as_string"], "count": b["doc_count"]} for b in buckets])
    return df

post_ts_df = get_post_time_series(query_str, start_date_str, end_date_str, lang_filter=language_filter)
st.subheader("1. 投稿数の時系列（10分刻み）")
#st.write(post_ts_df)
st.plotly_chart(px.line(post_ts_df, x="time", y="count", title="投稿数の時系列"))

#########################
# 2. 投稿数のユーザランキング（postindex）
#########################
def get_post_user_ranking(query, start, end):
    body = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"match": {"commit.record.text": query}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        },
        "aggs": {
            "users": {
                "terms": {
                    "field": "did",
                    "size": 100
                }
            }
        }
    }
    res = es.search(index="postindex-*", body=body)
    buckets = res["aggregations"]["users"]["buckets"]
    df = pd.DataFrame([{"user": b["key"], "post_count": b["doc_count"]} for b in buckets])
    df = df.sort_values("post_count", ascending=False)
    return df

user_rank_df = get_post_user_ranking(query_str, start_date_str, end_date_str)

# プロファイル情報を取得し、必要なカラムを追加
user_rank_df['profile_url'] = user_rank_df['user'].apply(lambda did: f"https://bsky.app/profile/{did}")
user_rank_df['profile_image'] = user_rank_df['user'].apply(get_profile_avatar_url)

# Bluesky APIを使ってアバターURLを取得
if bluesky_handle and bluesky_password:  # 認証情報がある場合のみアバターを取得
    user_rank_df['profile_image'] = None  # 認証情報がない場合はアバターをNoneにする
else:
    user_rank_df['profile_image'] = None  # 認証情報がない場合はアバターをNoneにする

st.subheader("2. 投稿数のユーザ別ランキング")
st.dataframe(
    user_rank_df,
    column_config={
        "user": st.column_config.TextColumn(
            "DID",
            disabled=True,  # DID を編集不可にする
        ),
        "post_count": st.column_config.NumberColumn(
            "投稿数",
            help="ユーザの投稿数"
        ),
        "profile_url": st.column_config.LinkColumn(
            "リンク",
            display_text="Open Profile",
        ),
        "profile_image": st.column_config.ImageColumn(
            "アバター",
            help="ユーザのプロフィール画像"
        )
    },
    hide_index=True,
)

#########################
# 3. ブロック数のユーザランキング（blockindex）
#########################
def get_block_user_ranking(start, end):
    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.operation": "create"}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        },
        "aggs": {
            "users": {
                "terms": {
                    "field": "did",
                    "size": 100
                }
            }
        }
    }
    res = es.search(index="blockindex-*", body=body)
    buckets = res["aggregations"]["users"]["buckets"]
    df = pd.DataFrame([{"user": b["key"], "block_count": b["doc_count"]} for b in buckets])
    df = df.sort_values("block_count", ascending=False)
    return df

block_rank_df = get_block_user_ranking(start_date_str, end_date_str)

# プロファイル情報を取得し、必要なカラムを追加
block_rank_df['profile_url'] = block_rank_df['user'].apply(lambda did: f"https://bsky.app/profile/{did}")
block_rank_df['profile_image'] = block_rank_df['user'].apply(get_profile_avatar_url)

# Bluesky APIを使ってアバターURLを取得
if bluesky_handle and bluesky_password: # 認証情報がある場合のみアバターを取得
    block_rank_df['profile_image'] = None # 認証情報がない場合はアバターをNoneにする
else:
    block_rank_df['profile_image'] = None # 認証情報がない場合はアバターをNoneにする


st.subheader("3. ブロックされた数のユーザランキング")
st.dataframe(
    block_rank_df,
    column_config={
        "user": st.column_config.TextColumn(
            "DID",
            disabled=True,  # DID を編集不可にする
        ),
        "block_count": st.column_config.NumberColumn(
            "ブロック数",
            help="ユーザがブロックされた数"
        ),
        "profile_url": st.column_config.LinkColumn(
            "リンク",
            display_text="Open Profile",
        ),
        "profile_image": st.column_config.ImageColumn(
            "アバター",
            help="ユーザのプロフィール画像"
        )
    },
    hide_index=True,
)



##########################################
# まず、postindex から対象投稿の URI を再構築する関数
##########################################
def get_post_uris_and_texts(query, start, end):
    # クエリ条件に合致する投稿を取得
    body = {
        "size": 1000,  # 必要件数に応じて調整
        "query": {
            "bool": {
                "must": [
                    {"match": {"commit.record.text": query}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        }
    }
    res = es.search(index="postindex-*", body=body)
    posts = []  # 各投稿の情報（URI やテキストなど）
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        try:
            did  = src["did"]
            coll = src["commit"]["collection"]
            rkey = src["commit"]["rkey"]
            # URI を "at://{did}/{collection}/{rkey}" として再構築
            uri = f"at://{did}/{coll}/{rkey}"
            # テキストを取得
            text = src["commit"]["record"].get("text", "")
            # 投稿時間を取得
            created_at = src["commit"]["record"].get("createdAt", "")
            posts.append({"uri": uri, "text": text, "did": did, "createdAt": created_at}) # 投稿時間を追加
        except Exception as e:
            continue
    return posts

##########################################
# 4. リポストされた数のランキング（対象投稿ごと）
##########################################
def get_repost_ranking(query, start, end):
    posts = get_post_uris_and_texts(query, start, end)
    if not posts:
        return pd.DataFrame()

    uris = [post['uri'] for post in posts]

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"commit.record.subject.uri": uris}},
                    {"term": {"commit.operation": "create"}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        },
        "aggs": {
            "post_reposts": {
                "terms": {
                    "field": "commit.record.subject.uri",
                    "size": len(uris)
                }
            }
        }
    }
    res = es.search(index="repostindex-*", body=body)
    buckets = res["aggregations"]["post_reposts"]["buckets"]

    data = []
    for b in buckets:
        uri = b["key"]
        repost_count = b["doc_count"]

        # **元の投稿の情報を取得**
        original_post = next((p for p in posts if p["uri"] == uri), None) # 元の投稿を検索
        if original_post:
            original_did = original_post["did"] # 元の投稿のdid
            original_user_info = get_user_info(original_did) # 元の投稿者のuserInfo
            original_text = original_post["text"] # 元の投稿のテキスト
            original_time = original_post.get("createdAt", None) # 元の投稿時間
        else:
            original_did = None
            original_user_info = {"displayName": "不明", "avatar": None}
            original_text = "不明"
            original_time = None

        data.append({
            "uri": uri,
            "repost_count": repost_count,
            # **元の投稿の情報を追加**
            "original_did": original_did,
            "original_user_name": original_user_info["displayName"],
            "original_user_avatar": original_user_info["avatar"],
            "original_text": original_text,
            "original_time": original_time # 元の投稿時間
        })

    df = pd.DataFrame(data)
    posts_df = pd.DataFrame(posts)
    merged = pd.merge(posts_df, df, on="uri", how="left").fillna(0)
    merged["repost_count"] = merged["repost_count"].astype(int)
    merged = merged.sort_values("repost_count", ascending=False)
    return merged

st.markdown("""
<style>
    .repost-container {
        background-color: #f9f9f9; /* 明るい背景色 */
        border: 1px solid #e1e1e1; /* 細いボーダー */
        padding: 15px; /* 少し広めのパディング */
        margin-bottom: 15px; /* 下マージンも広めに */
        border-radius: 10px; /* 角丸を少し大きめに */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* 影を追加 */
    }
    .repost-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px; /* 下マージン */
    }
    .repost-header img {
        border-radius: 50%; /* 丸いアイコン */
        margin-right: 10px;
        width: 50px; /* アイコンサイズを調整 */
        height: 50px; /* アイコンサイズを調整 */
        border: 2px solid #fff; /* 白いボーダー */
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* 影を追加 */
    }
    .repost-username {
        font-weight: bold; /* ユーザー名を太字に */
        color: #333; /* 少し濃いめの文字色 */
        margin-right: auto; /* 右端に寄せる */
    }
    .repost-time {
        font-size: 0.9em; /* 少し小さめのフォントサイズ */
        color: #777; /* 少し薄めの文字色 */
    }
    .repost-content {
        margin-bottom: 10px;
        color: #555; /* 本文の文字色 */
        line-height: 1.5; /* 行間を調整 */
    }
    .repost-link {
        color: #007bff; /* リンク色を強調 */
        text-decoration: none; /* 下線を削除 */
    }
    .repost-link:hover {
        text-decoration: underline; /* ホバー時に下線を表示 */
    }
    .repost-count {
        font-size: 1em; /* 少し大きめのフォントサイズ */
        color: #4CAF50; /* いい感じの緑色 */
        font-weight: bold;
    }

    .no-data-message {  /* データがない場合のメッセージのスタイル */
        background-color: #f9f9f9;
        border: 1px solid #e1e1e1;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 4. リポストランキング
repost_ranking_df = get_repost_ranking(query_str, start_date_str, end_date_str)
st.subheader("4. クエリにヒットした投稿のリポスト数ランキング")

if not repost_ranking_df.empty:
    page_size = 5
    num_pages = (len(repost_ranking_df) // page_size) + 1
    page_num = st.number_input("ページ番号", min_value=1, max_value=num_pages, value=1, key="repost_page")
    start_index = (page_num - 1) * page_size
    end_index = start_index + page_size

    st.write(f"Page {page_num} of {num_pages}")  # ページ数表示改善

    # 認証情報
    if not bluesky_handle or not bluesky_password:
        st.warning("ハンドルとパスワードが設定されていません。")
    else:
        try:
            token = get_auth_token(bluesky_handle, bluesky_password)
            for index, row in repost_ranking_df[start_index:end_index].iterrows():
                with st.container():  # コンテナを作成
                    st.write('<div class="repost-container">', unsafe_allow_html=True)
    
                    #元の投稿者の情報
                    original_avatar_url = get_avatar_url(row['original_user_name'], token)
                    if original_avatar_url:
                        original_avatar_display = f'<img src="{original_avatar_url}" width="40" alt="Profile Image">'
                    else:
                        original_avatar_display = '<span>No Avatar</span>'
    
                    # ヘッダー
                    if row['original_time']:
                        original_time = dt.datetime.strptime(row['original_time'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        original_time = "不明"
    
                    st.write(f"""
                        <div class="repost-header">
                            {original_avatar_display}
                            <span class="repost-username">元投稿ユーザー: {row['original_user_name']}</span>
                            <span class="repost-time">投稿時間: {original_time}</span>
                        </div>
                        """, unsafe_allow_html=True)
    
                    # 元の投稿内容
                    st.write(f'<div class="repost-content">**投稿内容:** {row["original_text"]}</div>', unsafe_allow_html=True)
    
                    # リンク
                    st.write(f'<a href="https://bsky.app/profile/{row["original_did"]}/post/{row["uri"].split("/")[-1]}" class="repost-link">元の投稿を見る</a>', unsafe_allow_html=True)
    
                    # リポスト数
                    st.write(f'<div class="repost-count">リポスト数: {row["repost_count"]}</div>', unsafe_allow_html=True)
    
                    st.write('</div>', unsafe_allow_html=True)  # post-containerの閉じタグ
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

##########################################
# 5. Like された数のランキング（対象投稿ごと）
##########################################
def get_like_ranking(query, start, end):
    posts = get_post_uris_and_texts(query, start, end)
    if not posts:
        return pd.DataFrame()

    uris = [post['uri'] for post in posts]

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"commit.record.subject.uri": uris}},
                    {"term": {"commit.operation": "create"}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        },
        "aggs": {
            "post_likes": {
                "terms": {
                    "field": "commit.record.subject.uri",
                    "size": len(uris)
                }
            }
        }
    }
    res = es.search(index="likeindex-*", body=body)
    buckets = res["aggregations"]["post_likes"]["buckets"]

    data = []
    for b in buckets:
        uri = b["key"]
        like_count = b["doc_count"]

        # **元の投稿の情報を取得**
        original_post = next((p for p in posts if p["uri"] == uri), None) # 元の投稿を検索
        if original_post:
            original_did = original_post["did"] # 元の投稿のdid
            original_user_info = get_user_info(original_did) # 元の投稿者のuserInfo
            original_text = original_post["text"] # 元の投稿のテキスト
            original_time = original_post.get("createdAt", None)  # 元の投稿時間
        else:
            original_did = None
            original_user_info = {"displayName": "不明", "avatar": None}
            original_text = "不明"
            original_time = None

        data.append({
            "uri": uri,
            "like_count": like_count,
            # **元の投稿の情報を追加**
            "original_did": original_did,
            "original_user_name": original_user_info["displayName"],
            "original_user_avatar": original_user_info["avatar"],
            "original_text": original_text,
            "original_time": original_time  # 元の投稿時間
        })

    df_aggs = pd.DataFrame(data)
    posts_df = pd.DataFrame(posts)
    merged = pd.merge(posts_df, df_aggs, on="uri", how="left").fillna(0)
    merged["like_count"] = merged["like_count"].astype(int)
    merged = merged.sort_values("like_count", ascending=False)
    return merged


# 結果表示

st.markdown("""
<style>
    /* 共通のスタイル */
    .post-container {
        background-color: #f9f9f9;
        border: 1px solid #e1e1e1;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .post-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .post-header img {
        border-radius: 50%;
        margin-right: 10px;
        width: 50px;
        height: 50px;
        border: 2px solid #fff;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .post-username {
        font-weight: bold;
        color: #333;
        margin-right: auto;
    }
    .post-time {
        font-size: 0.9em;
        color: #777;
    }
    .post-content {
        margin-bottom: 10px;
        color: #555;
        line-height: 1.5;
    }
    .post-link {
        color: #007bff;
        text-decoration: none;
    }
    .post-link:hover {
        text-decoration: underline;
    }
    .no-data-message {
        background-color: #f9f9f9;
        border: 1px solid #e1e1e1;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 10px;
    }

    /* Likeランキング固有のスタイル */
    .like-count {
        font-size: 1em;
        color: #e44d26;  /* 明るい赤色 */
        font-weight: bold;
    }

    /* リポストランキング固有のスタイル */
    .repost-count {
        font-size: 1em;
        color: #4CAF50;  /* いい感じの緑色 */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 5. Likeランキング
like_ranking_df = get_like_ranking(query_str, start_date_str, end_date_str)
st.subheader("5. クエリにヒットした投稿のLike数ランキング")
if not like_ranking_df.empty:
    # 認証情報
    if not bluesky_handle or not bluesky_password:
        st.warning("ハンドルとパスワードが設定されていません。")
    else:
        try:
            token = get_auth_token(bluesky_handle, bluesky_password)
            page_size = 5  # 1ページあたりの表示件数
            page_num = st.number_input("ページ番号", min_value=1, max_value=(len(like_ranking_df) // page_size) + 1, value=1, key="like_page")
            start_index = (page_num - 1) * page_size
            end_index = start_index + page_size
            for index, row in like_ranking_df[start_index:end_index].iterrows():
                # CSSスタイルを適用
                st.markdown(f'<div class="post-container">', unsafe_allow_html=True)

                # 元の投稿者の情報を取得
                original_avatar_url = get_avatar_url(row['original_user_name'], token)
                if original_avatar_url:
                    original_avatar_display = f'<img src="{original_avatar_url}" width="40" alt="Profile Image">'
                else:
                    original_avatar_display = '<span>No Avatar</span>'

                # ヘッダー
                if row['original_time']:
                    original_time = dt.datetime.strptime(row['original_time'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
                else:
                    original_time = "不明"

                st.markdown(f"""
                    <div class="post-header">
                        {original_avatar_display}
                        <span class="post-username">元投稿ユーザー: {row['original_user_name']}</span>
                        <span class="post-time">投稿時間: {original_time}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # 投稿内容
                st.markdown(f'<div class="post-content">**投稿内容:** {row["original_text"]}</div>', unsafe_allow_html=True)

                # リンク
                st.markdown(f'<a href="https://bsky.app/profile/{row["original_did"]}/post/{row["uri"].split("/")[-1]}" class="post-link">元の投稿を見る</a>', unsafe_allow_html=True)

                # Like数
                st.markdown(f'<div class="like-count">Like数: {row["like_count"]}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)  # post-containerの閉じタグ
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")


#########################
# 6. 投稿内容のクラスタリング（postindex）
#########################
def cluster_post_texts(query, start, end, n_clusters=5):
    body = {
        "size": 1000,  # 必要件数を取得。大量の場合は Scroll API を検討
        "query": {
            "bool": {
                "must": [
                    {"match": {"commit.record.text": query}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        }
    }
    res = es.search(index="postindex-*", body=body)
    docs = [hit["_source"]["commit"]["record"].get("text", "") for hit in res["hits"]["hits"] if hit["_source"]["commit"]["record"].get("text")]
    if not docs:
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())
    df = pd.DataFrame({
        "x": X_reduced[:, 0],
        "y": X_reduced[:, 1],
        "cluster": labels,
        "text": docs
    })
    return df

post_cluster_df = cluster_post_texts(query_str, start_date_str, end_date_str, n_clusters=5)
# 数値の cluster 列を文字列に変換して離散カテゴリとして扱う
post_cluster_df["cluster_str"] = post_cluster_df["cluster"].astype(str)

if post_cluster_df is not None:
    st.subheader("6. 投稿内容のクラスタリング")
    fig = px.scatter(
        post_cluster_df,
        x="x",
        y="y",
        color="cluster_str",
        hover_data=["text"],
        color_discrete_sequence=px.colors.qualitative.Plotly  # はっきりとした色分けのパレット
    )
    st.plotly_chart(fig)
else:
    st.write("投稿のクラスタリング結果が得られませんでした。")

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

st.subheader("全体の特徴語ワードクラウド")
# 1. 全体のワードクラウド（全投稿の頻度を使用）
all_texts = post_cluster_df["text"].tolist()

# CountVectorizer を使って全体の単語頻度を計算（英語の stopwords を除外）
vectorizer = CountVectorizer(stop_words="english")
all_matrix = vectorizer.fit_transform(all_texts)
overall_freq = np.array(all_matrix.sum(axis=0)).flatten()  # 各単語の総出現数
vocab = vectorizer.get_feature_names_out()
overall_dict = dict(zip(vocab, overall_freq))

# ワードクラウド生成（はっきりした色と高解像度設定）
overall_wc = WordCloud(width=800, height=400, background_color="white", scale=2).generate_from_frequencies(overall_dict)
plt.figure(figsize=(8,4), dpi=300)
plt.imshow(overall_wc, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt.gcf())
plt.close()


# 2. 各クラスタごとに、全体と比較して特徴的な単語を抽出してワードクラウド生成
st.subheader("クラスタごとの特徴語ワードクラウド（全体との差分）")

clusters = post_cluster_df["cluster"].unique()
# 各クラスタごとに処理
for cl in clusters:
    cluster_texts = post_cluster_df[post_cluster_df["cluster"] == cl]["text"].tolist()
    # 同じ vectorizer を使って、各クラスタの単語頻度を計算
    cluster_matrix = vectorizer.transform(cluster_texts)
    cluster_freq = np.array(cluster_matrix.sum(axis=0)).flatten()
    
    # 差分スコアの計算: (cluster_freq + 1) / (overall_freq + 1)
    diff_scores = (cluster_freq + 1) / (overall_freq + 1)
    # ここでは、比率が 1.0 より大きい単語（クラスタでより頻出している単語）を対象とする
    diff_dict = {word: score for word, score in zip(vocab, diff_scores) if score > 1.0}
    
    # 万が一差分が空の場合は、クラスタ内の頻度をそのまま使用（または別途処理）
    if not diff_dict:
        diff_dict = {word: freq for word, freq in zip(vocab, cluster_freq) if freq > 0}
    
    cluster_wc = WordCloud(width=800, height=400, background_color="white", scale=2).generate_from_frequencies(diff_dict)
    st.markdown(f"**クラスタ {cl} の特徴語**")
    plt.figure(figsize=(8,4), dpi=300)
    plt.imshow(cluster_wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.close()

#########################
# 7. プロフィール文のクラスタリング（profileindex）
#########################
def cluster_profile_descriptions(n_clusters=5):
    body = {
        "size": 1000,
        "query": {
            "match_all": {}
        }
    }
    res = es.search(index="profileindex-*", body=body)
    docs = [hit["_source"]["commit"]["record"].get("description", "") for hit in res["hits"]["hits"] if hit["_source"]["commit"]["record"].get("description")]
    if not docs:
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())
    df = pd.DataFrame({
        "x": X_reduced[:, 0],
        "y": X_reduced[:, 1],
        "cluster": labels,
        "description": docs
    })
    return df

profile_cluster_df = cluster_profile_descriptions(n_clusters=5)
# 数値の cluster 列を文字列に変換して離散カテゴリとして扱う
profile_cluster_df["cluster_str"] = profile_cluster_df["cluster"].astype(str)

if profile_cluster_df is not None:
    st.subheader("7. プロフィール文のクラスタリング")
    fig2 = px.scatter(
        profile_cluster_df, 
        x="x", 
        y="y", 
        color="cluster_str", 
        hover_data=["description"],
        color_discrete_sequence=px.colors.qualitative.Plotly  # はっきりとした色分けのパレット
    )
    st.plotly_chart(fig2)
else:
    st.write("プロフィール文のクラスタリング結果が得られませんでした。")

def generate_profile_wordclouds(df, cluster_col="cluster_str", text_col="description"):
    """
    各クラスタごとに、対象のプロフィール文を結合し、
    WordCloud オブジェクトを生成して返す。
    """
    # 各クラスタごとにテキストをまとめる
    cluster_texts = {}
    for cluster in df[cluster_col].unique():
        texts = df[df[cluster_col] == cluster][text_col].tolist()
        # 複数テキストを1つの文字列に連結
        cluster_texts[cluster] = " ".join(texts)
    
    # 各クラスタのワードクラウドを生成
    wordclouds = {}
    for cluster, text in cluster_texts.items():
        wc = WordCloud(
            width=800,
            height=400,
            scale=2,  # 高解像度生成
            background_color="white",
            stopwords=WordCloud().stopwords
        )
        wc.generate(text)
        wordclouds[cluster] = wc
    return wordclouds

# 例: profile_cluster_df には 'cluster' と 'description' の列があると仮定
# クラスタの値を文字列に変換しておく
profile_cluster_df["cluster_str"] = profile_cluster_df["cluster"].astype(str)

# 各クラスタごとのワードクラウドを生成
profile_wordclouds = generate_profile_wordclouds(profile_cluster_df, cluster_col="cluster_str", text_col="description")

st.subheader("7. プロフィール文のクラスタごとの特徴語（WordCloud）")
for cluster, wc in profile_wordclouds.items():
    st.markdown(f"**クラスタ {cluster}**")
    plt.figure(figsize=(8, 4), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.close()

#########################
# 8. プロフィールクラスタと投稿クラスタのヒートマップ
#########################
def create_cluster_heatmap(post_df, profile_df):
    # ここでは例として、両クラスタの組み合わせごとの件数をランダムに生成するサンプルです。
    # 実際は、ユーザごとの投稿クラスタとプロフィールクラスタをマージする必要があります。
    heat_data = np.random.randint(0, 50, (5, 5))  # 5×5 のヒートマップ用サンプルデータ
    return heat_data

if post_cluster_df is not None and profile_cluster_df is not None:
    heat_data = create_cluster_heatmap(post_cluster_df, profile_cluster_df)
    st.subheader("8. プロフィールクラスタと投稿内容クラスタのヒートマップ")
    fig_heat = px.imshow(heat_data, labels=dict(x="プロフィールクラスタ", y="投稿内容クラスタ", color="件数"),
                         x=list(range(5)), y=list(range(5)))
    st.plotly_chart(fig_heat)
else:
    st.write("ヒートマップの生成に必要なデータが不足しています。")

#########################
# 9. LIKE のネットワーク表示（likeindex）
#########################
def create_like_network(query, start, end):
    # 1) クエリに合致する投稿URIを取得
    posts = get_post_uris_and_texts(query, start, end) # ★修正箇所
    if not posts:
        st.write("No URIs found for the given query.")
        return None

    uris = [post['uri'] for post in posts] # ★修正箇所

    # 2) likeindex-* から subject.uri が上記 uris に含まれるものだけ検索
    body = {
        "size": 1000,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"commit.record.subject.uri": uris}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}},
                    {"term": {"commit.operation": "create"}}  # Like されたドキュメント
                ]
            }
        }
    }
    res = es.search(index="likeindex-*", body=body)
    # 3) ソース(投稿URI) → ターゲット(Likeしたユーザ DID) のエッジ
    edges = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        commit = src.get("commit", {})
        record = commit.get("record", {})
        subject = record.get("subject", {})
        subject_uri = subject.get("uri")
        if not subject_uri:
            continue
        user_did = src.get("did")  # Likeしたユーザ
        if user_did:
            # ソース: クエリ該当投稿のURI, ターゲット: Likeしたユーザ
            edges.append((subject_uri, user_did))

    # 4) 無向グラフを作成
    G = nx.Graph()
    G.add_edges_from(edges)

    if G.number_of_nodes() == 0:
        st.write("No like edges found.")
        return None

    # 5) 度数が1以下のノードを除外
    G = G.subgraph(n for n in G if G.degree(n) > 1)
    if G.number_of_nodes() == 0:
        st.write("After filtering degree<=1, no connected nodes remain.")
        return None

    # 6) 最大連結成分だけ抽出
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc_nodes).copy()

    if G.number_of_nodes() == 0:
        st.write("No nodes in the largest connected component.")
        return None

    return G

like_graph = create_like_network(query_str, start_date_str, end_date_str)
st.subheader("9. LIKE のネットワーク表示")
if like_graph.number_of_nodes() > 0:
    # Pyvisネットワークの作成（高さや幅は適宜調整してください）
    net = Network(height="600px", width="100%", notebook=True)
    net.from_nx(like_graph)
            
    # 各ノードの属性を上書きして、labelは空、titleにノードIDを設定
    for node in net.nodes:
        node['title'] = str(node['id'])  # マウスオーバー時のツールチップに表示
        node['label'] = ""              # ノードのラベルは非表示
    
    # ノード追加：常にラベルは表示せず、マウスオーバー時にtitleとして表示
#    for node in like_graph.nodes():
#        net.add_node(node, title=str(node), label="")  
#    for source, target in like_graph.edges():
#        net.add_edge(source, target)
    
    # HTMLとして保存
    net.show("like_network.html")
    
    # 保存したHTMLファイルを読み込み、Streamlitで表示
    with open("like_network.html", "r", encoding="utf-8") as html_file:
        source_code = html_file.read()
        # デフォルトでは枠が "1px solid lightgray" になっているので、"border: none" に変更
        source_code = source_code.replace("border: 1px solid lightgray", "border: none")
#        source_code = source_code.replace("border: 1px solid rgba(0, 0, 0, .125)", "border: none")
                 

    components.html(source_code, height=600, width=800)
else:
    st.write("LIKE ネットワークのデータがありません。")


#like_graph = create_like_network(query_str, start_date_str, end_date_str)
#st.subheader("9. LIKE のネットワーク表示")
#if like_graph.number_of_nodes() > 0:
#    plt.figure(figsize=(6, 6))
#    pos = nx.spring_layout(like_graph, k=0.5)
#    nx.draw(like_graph, pos, with_labels=True, node_size=500, arrowsize=20)
#    st.pyplot(plt)
#else:
#    st.write("LIKE ネットワークのデータがありません。")

#########################
# 10. リポストのネットワーク表示（repostindex）
#########################
def create_repost_network(query, start, end):
    # 1) クエリに合致する投稿URIを取得
    posts = get_post_uris_and_texts(query, start, end) # ★修正箇所
    if not posts:
        st.write("No URIs found for the given query.")
        return None
    uris = [post['uri'] for post in posts] # ★修正箇所

    # 2) repostindex から subject.uri が上記 uris に含まれるリポストのみ検索
    body = {
        "size": 1000,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"commit.record.subject.uri": uris}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}},
                    {"term": {"commit.operation": "create"}}
                ]
            }
        }
    }
    res = es.search(index="repostindex-*", body=body)
    edges = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        commit = src.get("commit", {})
        record = commit.get("record", {})
        subject = record.get("subject", {})
        subject_uri = subject.get("uri")
        if not subject_uri:
            continue
        user_did = src.get("did")  # リポストしたユーザ
        if user_did:
            # ソース: クエリ該当投稿のURI, ターゲット: リポストしたユーザDID
            edges.append((subject_uri, user_did))

    # 3) 無向グラフを作成
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(edges)

    if G.number_of_nodes() == 0:
        st.write("No repost edges found.")
        return None

    # 4) edge filtering: 度数が1以下のノードは除外
    #   => たとえばノードが単に自己ループしかない or ほぼ繋がりがない場合を除外
    G = G.subgraph(n for n in G if G.degree(n) > 1)

    # フィルタ後にノードが0になったら終了
    if G.number_of_nodes() == 0:
        st.write("After filtering degree<=1, no connected nodes remain.")
        return None

    # 5) 最大連結成分を抽出
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc_nodes).copy()

    if G.number_of_nodes() == 0:
        st.write("No nodes in the largest connected component.")
        return None

    return G

repost_graph = create_repost_network(query_str, start_date_str, end_date_str)
st.subheader("10. リポストのネットワーク表示")
if repost_graph.number_of_nodes() > 0:
    # Pyvisネットワークの作成（表示サイズは適宜調整）
    net = Network(height="600px", width="100%", notebook=True)
    
    # NetworkXのグラフから読み込み
    net.from_nx(repost_graph)
    
    # 各ノードの属性を上書き
    for node in net.nodes:
        node['title'] = str(node['id'])  # ツールチップ用にノードIDを設定
        node['label'] = ""              # ラベルは空にして非表示にする
    
    # HTMLファイルとして出力
    net.show("repost_network.html")
    
    # 生成されたHTML内のCSSで指定されている枠線を非表示に変更
    with open("repost_network.html", "r", encoding="utf-8") as html_file:
        source_code = html_file.read()
    source_code = source_code.replace('border: 1px solid lightgray', 'border: none')
    
    # StreamlitでHTMLを埋め込み表示
    components.html(source_code, height=600, width=800)
else:
    st.write("リポストネットワークのデータがありません。")

#repost_graph = create_repost_network(query_str, start_date_str, end_date_str)
#st.subheader("10. リポストのネットワーク表示")
#if repost_graph.number_of_nodes() > 0:
#    plt.figure(figsize=(6, 6))
#    pos = nx.spring_layout(repost_graph, k=0.3, iterations=50)
#    nx.draw(repost_graph, pos, with_labels=True, node_size=500, arrowsize=20)
#    st.pyplot(plt)
#else:
#    st.write("リポストネットワークのデータがありません。")

