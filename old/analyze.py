import pandas as pd
import sys
import streamlit_authenticator as stauth
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
import yaml
from yaml.loader import SafeLoader

import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation as LDA
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from collections import Counter, defaultdict
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import matplotlib.colors as mcolors
from pyvis.network import Network
from transformers import pipeline
import seaborn as sns
import tempfile
from transformers import AutoTokenizer
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification

es_addr="http://cicero.csis.oita-u.ac.jp:9200" #Elasticsearchアドレス
es_request_timeout = 600 #タイムアウトする時間
es_timeout = "10m" #サーバ処理時間の上限

def analyze(config):
    import streamlit as st
    # 分析
    
    # Elasticsearch の接続先（環境に合わせて変更してください）
    # es = Elasticsearch(es_addr) #接続するだけ
    es = Elasticsearch(
        hosts=[es_addr],
        request_timeout=es_request_timeout,     # クライアント側のタイムアウト（秒）
        retry_on_timeout=True,                  # タイムアウト時の再試行を有効に
        max_retries=5,                          # 再試行回数
    )
    
    # サイドバー入力：クエリと期間
    st.sidebar.header("入力パラメータ")
    option = st.sidebar.radio(
        "クエリの種類を選択してください。：",
        ["二つ", "３つ以上", "日本の政党と炎上ポスト"],
        index=0
    )
    
    if option == "二つ":
        query_str_a = st.sidebar.text_input("クエリA（例: Trump）", "Trump") #文字入力欄を作る
        query_str_b = st.sidebar.text_input("クエリB（例: UNO）", "UNO") #クエリB
        
    elif option == "３つ以上":
        pass
        num_queries = st.sidebar.number_input("クエリの数", min_value=3, max_value=10, value=3)
        queries = []
        for i in range(num_queries):
            q = st.sidebar.text_input(f"クエリ {i+1}", f"Query {i+1}")
            queries.append(q)

    elif option == "日本の政党と炎上ポスト":
        query_list = [
            "自民",
            "国民民主",
            "社民",
            "立憲",
            "公明",
            "維新",
            "共産党",
            "れいわ",
            "参政",
            "チームみらい"
            ]
        query_list_tanaka = [
            "永野 芽郁",
            "田中 圭",
            "奥さん 妻",
            "不倫",
            "一途"
            ]
        query_list_togisen = [
            "都議選",
            "自民",
            "都民",
            "公明",
            "共産",
            "立憲 立民",
            "ネット",
            "国民 国民民主",
            "参政",
            "無所属"
        ]
        pass
    
    today = dt.date.today() #今日の日付
    yesterday = today + relativedelta(days=-1) #昨日
    twodaysago = today + relativedelta(days=-2) #一昨日
    start_date = st.sidebar.date_input("開始日", value=twodaysago) #日付入力欄を作る
    end_date = st.sidebar.date_input("終了日", value=yesterday)
    start_date_togisen = st.sidebar.date_input("都議選開始日", value=twodaysago) #日付入力欄を作る
    end_date_togisen = st.sidebar.date_input("都議選終了日", value=yesterday)
    language_option = st.sidebar.selectbox("対象言語", ["すべて", "英語", "日本語"])
    snowball_iterations = st.sidebar.slider("雪だるま式サンプリングの繰り返し回数", min_value=1, max_value=10, value=5)
    snowball_users_per_iter = st.sidebar.slider("各ステップで選択するユーザー数", min_value=1, max_value=50, value=10)
    
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

    #  Streamlit アプリパスワード入力
    bluesky_handle =  config["bsky_user"]["handle"]
    bluesky_password = config["bsky_user"]["password"]

    if option == "二つ":
        st.write("クエリAの投稿内容を表示中...")
        
        query_body_a = {
          "query": {
            "bool": {
              "must": [
                { "match": { "commit.record.text": query_str_a } },
                { "range": { "commit.record.createdAt": { "gte": start_date.isoformat(), "lte": end_date .isoformat()} } }
              ]
            }
          }
        }
    
        st.write("クエリBの投稿内容を表示中...")
        query_body_b = {
          "query": {
            "bool": {
              "must": [
                { "match": { "commit.record.text": query_str_b } },
                { "range": { "commit.record.createdAt": { "gte": start_date.isoformat(), "lte": end_date .isoformat()} } }
              ]
            }
          }
        }
    
        post_df_a = get_post_texts(es, query_body_a)
        # post_df_a は既に存在する DataFrame
        post_df_a["label"] = "A"
        st.dataframe(post_df_a)
    
        post_ts_df_a = get_post_time_series(es, query_str_a, start_date_str, end_date_str, lang_filter=language_filter)
    
        post_df_b = get_post_texts(es, query_body_b)
        # post_df_a は既に存在する DataFrame
        post_df_b["label"] = "B"
        st.dataframe(post_df_b)
    
        post_ts_df_b = get_post_time_series(es, query_str_b, start_date_str, end_date_str, lang_filter=language_filter)
    
        combined_post_df = pd.concat([post_df_a, post_df_b], ignore_index=True)
        
        st.subheader("1-1. 全体の投稿数の時系列（10分刻み）-クエリA")
        #st.write(post_ts_df)
        st.plotly_chart(px.line(post_ts_df_a, x="time", y="count", title="投稿数の時系列-クエリA"))
    
        st.subheader("1-2. 全体の投稿数の時系列（10分刻み）-クエリB")
        #st.write(post_ts_df)
        st.plotly_chart(px.line(post_ts_df_b, x="time", y="count", title="投稿数の時系列-クエリB"))
    
        # Bluesky API用トークン取得（すでにconfigから）
        token = get_auth_token(bluesky_handle, bluesky_password) if bluesky_handle and bluesky_password else None ##俣江追加コード
        
        user_rank_df_a = get_post_user_ranking(es, query_str_a, start_date_str, end_date_str)
    
    # ユーザ名と表示名取得（並列取得）(matae)
        if token:
            did_list_a = user_rank_df_a['did'].tolist()
            did_info_map_a = fetch_display_info_parallel(did_list_a, token)
            user_rank_df_a["handle"] = user_rank_df_a["did"].map(lambda did: did_info_map_a.get(did, {}).get("handle", did))
            user_rank_df_a["displayName"] = user_rank_df_a["did"].map(lambda did: did_info_map_a.get(did, {}).get("displayName", ""))
        else:
            user_rank_df_a["handle"] = user_rank_df_a["did"]
            user_rank_df_a["displayName"] = "未取得"
            
        # プロファイル情報を取得し、必要なカラムを追加
        user_rank_df_a['profile_url'] = user_rank_df_a['did'].apply(lambda did: f"https://bsky.app/profile/{did}")
        user_rank_df_a['profile_image'] = user_rank_df_a['did'].apply(get_profile_avatar_url)
    
        # 認証情報がない場合はアバターを無効化（俣江）
        if not token:
            user_rank_df_a['profile_image'] = None
    
        # Bluesky APIを使ってアバターURLを取得
        if bluesky_handle and bluesky_password:  # 認証情報がある場合のみアバターを取得
            user_rank_df_a['profile_image'] = None  # 認証情報がない場合はアバターをNoneにする
        else:
            user_rank_df_a['profile_image'] = None  # 認証情報がない場合はアバターをNoneにする
        
        user_rank_df_b = get_post_user_ranking(es, query_str_b, start_date_str, end_date_str)
    
    # ユーザ名と表示名取得（並列取得）(matae)
        if token:
            did_list_b = user_rank_df_b['did'].tolist()
            did_info_map_b = fetch_display_info_parallel(did_list_b, token)
            user_rank_df_b["handle"] = user_rank_df_b["did"].map(lambda did: did_info_map_b.get(did, {}).get("handle", did))
            user_rank_df_b["displayName"] = user_rank_df_b["did"].map(lambda did: did_info_map_b.get(did, {}).get("displayName", ""))
        else:
            user_rank_df_b["handle"] = user_rank_df_b["did"]
            user_rank_df_b["displayName"] = "未取得"
            
        # プロファイル情報を取得し、必要なカラムを追加
        user_rank_df_b['profile_url'] = user_rank_df_b['did'].apply(lambda did: f"https://bsky.app/profile/{did}")
        user_rank_df_b['profile_image'] = user_rank_df_b['did'].apply(get_profile_avatar_url)
    
        # 認証情報がない場合はアバターを無効化（俣江）
        if not token:
            user_rank_df_b['profile_image'] = None
    
        # Bluesky APIを使ってアバターURLを取得
        if bluesky_handle and bluesky_password:  # 認証情報がある場合のみアバターを取得
            user_rank_df_b['profile_image'] = None  # 認証情報がない場合はアバターをNoneにする
        else:
            user_rank_df_b['profile_image'] = None  # 認証情報がない場合はアバターをNoneにする

        # st.subheader("2. 全体の投稿数のユーザ別ランキング")
        # st.dataframe(
        #     user_rank_df[["displayName", "handle", "post_count", "profile_url", "profile_image"]], ##matae
        #     column_config={
        #         "displayName": st.column_config.TextColumn(
        #             "ユーザ名",
        #             disabled=True,  # DID を編集不可にする
        #         ),
        #         "handle": st.column_config.TextColumn( ##俣江
        #             "ユーザID",
        #             help="Blueskyのハンドル名"
        #         ),
        #         "post_count": st.column_config.NumberColumn(
        #             "投稿数",
        #             help="ユーザの投稿数"
        #         ),
        #         "profile_url": st.column_config.LinkColumn(
        #             "リンク",
        #             display_text="Open Profile",
        #         ),
        #         "profile_image": st.column_config.ImageColumn(
        #             "アバター",
        #             help="ユーザのプロフィール画像"
        #         )
        #     },
        #     hide_index=True,
        # )
    
        # block_rank_df = get_block_user_ranking(es, start_date_str, end_date_str)
    
        # Bluesky APIを使ってアバターURLを取得
        # if bluesky_handle and bluesky_password: # 認証情報がある場合のみアバターを取得
        #     block_rank_df['profile_image'] = None # 認証情報がない場合はアバターをNoneにする
        # else:
        #     block_rank_df['profile_image'] = None # 認証情報がない場合はアバターをNoneにする
    
        # # 4. 表示名/ハンドルの取得（並列化、高速化）
        # if token:
        #     did_list_block = block_rank_df['did'].tolist()
        #     did_info_map_block = fetch_display_info_parallel(did_list_block, token)
        #     block_rank_df["handle"] = block_rank_df["did"].map(lambda did: did_info_map_block.get(did, {}).get("handle", did))
        #     block_rank_df["displayName"] = block_rank_df["did"].map(lambda did: did_info_map_block.get(did, {}).get("displayName", "不明"))
        # else:
        #     block_rank_df["handle"] = block_rank_df["did"]
        #     block_rank_df["displayName"] = "未取得"
    
        # # プロファイル情報を取得し、必要なカラムを追加
        # block_rank_df['profile_url'] = block_rank_df['did'].apply(lambda did: f"https://bsky.app/profile/{did}")
        # block_rank_df['profile_image'] = block_rank_df['did'].apply(get_profile_avatar_url)
    
    
        # st.subheader("3. ブロックされた数のユーザランキング")
        # st.dataframe(
        #     block_rank_df[["displayName", "handle", "block_count", "profile_url", "profile_image"]],
        #     column_config={
        #         "displayName": st.column_config.TextColumn(
        #             "ユーザ名",
        #         ),
        #         "handle": st.column_config.TextColumn(
        #             "ユーザID",
        #             help="ユーザがブロックされた数"
        #         ),
        #         "block_count":st.column_config.NumberColumn(
        #             "ブロック数",
        #             help="ユーザがブロックされた数"
        #         ),
        #         "profile_url": st.column_config.LinkColumn(
        #             "リンク",
        #             display_text="Open Profile",
        #         ),
        #         "profile_image": st.column_config.ImageColumn(
        #             "アバター",
        #             help="ユーザのプロフィール画像"
        #         )
        #     },
        #     hide_index=True,
        # )
    
    
        # st.markdown("""
        # <style>
        #     /* .repost-container {
        #          background-color: #f9f9f9; /* 明るい背景色 */
        #          border: 1px solid #e1e1e1; /* 細いボーダー */
        #          padding: 15px; /* 少し広めのパディング */
        #          margin-bottom: 15px; /* 下マージンも広めに */
        #          border-radius: 10px; /* 角丸を少し大きめに */
        #          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* 影を追加 */
        #      } */
        #     .repost-container {
        #         background-color: #1e1e1e;
        #         border: 1px solid #333;
        #         padding: 15px;
        #         margin-bottom: 20px;
        #         border-radius: 10px;
        #         box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);
        #     }
        #     .repost-header {
        #         display: flex;
        #         align-items: center;
        #         margin-bottom: 10px; /* 下マージン */
        #     }
        #     .repost-header img {
        #         border-radius: 50%; /* 丸いアイコン */
        #         margin-right: 10px;
        #         width: 50px; /* アイコンサイズを調整 */
        #         height: 50px; /* アイコンサイズを調整 */
        #         border: 2px solid #fff; /* 白いボーダー */
        #         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* 影を追加 */
        #     }
        #     .repost-username {
        #         font-weight: bold; /* ユーザー名を太字に */
        #         color: #ffffff; /* 少し濃いめの文字色 */
        #         margin-right: auto; /* 右端に寄せる */
        #     }
        #     .repost-time {
        #         font-size: 0.9em; /* 少し小さめのフォントサイズ */
        #         color: #777; /* 少し薄めの文字色 */
        #     }
            
        #     */.repost-content {
        #         margin-bottom: 10px;
        #         color: #555; /* 本文の文字色 */
        #         line-height: 1.5; /* 行間を調整 */
        #     } */
        #     .repost-content {
        #         margin-bottom: 10px;
        #         color: #f0f0f0;  /* 明るめの色に修正 */
        #         line-height: 1.5;
        #         white-space: pre-wrap;
        #     }
        #     .repost-link {
        #         color: #007bff; /* リンク色を強調 */
        #         text-decoration: none; /* 下線を削除 */
        #     }
        #     .repost-link:hover {
        #         text-decoration: underline; /* ホバー時に下線を表示 */
        #     }
        #     .repost-count {
        #         font-size: 1em; /* 少し大きめのフォントサイズ */
        #         color: #4CAF50; /* いい感じの緑色 */
        #         font-weight: bold;
        #     }
    
        #     .no-data-message {  /* データがない場合のメッセージのスタイル */
        #         background-color: #f9f9f9;
        #         border: 1px solid #e1e1e1;
        #         padding: 15px;
        #         margin-bottom: 15px;
        #         border-radius: 10px;
        #     }
        # </style>
        # """, unsafe_allow_html=True)
    
        # 4. リポストランキング
        # repost_ranking_df = get_repost_ranking(es, query_str, start_date_str, end_date_str)
        # st.subheader("4. クエリにヒットした投稿のリポスト数ランキング")
    
        # if not repost_ranking_df.empty:
        #     page_size = 5
        #     num_pages = (len(repost_ranking_df) // page_size) + 1
        #     page_num = st.number_input("ページ番号", min_value=1, max_value=num_pages, value=1, key="repost_page")
        #     start_index = (page_num - 1) * page_size
        #     end_index = start_index + page_size
    
        #     st.write(f"Page {page_num} of {num_pages}")  # ページ数表示改善
    
        #     # 認証情報
        #     if not bluesky_handle or not bluesky_password:
        #         st.warning("ハンドルとパスワードが設定されていません。")
        #     else:
        #         try:
        #             token = get_auth_token(bluesky_handle, bluesky_password)
        #             for index, row in repost_ranking_df[start_index:end_index].iterrows():
        #                 with st.container():  # コンテナを作成
        #                     st.write('<div class="repost-container">', unsafe_allow_html=True)
        
        #                     #元の投稿者の情報
        #                     original_avatar_url = get_avatar_url(row['original_user_name'], token)
        #                     if original_avatar_url:
        #                         original_avatar_display = f'<img src="{original_avatar_url}" width="40" alt="Profile Image">'
        #                     else:
        #                         original_avatar_display = '<span>No Avatar</span>'
        
        #                     # ヘッダー
        #                     if row['original_time']:
        #                         original_time = dt.datetime.strptime(row['original_time'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
        #                     else:
        #                         original_time = "不明"
            
        #                     st.write(f"""
        #                         <div class="repost-header">
        #                             {original_avatar_display}
        #                             <span class="repost-username">元投稿ユーザー: {row['original_user_name']}</span>
        #                             <span class="repost-time">投稿時間: {original_time}</span>
        #                         </div>
        #                         """, unsafe_allow_html=True)
        
        #                     # 元の投稿内容
        #                     st.write(f'<div class="repost-content">**投稿内容:** {row["original_text"]}</div>', unsafe_allow_html=True)
            
        #                     # リンク
        #                     st.write(f'<a href="https://bsky.app/profile/{row["original_did"]}/post/{row["uri"].split("/")[-1]}" class="repost-link">元の投稿を見る</a>', unsafe_allow_html=True)
        
        #                     # リポスト数
        #                     st.write(f'<div class="repost-count">リポスト数: {row["repost_count"]}</div>', unsafe_allow_html=True)
        
        #                     st.write('</div>', unsafe_allow_html=True)  # post-containerの閉じタグ
        #         except Exception as e:
        #             st.error(f"エラーが発生しました: {e}")
    
    
        # # 結果表示
    
        # st.markdown("""
        # <style>
        #     /* 共通のスタイル */
        #    /* .post-container {
        #         background-color: #f9f9f9;
        #         border: 1px solid #e1e1e1;
        #         padding: 15px;
        #         margin-bottom: 15px;
        #         border-radius: 10px;
        #         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        #     } */
        #     .post-container {
        #         background-color: #1e1e1e;  /* ダークグレー背景 */
        #         border: 1px solid #444444;  /* 暗いグレーの枠線 */
        #         padding: 15px;
        #         margin-bottom: 15px;
        #         border-radius: 10px;
        #         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
        #     }
        #     .post-header {
        #         display: flex;
        #         align-items: center;
        #         margin-bottom: 10px;
        #     }
        #     /*.post-header img {
        #         border-radius: 50%;
        #         margin-right: 10px;
        #         width: 50px;
        #         height: 50px;
        #         border: 2px solid #fff;
        #         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        #     }*/
        #     .post-header img {
        #         border-radius: 50%;
        #         margin-right: 10px;
        #         width: 50px;
        #         height: 50px;
        #         border: 2px solid #fff;
        #         box-shadow: 0 1px 3px rgba(255, 255, 255, 0.2);
        #     }
        #     .post-username {
        #         font-weight: bold;
        #         color: #ffffff;
        #         margin-right: auto;
        #     }
        #     .post-time {
        #         font-size: 0.9em;
        #         color: #777;
        #     }
        #     .post-content {
        #         margin-bottom: 10px;
        #         color: #ffffff;
        #         line-height: 1.5;
        #     }
        #     .post-link {
        #         color: #007bff;
        #         text-decoration: none;
        #     }
        #     .post-link:hover {
        #         text-decoration: underline;
        #     }
        #     .no-data-message {
        #         background-color: #f9f9f9;
        #         border: 1px solid #e1e1e1;
        #         padding: 15px;
        #         margin-bottom: 15px;
        #         border-radius: 10px;
        #     }
    
        #     /* Likeランキング固有のスタイル */
        #     .like-count {
        #         font-size: 1em;
        #         color: #e44d26;  /* 明るい赤色 */
        #         font-weight: bold;
        #     }
    
        #     /* リポストランキング固有のスタイル */
        #     .repost-count {
        #         font-size: 1em;
        #         color: #4CAF50;  /* いい感じの緑色 */
        #         font-weight: bold;
        #     }
        # </style>
        # """, unsafe_allow_html=True)
    
        # # 5. Likeランキング
        # like_ranking_df = get_like_ranking(es, query_str, start_date_str, end_date_str)
        # st.subheader("5. クエリにヒットした投稿のLike数ランキング")
        # if not like_ranking_df.empty:
        #     # 認証情報
        #     if not bluesky_handle or not bluesky_password:
        #         st.warning("ハンドルとパスワードが設定されていません。")
        #     else:
        #         try:
        #             token = get_auth_token(bluesky_handle, bluesky_password)
        #             page_size = 5  # 1ページあたりの表示件数
        #             page_num = st.number_input("ページ番号", min_value=1, max_value=(len(like_ranking_df) // page_size) + 1, value=1, key="like_page")
        #             start_index = (page_num - 1) * page_size
        #             end_index = start_index + page_size
        #             for index, row in like_ranking_df[start_index:end_index].iterrows():
        #                 # CSSスタイルを適用
        #                 st.markdown(f'<div class="post-container">', unsafe_allow_html=True)
        
        #                 # 元の投稿者の情報を取得
        #                 original_avatar_url = get_avatar_url(row['original_user_name'], token)
        #                 if original_avatar_url:
        #                     original_avatar_display = f'<img src="{original_avatar_url}" width="40" alt="Profile Image">'
        #                 else:
        #                     original_avatar_display = '<span>No Avatar</span>'
    
        #                 # ヘッダー
        #                 if row['original_time']:
        #                     original_time = dt.datetime.strptime(row['original_time'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
        #                 else:
        #                     original_time = "不明"
    
        #                 st.markdown(f"""
        #                     <div class="post-header">
        #                         {original_avatar_display}
        #                         <span class="post-username">元投稿ユーザー: {row['original_user_name']}</span>
        #                         <span class="post-time">投稿時間: {original_time}</span>
        #                     </div>
        #                     """, unsafe_allow_html=True)
    
        #                 # 投稿内容
        #                 st.markdown(f'<div class="post-content">**投稿内容:** {row["original_text"]}</div>', unsafe_allow_html=True)
        
        #                 # リンク
        #                 st.markdown(f'<a href="https://bsky.app/profile/{row["original_did"]}/post/{row["uri"].split("/")[-1]}" class="post-link">元の投稿を見る</a>', unsafe_allow_html=True)
    
        #                 # Like数
        #                 st.markdown(f'<div class="like-count">Like数: {row["like_count"]}</div>', unsafe_allow_html=True)
    
        #                 st.markdown('</div>', unsafe_allow_html=True)  # post-containerの閉じタグ
        #         except Exception as e:
        #             st.error(f"エラーが発生しました: {e}")
        
    
        post_cluster_df_a = cluster_post_texts(es, query_str_a, start_date_str, end_date_str, n_clusters=5)
        # 数値の cluster 列を文字列に変換して離散カテゴリとして扱う
        post_cluster_df_a["cluster_str"] = post_cluster_df_a["cluster"].astype(str)
    
        if post_cluster_df_a is not None:
            st.subheader("6-A. クエリAにヒットした投稿の投稿内容のクラスタリング")
            fig = px.scatter(
                post_cluster_df_a,
                x="x",
                y="y",
                color="cluster_str",
                hover_data=["text"],
                color_discrete_sequence=px.colors.qualitative.Plotly  # はっきりとした色分けのパレット
            )
            st.plotly_chart(fig)
        else:
            st.write("投稿のクラスタリング結果が得られませんでした。")
    
        st.subheader("全体の特徴語ワードクラウド")
        # 1. 全体のワードクラウド（全投稿の頻度を使用）
        all_texts_a = post_cluster_df_a["text"].tolist()
        cleaned_texts_a = [remove_urls(text) for text in all_texts_a] # ← URL除去
    
        # CountVectorizer を使って全体の単語頻度を計算（英語の stopwords を除外）
        vectorizer = CountVectorizer(stop_words="english")
        all_matrix = vectorizer.fit_transform(cleaned_texts_a)
        overall_freq_a = np.array(all_matrix.sum(axis=0)).flatten()  # 各単語の総出現数
        vocab = vectorizer.get_feature_names_out()
        overall_dict_a = dict(zip(vocab, overall_freq_a))
    
        # ワードクラウド生成（はっきりした色と高解像度設定）
        overall_wc_a = WordCloud(width=800, height=400, background_color="white", scale=2).generate_from_frequencies(overall_dict_a)
        plt.figure(figsize=(8,4), dpi=300)
        plt.imshow(overall_wc_a, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())
        plt.close()
    
    
        # 2. 各クラスタごとに、全体と比較して特徴的な単語を抽出してワードクラウド生成
        st.subheader("クラスタごとの特徴語ワードクラウド（全体との差分）")
    
        clusters = post_cluster_df_a["cluster"].unique()
        # 各クラスタごとに処理
        for cl in clusters:
            cluster_texts_a = post_cluster_df_a[post_cluster_df_a["cluster"] == cl]["text"].tolist()
            cleaned_cluster_texts_a = [remove_urls(text) for text in cluster_texts_a]  # ← URL除去
            # 同じ vectorizer を使って、各クラスタの単語頻度を計算
            cluster_matrix_a = vectorizer.transform(cluster_texts_a)
            cluster_freq_a = np.array(cluster_matrix_a.sum(axis=0)).flatten()
        
            # 差分スコアの計算: (cluster_freq + 1) / (overall_freq + 1)
            diff_scores_a = (cluster_freq_a + 1) / (overall_freq_a + 1)
            # ここでは、比率が 1.0 より大きい単語（クラスタでより頻出している単語）を対象とする
            diff_dict_a = {word: score for word, score in zip(vocab, diff_scores_a) if score > 1.0}
        
            # 万が一差分が空の場合は、クラスタ内の頻度をそのまま使用（または別途処理）
            if not diff_dict_a:
                diff_dict_a = {word: freq for word, freq in zip(vocab, cluster_freq_a) if freq > 0}
        
            cluster_wc_a = WordCloud(width=800, height=400, background_color="white", scale=2).generate_from_frequencies(diff_dict_a)
            st.markdown(f"**クラスタ {cl} の特徴語**")
            plt.figure(figsize=(8,4), dpi=300)
            plt.imshow(cluster_wc_a, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()
    
        post_cluster_df_b = cluster_post_texts(es, query_str_b, start_date_str, end_date_str, n_clusters=5)
        # 数値の cluster 列を文字列に変換して離散カテゴリとして扱う
        post_cluster_df_b["cluster_str"] = post_cluster_df_b["cluster"].astype(str)
    
        if post_cluster_df_b is not None:
            st.subheader("6-B. クエリBにヒットした投稿の投稿内容のクラスタリング")
            fig = px.scatter(
                post_cluster_df_b,
                x="x",
                y="y",
                color="cluster_str",
                hover_data=["text"],
                color_discrete_sequence=px.colors.qualitative.Plotly  # はっきりとした色分けのパレット
            )
            st.plotly_chart(fig)
        else:
            st.write("投稿のクラスタリング結果が得られませんでした。")
    
        st.subheader("全体の特徴語ワードクラウド")
        # 1. 全体のワードクラウド（全投稿の頻度を使用）
        all_texts_b = post_cluster_df_b["text"].tolist()
        cleaned_texts_b = [remove_urls(text) for text in all_texts_b] # ← URL除去
    
        # CountVectorizer を使って全体の単語頻度を計算（英語の stopwords を除外）
        vectorizer = CountVectorizer(stop_words="english")
        all_matrix = vectorizer.fit_transform(cleaned_texts_b)
        overall_freq_b = np.array(all_matrix.sum(axis=0)).flatten()  # 各単語の総出現数
        vocab = vectorizer.get_feature_names_out()
        overall_dict_b = dict(zip(vocab, overall_freq_b))
    
        # ワードクラウド生成（はっきりした色と高解像度設定）
        overall_wc_b = WordCloud(width=800, height=400, background_color="white", scale=2).generate_from_frequencies(overall_dict_b)
        plt.figure(figsize=(8,4), dpi=300)
        plt.imshow(overall_wc_b, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())
        plt.close()
    
    
        # 2. 各クラスタごとに、全体と比較して特徴的な単語を抽出してワードクラウド生成
        st.subheader("クラスタごとの特徴語ワードクラウド（全体との差分）")
    
        clusters = post_cluster_df_b["cluster"].unique()
        # 各クラスタごとに処理
        for cl in clusters:
            cluster_texts_b = post_cluster_df_b[post_cluster_df_b["cluster"] == cl]["text"].tolist()
            cleaned_cluster_texts_a = [remove_urls(text) for text in cluster_texts_b]  # ← URL除去
            # 同じ vectorizer を使って、各クラスタの単語頻度を計算
            cluster_matrix_b = vectorizer.transform(cluster_texts_b)
            cluster_freq_b = np.array(cluster_matrix_b.sum(axis=0)).flatten()
        
            # 差分スコアの計算: (cluster_freq + 1) / (overall_freq + 1)
            diff_scores_b = (cluster_freq_b + 1) / (overall_freq_b + 1)
            # ここでは、比率が 1.0 より大きい単語（クラスタでより頻出している単語）を対象とする
            diff_dict_b = {word: score for word, score in zip(vocab, diff_scores_b) if score > 1.0}
        
            # 万が一差分が空の場合は、クラスタ内の頻度をそのまま使用（または別途処理）
            if not diff_dict_b:
                diff_dict_b = {word: freq for word, freq in zip(vocab, cluster_freq_b) if freq > 0}
        
            cluster_wc_b = WordCloud(width=800, height=400, background_color="white", scale=2).generate_from_frequencies(diff_dict_b)
            st.markdown(f"**クラスタ {cl} の特徴語**")
            plt.figure(figsize=(8,4), dpi=300)
            plt.imshow(cluster_wc_b, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()
    
    
        # profile_cluster_df = cluster_profile_descriptions(es, n_clusters=5)
        # # 数値の cluster 列を文字列に変換して離散カテゴリとして扱う
        # profile_cluster_df["cluster_str"] = profile_cluster_df["cluster"].astype(str)
    
        # if profile_cluster_df is not None:
        #     st.subheader("7. プロフィール文のクラスタリング")
        #     fig2 = px.scatter(
        #         profile_cluster_df, 
        #         x="x", 
        #         y="y", 
        #         color="cluster_str", 
        #         hover_data=["description"],
        #         color_discrete_sequence=px.colors.qualitative.Plotly  # はっきりとした色分けのパレット
        #     )
        #     st.plotly_chart(fig2)
        # else:
        #     st.write("プロフィール文のクラスタリング結果が得られませんでした。")
    
    
        # # 例: profile_cluster_df には 'cluster' と 'description' の列があると仮定
        # # クラスタの値を文字列に変換しておく
        # profile_cluster_df["cluster_str"] = profile_cluster_df["cluster"].astype(str)
    
        # # 各クラスタごとのワードクラウドを生成
        # profile_wordclouds = generate_profile_wordclouds(profile_cluster_df, cluster_col="cluster_str", text_col="description")
    
        # st.subheader("7. プロフィール文のクラスタごとの特徴語（WordCloud）")
        # for cluster, wc in profile_wordclouds.items():
        #     st.markdown(f"**クラスタ {cluster}**")
        #     plt.figure(figsize=(8, 4), dpi=300)
        #     plt.imshow(wc, interpolation="bilinear")
        #     plt.axis("off")
        #     st.pyplot(plt.gcf())
        #     plt.close()
    
    
        # if post_cluster_df_a is not None and profile_cluster_df is not None:
        #     heat_data = create_cluster_heatmap(post_cluster_df_a, profile_cluster_df)
        #     st.subheader("8. プロフィールクラスタと投稿内容クラスタのヒートマップ")
        #     fig_heat = px.imshow(heat_data, labels=dict(x="プロフィールクラスタ", y="投稿内容クラスタ", color="件数"),
        #                      x=list(range(5)), y=list(range(5)))
        #     st.plotly_chart(fig_heat)
        # else:
        #     st.write("ヒートマップの生成に必要なデータが不足しています。")
    
    
    #     like_graph = create_like_network(es, query_str, start_date_str, end_date_str)
    #     st.subheader("9. LIKE のネットワーク表示")
    #     if like_graph.number_of_nodes() > 0:
    #         # Pyvisネットワークの作成（高さや幅は適宜調整してください）
    #         net = Network(height="600px", width="100%", notebook=True)
    #         net.from_nx(like_graph)
                
    #         # 各ノードの属性を上書きして、labelは空、titleにノードIDを設定
    #         for node in net.nodes:
    #             node['title'] = str(node['id'])  # マウスオーバー時のツールチップに表示
    #             node['label'] = ""              # ノードのラベルは非表示
        
    #         # ノード追加：常にラベルは表示せず、マウスオーバー時にtitleとして表示
    #     #    for node in like_graph.nodes():
    #     #        net.add_node(node, title=str(node), label="")  
    #     #    for source, target in like_graph.edges():
    #     #        net.add_edge(source, target)
        
    #         # HTMLとして保存
    #         net.show("like_network.html")
        
    #         # 保存したHTMLファイルを読み込み、Streamlitで表示
    #         with open("like_network.html", "r", encoding="utf-8") as html_file:
    #             source_code = html_file.read()
    #             # デフォルトでは枠が "1px solid lightgray" になっているので、"border: none" に変更
    #             source_code = source_code.replace("border: 1px solid lightgray", "border: none")
    # #            source_code = source_code.replace("border: 1px solid rgba(0, 0, 0, .125)", "border: none")
                     
    
    #         components.html(source_code, height=600, width=800)
    #     else:
    #         st.write("LIKE ネットワークのデータがありません。")
    
    
        #like_graph = create_like_network(es, query_str, start_date_str, end_date_str)
        #st.subheader("9. LIKE のネットワーク表示")
        #if like_graph.number_of_nodes() > 0:
        #    plt.figure(figsize=(6, 6))
        #    pos = nx.spring_layout(like_graph, k=0.5)
        #    nx.draw(like_graph, pos, with_labels=True, node_size=500, arrowsize=20)
        #    st.pyplot(plt)
        #else:
        #    st.write("LIKE ネットワークのデータがありません。")
    
    
        # repost_graph = create_repost_network(es, query_str, start_date_str, end_date_str)
        # st.subheader("10. リポストのネットワーク表示")
        # if repost_graph.number_of_nodes() > 0:
        #     # Pyvisネットワークの作成（表示サイズは適宜調整）
        #     net = Network(height="600px", width="100%", notebook=True)
        
        #     # NetworkXのグラフから読み込み
        #     net.from_nx(repost_graph)
        
        #     # 各ノードの属性を上書き
        #     for node in net.nodes:
        #         node['title'] = str(node['id'])  # ツールチップ用にノードIDを設定
        #         node['label'] = ""              # ラベルは空にして非表示にする
        
        #     # HTMLファイルとして出力
        #     net.show("repost_network.html")
        
        #     # 生成されたHTML内のCSSで指定されている枠線を非表示に変更
        #     with open("repost_network.html", "r", encoding="utf-8") as html_file:
        #         source_code = html_file.read()
        #     source_code = source_code.replace('border: 1px solid lightgray', 'border: none')
        
        #     # StreamlitでHTMLを埋め込み表示
        #     components.html(source_code, height=600, width=800)
        # else:
        #     st.write("リポストネットワークのデータがありません。")
    
        #repost_graph = create_repost_network(es, query_str, start_date_str, end_date_str)
        #st.subheader("10. リポストのネットワーク表示")
        #if repost_graph.number_of_nodes() > 0:
        #    plt.figure(figsize=(6, 6))
        #    pos = nx.spring_layout(repost_graph, k=0.3, iterations=50)
        #    nx.draw(repost_graph, pos, with_labels=True, node_size=500, arrowsize=20)
        #    st.pyplot(plt)
        #else:
        #    st.write("リポストネットワークのデータがありません。")
        # 11. 単語ランキング表示（Streamlit）
        st.subheader("11. 投稿単語ランキング（ストップワード・URL除去後）")
        
        # CSSスタイル
        st.markdown("""
        <style>
        .word-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }
        .word-table th, .word-table td {
            border: 1px solid #444;
            padding: 8px 12px;
            color: #fff;
        }
        .word-table th {
            background-color: #333;
            font-weight: bold;
            text-align: left;
        }
        .word-table tr:nth-child(even) {
            background-color: #2c2c2c;
        }
        .word-table tr:nth-child(odd) {
            background-color: #1e1e1e;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 関数を使ってランキングを取得
        word_ranking_a = get_word_frequency_ranking(es, query_str_a, start_date_str, end_date_str, lang_filter=language_filter)
        
        # HTMLでランキング表示
        if word_ranking_a:
            ranking_html_a = """
            <table class="word-table">
                <thead>
                    <tr><th>順位</th><th>単語</th><th>出現回数</th></tr>
                </thead><tbody>
            """
            for idx, (word, count) in enumerate(word_ranking_a, start=1):
                ranking_html_a += f"<tr><td>{idx}</td><td>{word}</td><td>{count}</td></tr>"
            ranking_html_a += "</tbody></table>"
        
            st.markdown(ranking_html_a, unsafe_allow_html=True)
        else:
            st.info("単語ランキングを生成できる投稿が見つかりませんでした。")
    
        # 関数を使ってランキングを取得
        word_ranking_b = get_word_frequency_ranking(es, query_str_b, start_date_str, end_date_str, lang_filter=language_filter)
        
        # HTMLでランキング表示
        if word_ranking_b:
            ranking_html_b = """
            <table class="word-table">
                <thead>
                    <tr><th>順位</th><th>単語</th><th>出現回数</th></tr>
                </thead><tbody>
            """
            for idx, (word, count) in enumerate(word_ranking_b, start=1):
                ranking_html_b += f"<tr><td>{idx}</td><td>{word}</td><td>{count}</td></tr>"
            ranking_html_b += "</tbody></table>"
        
            st.markdown(ranking_html_b, unsafe_allow_html=True)
        else:
            st.info("単語ランキングを生成できる投稿が見つかりませんでした。")
    
        # # === トピックモデリング（LDA） ===
        # if post_cluster_df is not None:
        #     st.subheader("12.クエリにヒットした投稿のトピックモデリング（LDA）")
        #     topic_data = perform_topic_modeling(post_cluster_df["text"].tolist(), n_topics=5, n_words=10)
        
        #     st.markdown("""
        #     <style>
        #         .topic-box {
        #             background-color: #2a2a2a;
        #             border: 1px solid #444;
        #             border-radius: 10px;
        #             padding: 10px;
        #             margin-bottom: 10px;
        #         }
        #         .topic-title {
        #             font-weight: bold;
        #             color: #4CAF50;
        #             font-size: 1.1em;
        #         }
        #         .topic-words {
        #             color: #ddd;
        #         }
        #     </style>
        #     """, unsafe_allow_html=True)
        
        #     for idx, words in topic_data:
        #         st.markdown(f"""
        #         <div class="topic-box">
        #             <div class="topic-title">トピック {idx + 1}</div>
        #             <div class="topic-words">{"、".join(words)}</div>
        #         </div>
        #         """, unsafe_allow_html=True)
    
        ##BERTopicによるトピックモデリングと可視化
        # if post_cluster_df is not None:
        #     st.subheader("13. BERTopicによるトピックモデリングと可視化")
        #     texts = post_cluster_df["text"].tolist()
        
        #     with st.spinner("BERTopic によるトピック抽出中..."):
        #         topic_model, topics, probs, cleaned_texts = run_bertopic_modeling(texts)
        
        #     render_bertopic_visualizations(topic_model,texts)
        
        # st.subheader("リポストユーザー間ネットワーク（最大連結成分）")
        # create_repost_user_network(es, query_str, start_date_str, end_date_str)
    
        # test_repost_data_fetch(es)
    
        # st.subheader("リプライネットワークの可視化")
        # create_reply_user_network(es, query_str, start_date_str, end_date_str)
    
        # topic_model, combined_topicmodeling_df, embedding_model = run_bertopic_on_dataframe(
        #     combined_post_df,
        #     text_column="text",
        #     language="multilingual",
        #     model_name="intfloat/multilingual-e5-base"
        # )
        # st.subheader("💬 BERTopic 可視化")
        
        # show_topic_summary_streamlit(topic_model)
        # show_topic_scatter_streamlit(combined_topicmodeling_df, topic_model, embedding_model)
        
        # st.subheader("引用投稿の一覧-クエリA")
        # df_quoted_posts_a = fetch_quoted_posts(
        #     es=es,
        #     query=query_str_a,
        #     start_date=start_date.isoformat(),
        #     end_date=end_date.isoformat()
        # )
        # st.dataframe(df_quoted_posts_a)
    
        # st.subheader("引用投稿の一覧-クエリB")
        # df_quoted_posts_b = fetch_quoted_posts(
        #     es=es,
        #     query=query_str_b,
        #     start_date=start_date.isoformat(),
        #     end_date=end_date.isoformat()
        # )
        # st.dataframe(df_quoted_posts_b)
        
        # st.subheader("引用投稿のネットワークの可視化-クエリA")
        # # create_quote_embed_network(es, start_date_str, end_date_str)
        # create_recursive_quote_network(post_df_a, df_quoted_posts_a)
    
        # st.subheader("引用投稿のネットワークの可視化-クエリB")
        # # create_quote_embed_network(es, start_date_str, end_date_str)
        # create_recursive_quote_network(post_df_b, df_quoted_posts_b)
        
        # st.subheader("雪だるま式のリポストネットワークの可視化-クエリA")
        # G_a = nx.DiGraph
        # G_a = create_recursive_repost_user_network_from_df_parallel(
        #     es=es,
        #     start_date=start_date,
        #     end_date=end_date,
        #     iterations=snowball_iterations,
        #     users_per_iteration=snowball_users_per_iter,
        #     posts_df=post_df_a
        #     )
    
        # st.subheader("雪だるま式のリポストネットワークの可視化-クエリB")
        # G_b = nx.DiGraph
        # G_b = create_recursive_repost_user_network_from_df_parallel(
        #     es=es,
        #     start_date=start_date,
        #     end_date=end_date,
        #     iterations=snowball_iterations,
        #     users_per_iteration=snowball_users_per_iter,
        #     posts_df=post_df_b
        #     )
    
        # st.subheader("雪だるま式のリポストネットワークの可視化-グラフの統合")
        # G_merged_repost = nx.DiGraph
        # G_merged_repost = merge_directed_graphs_on_common_nodes(G_a, G_b)
    
        st.subheader("雪だるま式の全てのインタラクションのネットワークの可視化-クエリA")
        G_a_allinter = nx.MultiGraph
        G_a_allinter, df_allinter_a = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=12,
            posts_df=post_df_a,  # 前提：既に読み込み済みのDataFrame
            label="A"
            )
    
        st.subheader("雪だるま式の全てのインタラクションのネットワークの可視化-クエリB")
        G_b_allinter = nx.MultiGraph
        G_b_allinter, df_allinter_b = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=12,
            posts_df=post_df_b,# 前提：既に読み込み済みのDataFrame
            label="B"
            )
    
        st.subheader("雪だるま式の全てのインタラクションのネットワークの可視化-グラフの統合")
        G_merged_all = nx.MultiGraph
        G_merged_all = merge_graphs_on_common_nodes(G_a_allinter, G_b_allinter)
    
        combined_df_allinter = pd.concat([df_allinter_a, df_allinter_b], ignore_index=True)
    
        topic_model, combined_df_allinter_topic, embedding_model = run_bertopic_on_dataframe(
            combined_df_allinter,
            text_column="text",
            language="multilingual",
            model_name="intfloat/multilingual-e5-base"
        )
    
        st.dataframe(combined_df_allinter_topic)
        
        st.subheader("💬 BERTopic 可視化-インタラクションネットワーク")
        
        show_topic_summary_streamlit(topic_model)
        show_topic_scatter_streamlit(combined_df_allinter_topic, topic_model, embedding_model)
    
        visualize_network_with_topics(G_merged_all, combined_df_allinter_topic)
    
        inter_matrix, neg_inter_matrix = generate_neg_inter_matrix(G_merged_all, combined_df_allinter_topic)
        
        st.subheader("SPINスコア可視化")
        result_spin = calculate_spin_score(inter_matrix, neg_inter_matrix)
        display_spin_result(result_spin, neg_inter_matrix)

    if option == "３つ以上":
        pass
    if option == "日本の政党と炎上ポスト":
        st.subheader("日本の政党の分極を計測します")
        political_party_post_df = get_post_texts_list(es, query_list, start_date, end_date)
        st.write("日本のそれぞれの政党に関する投稿取得終了")
        labeled_party_df = split_df_by_label(political_party_post_df, label_col="label")
        st.write("以下、グラフ構築開始")
        G_jimin_allinter, df_allinter_jimin = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["自民"],# 前提：既に読み込み済みのDataFrame
            label="自民"
            )
        st.write("自民党終了")
        G_rikken_allinter, df_allinter_rikken = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["立憲"],# 前提：既に読み込み済みのDataFrame
            label="立憲"
            )
        st.write("立憲民主党終了")
        G_ishin_allinter, df_allinter_ishin = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["維新"],# 前提：既に読み込み済みのDataFrame
            label="維新"
            )
        st.write("日本維新の党終了")
        G_koumei_allinter, df_allinter_koumei = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["公明"],# 前提：既に読み込み済みのDataFrame
            label="公明"
            )
        st.write("公明党終了")
        G_shamin_allinter, df_allinter_shamin = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["社民"],# 前提：既に読み込み済みのDataFrame
            label="社民"
            )
        st.write("社民党終了")
        G_reiwa_allinter, df_allinter_reiwa = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["れいわ"],# 前提：既に読み込み済みのDataFrame
            label="れいわ"
            )
        st.write("れいわ新政党終了")
        G_minshu_allinter, df_allinter_minshu = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["国民民主"],# 前提：既に読み込み済みのDataFrame
            label="国民民主"
            )
        st.write("国民民主党終了")
        G_kyousan_allinter, df_allinter_kyousan = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["共産党"],# 前提：既に読み込み済みのDataFrame
            label="共産党"
            )
        st.write("共産党終了")
        G_sansei_allinter, df_allinter_sansei = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date,
            end_date=end_date,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_party_df["参政"],# 前提：既に読み込み済みのDataFrame
            label="参政"
            )
        st.write("参政党終了")
        st.subheader("投稿を取得し、グラフを作成しました。")
        
        list_party_G = [G_jimin_allinter, G_rikken_allinter, G_ishin_allinter, G_koumei_allinter, G_shamin_allinter, G_reiwa_allinter, G_minshu_allinter, G_kyousan_allinter, G_sansei_allinter]
        
        combined_df_allinter_party = pd.concat([df_allinter_jimin, df_allinter_rikken, df_allinter_ishin, df_allinter_koumei, df_allinter_shamin, df_allinter_reiwa, df_allinter_minshu, df_allinter_kyousan, df_allinter_sansei], ignore_index=True)
        G_merged_party = merge_graphs_on_common_nodes_multi(list_party_G)

        st.subheader("作成したグラフから、ラベル伝搬を行います。")

        node_communities = label_propagation_community_detection(
            G_merged_party,
            max_iter=10,
            num_classes=None,
            tau=0.5,
            create_others=True,
            label_attr="label"
        )
        #visualize_community_network_streamlit(G_merged_party, node_communities)
        st.subheader("ラベル伝搬終了。ネガティブ行列形成を行います。")

        inter_matrix_party, neg_inter_matrix_party = generate_interaction_matrices_from_graph(G_merged_party, node_communities, combined_df_allinter_party)

        result_party = calculate_spin_score_by_label(inter_matrix_party, neg_inter_matrix_party)
        st.subheader("SPINスコアを出力します。")
        display_spin_result(result_party, neg_inter_matrix_party)

        topic_model_party, combined_df_allinter_party_topic, embedding_model_party = run_bertopic_on_dataframe(
            combined_df_allinter_party,
            text_column="text",
            language="multilingual"
        )
        
        st.subheader("BERTopicを踏まえてみてみる。")
        show_topic_summary_streamlit(topic_model_party)
        show_topic_scatter_streamlit(combined_df_allinter_party_topic, topic_model_party, embedding_model_party)
        visualize_network_with_topics(G_merged_party, combined_df_allinter_party_topic)
        st.subheader("BERTopicのノードのラベル伝搬")
        node_topics = topic_label_propagation(
            G_merged_party,
            combined_df_allinter_party_topic,
            topic_col="topic",
            prob_col="topic_prob"
        )
        inter_matrix_party_topic, neg_inter_matrix_party_topic = generate_interaction_matrices_from_graph(G_merged_party, node_topics, combined_df_allinter_party_topic)
        st.subheader("BERTopicのノードのネガティブ行列計算")
        result_party_topic = calculate_spin_score_by_label(inter_matrix_party_topic, neg_inter_matrix_party_topic)
        st.subheader("BERTopicによるSPINスコア可視化")
        display_spin_result(result_party_topic, neg_inter_matrix_party_topic)
        
        st.subheader("東京の都議選の分極を計測します")
        togisen_post_df = get_post_texts_list(es, query_list_togisen, start_date_togisen, end_date_togisen)
        st.write("日本のそれぞれの政党に関する投稿取得終了")
        labeled_togisen_df = split_df_by_label(togisen_post_df, label_col="label")
        st.write("以下、グラフ構築開始")
        G_ts_allinter, df_allinter_ts = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["都議選"],# 前提：既に読み込み済みのDataFrame
            label="都議選"
            )
        st.write("都議選終了")
        G_ts_jimin_allinter, df_allinter_ts_jimin = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["自民"],# 前提：既に読み込み済みのDataFrame
            label="自民"
            )
        st.write("自民党終了")
        G_ts_tomin_allinter, df_allinter_ts_tomin = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["都民"],# 前提：既に読み込み済みのDataFrame
            label="都民"
            )
        st.write("都民党終了")
        G_ts_koumei_allinter, df_allinter_ts_koumei = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["公明"],# 前提：既に読み込み済みのDataFrame
            label="公明"
            )
        st.write("公明党終了")
        G_ts_kyousan_allinter, df_allinter_ts_kyousan = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["共産"],# 前提：既に読み込み済みのDataFrame
            label="共産"
            )
        st.write("共産党終了")
        G_ts_rikken_allinter, df_allinter_ts_rikken = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["立憲 立民"],# 前提：既に読み込み済みのDataFrame
            label="立憲 立民"
            )
        st.write("立憲民主党終了")
        G_ts_net_allinter, df_allinter_ts_net = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["ネット"],# 前提：既に読み込み済みのDataFrame
            label="ネット"
            )
        st.write("生活者ネット党終了")
        G_ts_kokumin_allinter, df_allinter_ts_kokumin = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["国民 国民民主"],# 前提：既に読み込み済みのDataFrame
            label="国民"
            )
        st.write("国民民主党終了")
        G_ts_sansei_allinter, df_allinter_ts_sansei = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["参政"],# 前提：既に読み込み済みのDataFrame
            label="参政"
            )
        st.write("参政党終了")
        G_ts_mushozoku_allinter, df_allinter_ts_mushozoku = create_recursive_interaction_user_network_parallel(
            es=es,
            start_date=start_date_togisen,
            end_date=end_date_togisen,
            iterations=8,
            users_per_iteration=10,
            posts_df=labeled_togisen_df["無所属"],# 前提：既に読み込み済みのDataFrame
            label="無所属"
            )
        st.write("無所属終了")
        st.subheader("投稿を取得し、グラフを作成しました。")
        
        list_togisen_G = [G_ts_allinter, G_ts_jimin_allinter, G_ts_tomin_allinter, G_ts_koumei_allinter, G_ts_kyousan_allinter, G_ts_rikken_allinter, G_ts_net_allinter, G_ts_kokumin_allinter, G_ts_sansei_allinter, G_ts_mushozoku_allinter]
        
        combined_df_allinter_togisen = pd.concat([df_allinter_ts, df_allinter_ts_jimin, df_allinter_ts_tomin, df_allinter_ts_koumei, df_allinter_ts_kyousan, df_allinter_ts_rikken, df_allinter_ts_net, df_allinter_ts_kokumin, df_allinter_ts_sansei, df_allinter_ts_mushozoku], ignore_index=True)
        G_merged_togisen = merge_graphs_on_common_nodes_multi(list_togisen_G)

        st.subheader("作成したグラフから、ラベル伝搬を行います。")

        node_communities_togisen = label_propagation_community_detection(
            G_merged_togisen,
            max_iter=10,
            num_classes=None,
            tau=0.5,
            create_others=True,
            label_attr="label"
        )
        #visualize_community_network_streamlit(G_merged_party, node_communities)
        st.subheader("ラベル伝搬終了。ネガティブ行列形成を行います。")

        inter_matrix_togisen, neg_inter_matrix_togisen = generate_interaction_matrices_from_graph(G_merged_togisen, node_communities_togisen, combined_df_allinter_togisen)

        result_togisen = calculate_spin_score_by_label(inter_matrix_togisen, neg_inter_matrix_togisen)
        st.subheader("SPINスコアを出力します。")
        display_spin_result(result_togisen, neg_inter_matrix_togisen)

        topic_model_togisen, combined_df_allinter_togisen_topic, embedding_model_togisen = run_bertopic_on_dataframe(
            combined_df_allinter_togisen,
            text_column="text",
            language="multilingual"
        )
        
        st.subheader("BERTopicを踏まえてみてみる。")
        show_topic_summary_streamlit(topic_model_togisen)
        show_topic_scatter_streamlit(combined_df_allinter_togisen_topic, topic_model_togisen, embedding_model_togisen)
        visualize_network_with_topics(G_merged_togisen, combined_df_allinter_togisen_topic)
        st.subheader("BERTopicのノードのラベル伝搬")
        node_topics_togisen = topic_label_propagation(
            G_merged_togisen,
            combined_df_allinter_togisen_topic,
            topic_col="topic",
            prob_col="topic_prob"
        )
        inter_matrix_togisen_topic, neg_inter_matrix_togisen_topic = generate_interaction_matrices_from_graph(G_merged_togisen, node_topics_togisen, combined_df_allinter_togisen_topic)
        st.subheader("BERTopicのノードのネガティブ行列計算")
        result_togisen_topic = calculate_spin_score_by_label(inter_matrix_togisen_topic, neg_inter_matrix_togisen_topic)
        st.subheader("BERTopicによるSPINスコア可視化")
        display_spin_result(result_togisen_topic, neg_inter_matrix_togisen_topic)

    return

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
    import requests

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
    import requests

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

#########################
# 1. 投稿数の時系列グラフ（postindex）
#########################
def get_post_time_series(es, query, start, end, lang_filter=None):
    # 基本の must 条件
    must_clauses = [ #満たすべき条件
        {"match": {"commit.record.text": query}}, #キーワード検索
        {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}} #日付フィルター
    ]
    # lang_filter が設定されていれば、"langs" フィールドに対する term フィルタを追加
    if lang_filter:
        must_clauses.append({"term": {"commit.record.langs": lang_filter}}) #言語フィルター
    
    body = {
        "size": 0, #ヒットしたデータは不要。件数のみ集計
        "query": {
            "bool": {
                "must": must_clauses #条件
            }
        },
        "aggs": { #10分ごとの投稿集計
            "posts_over_time": {
                "date_histogram": {
                    "field": "commit.record.createdAt",
                    "fixed_interval": "10m" #ここで時間間隔の変更が出来る
                }
            }
        }
    }

#    st.write(body)
    
    #postindexという名前がついたインデックス群から検索する
    res = es.search(index="postindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout) 
    #データを成形して、表に変換
    buckets = res["aggregations"]["posts_over_time"]["buckets"]
    df = pd.DataFrame([{"time": b["key_as_string"], "count": b["doc_count"]} for b in buckets])
    return df

#########################
# 2. 投稿数のユーザランキング（postindex）
#########################
def get_post_user_ranking(es, query, start, end):
    body = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"query_string": 
                     {"query": query, "default_field": "commit.record.text"}},
                    {"range": {"indexedAt": {"gte": start, "lte": end}}}
                ]
            }
        },
        "aggs": {
            "users": {
                "terms": {
                    "field": "did.keyword",
                    "size": 100
                }
            }
        }
    }

    res = es.search(index="postindex-*", body=body, request_timeout=60, timeout="1m")

    # 安全にバケット取得
    buckets = res.get("aggregations", {}).get("users", {}).get("buckets", [])

    if not buckets:
        import pandas as pd
        return pd.DataFrame(columns=["did", "post_count"])

    df = pd.DataFrame([
        {"did": b.get("key"), "post_count": b.get("doc_count", 0)}
        for b in buckets if "key" in b
    ])

    # カラム名が存在するかチェック
    if "post_count" not in df.columns:
        df["post_count"] = 0

    return df.sort_values("post_count", ascending=False)

#########################
# 3. ブロック数のユーザランキング（blockindex）
#########################
def get_block_user_ranking(es, start, end):
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
                    "field": "did.keyword",
                    "size": 100
                }
            }
        }
    }

    res = es.search(index="blockindex-*", body=body, request_timeout=60, timeout="1m")
    buckets = res.get("aggregations", {}).get("users", {}).get("buckets", [])

    if not buckets:
        return pd.DataFrame(columns=["did", "block_count"])

    df = pd.DataFrame([
        {"did": b.get("key"), "block_count": b.get("doc_count", 0)}
        for b in buckets if "key" in b
    ])

    return df.sort_values("block_count", ascending=False)

##########################################
# Elasticsearchから取得している内容を見る関数
##########################################
def get_post_texts(es, query_body, index="postindex-*", size=1000):
    res = es.search(index=index, body=query_body, size=size)
    hits = res["hits"]["hits"]
    
    texts = []

    for h in hits:
        source = h["_source"]
        commit = source.get("commit", {})
        record = commit.get("record", {})
        text = record.get("text")
        did = source.get("did")
        rkey = commit.get("rkey")

        if text and did and rkey:
            uri = f"at://{did}/app.bsky.feed.post/{rkey}"
            web_url = f"https://bsky.app/profile/{did}/post/{rkey}"

            # --- Like Count ---
            like_query = {
                "size": 0,
                "query": {
                    "term": {
                        "commit.record.subject.uri.keyword": uri
                    }
                },
                "aggs": {
                    "like_count": {
                        "value_count": {
                            "field": "commit.record.subject.uri.keyword"
                        }
                    }
                }
            }
            like_res = es.search(index="likeindex-*", body=like_query)
            like_count = like_res.get("aggregations", {}).get("like_count", {}).get("value", 0)

            # --- Repost Count ---
            repost_query = {
                "size": 0,
                "query": {
                    "term": {
                        "commit.record.subject.uri.keyword": uri
                    }
                },
                "aggs": {
                    "repost_count": {
                        "value_count": {
                            "field": "commit.record.subject.uri.keyword"
                        }
                    }
                }
            }
            repost_res = es.search(index="repostindex-*", body=repost_query)
            repost_count = repost_res.get("aggregations", {}).get("repost_count", {}).get("value", 0)

            texts.append({
                "did": did,
                "text": text,
                "created_at": record.get("createdAt"),
                "rkey": rkey,
                "uri": uri,
                "url": web_url,
                "like_count": like_count,
                "repost_count": repost_count
            })

    return pd.DataFrame(texts)


def build_query_body(query_str, start_date, end_date):
    return {
        "query": {
            "bool": {
                "must": [
                    {"match": {"commit.record.text": query_str}},
                    {"range": {"commit.record.createdAt": {"gte": start_date, "lte": end_date}}}
                ]
            }
        }
    }

def get_post_texts_list(es, query_list, start_date, end_date, index="postindex-*", size=2000):
    all_results = []

    for query_str in query_list:
        query_body = build_query_body(query_str, start_date, end_date)
        res = es.search(index=index, body=query_body, size=size)
        hits = res["hits"]["hits"]

        for h in hits:
            source = h["_source"]
            commit = source.get("commit", {})
            record = commit.get("record", {})
            text = record.get("text")
            did = source.get("did")
            rkey = commit.get("rkey")

            if text and did and rkey:
                uri = f"at://{did}/app.bsky.feed.post/{rkey}"

                # Like count
                like_query = {
                    "size": 0,
                    "query": {
                        "term": {
                            "commit.record.subject.uri.keyword": uri
                        }
                    },
                    "aggs": {
                        "like_count": {
                            "value_count": {
                                "field": "commit.record.subject.uri.keyword"
                            }
                        }
                    }
                }
                like_res = es.search(index="likeindex-*", body=like_query)
                like_count = like_res.get("aggregations", {}).get("like_count", {}).get("value", 0)

                # Repost count
                repost_query = {
                    "size": 0,
                    "query": {
                        "term": {
                            "commit.record.subject.uri.keyword": uri
                        }
                    },
                    "aggs": {
                        "repost_count": {
                            "value_count": {
                                "field": "commit.record.subject.uri.keyword"
                            }
                        }
                    }
                }
                repost_res = es.search(index="repostindex-*", body=repost_query)
                repost_count = repost_res.get("aggregations", {}).get("repost_count", {}).get("value", 0)

                all_results.append({
                    "did": did,
                    "text": text,
                    "rkey": rkey,
                    "uri": uri,
                    "like_count": like_count,
                    "repost_count": repost_count,
                    "label": query_str
                })

    return pd.DataFrame(all_results)

##########################################
# 引数のdfにある、labelによって、dfに分割する関数
##########################################
def split_df_by_label(df, label_col="label"):
    """
    指定された列（label）で DataFrame をグループ分けし、
    各ラベルごとの DataFrame を辞書として返す。

    Parameters:
        df (pd.DataFrame): 入力データフレーム
        label_col (str): ラベルに使用する列名（デフォルトは "label"）

    Returns:
        dict[str, pd.DataFrame]: ラベル名をキー、サブDataFrameを値とする辞書
    """
    if label_col not in df.columns:
        raise ValueError(f"指定された列 '{label_col}' が DataFrame に存在しません。")

    return {label: sub_df.reset_index(drop=True) for label, sub_df in df.groupby(label_col)}

##########################################
# Elasticsearchから引用ポストについて取得する関数
##########################################

def fetch_quoted_posts(es, query, start_date: str, end_date: str):
    """
    Elasticsearch から引用ポストを取得し、自身のURIと共にDataFrame に整形する。
    """
    body = {
        "size": 8000,
        "query": {
            "bool": {
                "must": [
                    {
                        "exists": {
                            "field": "commit.record.embed.record.uri"
                        }
                    },
                    {
                        "query_string": {
                            "query": query,
                            "default_field": "commit.record.text"
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_date,
                                "lt": end_date
                            }
                        }
                    }
                ]
            }
        },
        "_source": [
            "@timestamp",
            "did",
            "commit.rkey",
            "commit.record.text",
            "commit.record.embed.record.uri"
        ],
        "sort": [
            {
                "@timestamp": {
                    "order": "asc"
                }
            }
        ]
    }

    res = es.search(index="postindex-*", body=body)
    hits = res["hits"]["hits"]

    data = []
    for hit in hits:
        src = hit["_source"]
        did = src.get("did")
        rkey = src.get("commit", {}).get("rkey")
        post_uri = f"at://{did}/app.bsky.feed.post/{rkey}" if did and rkey else None
        text = src.get("commit", {}).get("record", {}).get("text")
        quoted_uri = src.get("commit", {}).get("record", {}).get("embed", {}).get("record", {}).get("uri")
        timestamp = src.get("@timestamp")
        quoted_url = uri_to_url(quoted_uri)

        data.append({
            "timestamp": timestamp,
            "did": did,
            "text": text,
            "uri": post_uri,
            "quoted_uri": quoted_uri,
            "quoted_url": quoted_url
        })

    return pd.DataFrame(data)

def uri_to_url(uri: str) -> str:
    """
    at://形式のURIをbsky.appのURLに変換する
    """
    try:
        parts = uri.split("/")
        did = parts[2]
        rkey = parts[-1]
        return f"https://bsky.app/profile/{did}/post/{rkey}"
    except Exception:
        return ""

##########################################
# まず、postindex から対象投稿の URI を再構築する関数
##########################################
def get_post_uris_and_texts(es, query, start, end):
    # クエリ条件に合致する投稿を取得
    body = {
        "size": 1000, #必要件数に応じて調整
        "query": {
            "bool": {
                "must": [
                    {"match": {"commit.record.text": query}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        }
    }
    res = es.search(index="postindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
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
def get_repost_ranking(es, query, start, end):
    posts = get_post_uris_and_texts(es, query, start, end)
    if not posts:
        return pd.DataFrame()

    # 元投稿のURIとテキスト
    posts_df = pd.DataFrame(posts)
    if "uri" not in posts_df.columns:
        posts_df["uri"] = [p.get("uri") for p in posts]  # 明示的に列を作成

    uris = posts_df["uri"].tolist()

    # Repost情報をElasticsearchから取得
    body = {
        "size": 1000,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"commit.record.subject.uri": uris}},
                    {"term": {"commit.operation": "create"}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ]
            }
        }
    }
    res = es.search(index="repostindex-*", body=body, request_timeout=60, timeout="1m")

    repost_counts = {}
    for hit in res["hits"]["hits"]:
        subject_uri = hit["_source"]["commit"]["record"]["subject"]["uri"]
        repost_counts[subject_uri] = repost_counts.get(subject_uri, 0) + 1

    # DataFrameへ変換
    df = pd.DataFrame(list(repost_counts.items()), columns=["uri", "repost_count"])

    # マージ（ここで 'uri' が必要）
    merged = pd.merge(posts_df, df, on="uri", how="left").fillna(0)

    # repost_countをint型にする
    merged["repost_count"] = merged["repost_count"].astype(int)

    return merged.sort_values(by="repost_count", ascending=False)

##########################################
# 5. Like された数のランキング（対象投稿ごと）
##########################################
def get_like_ranking(es, query, start, end):
    posts = get_post_uris_and_texts(es, query, start, end)
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
                    "field": "commit.record.subject.uri.keyword",
                    "size": len(uris)
                }
            }
        }
    }
    res = es.search(index="likeindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
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

    if not df_aggs.empty and "did" in df_aggs.columns and "rkey" in df_aggs.columns:
        df_aggs["uri"] = df_aggs.apply(
            lambda row: f"at://{row['did']}/app.bsky.feed.post/{row['rkey']}", axis=1
        )
    else:
        # 空のDataFrameに必要な列だけ明示的に作る
        df_aggs = pd.DataFrame(columns=["did", "rkey", "like_count", "uri"])

    posts_df = pd.DataFrame(posts)
    merged = pd.merge(posts_df, df_aggs, on="uri", how="left").fillna(0)
    merged["like_count"] = merged["like_count"].astype(int)
    merged = merged.sort_values("like_count", ascending=False)
    return merged



#########################
# 6. 投稿内容のクラスタリング（postindex）
#########################
def cluster_post_texts(es, query, start, end, n_clusters=5):
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
    res = es.search(index="postindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
    docs = [hit["_source"]["commit"]["record"].get("text", "") for hit in res["hits"]["hits"] if hit["_source"]["commit"]["record"].get("text")]
    if not docs:
        return None
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)

    # X は scipy sparse matrix なので、明示的に .toarray() にする
    X_dense = X.toarray()
    
    # PCA で次元削減（2次元へ）
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_dense)
    
    # 後で profile_cluster_df に使えるようにしておく
    x_coords = X_reduced[:, 0]
    y_coords = X_reduced[:, 1]
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

#########################
# 7. プロフィール文のクラスタリング（profileindex）
#########################
def cluster_profile_descriptions(es, n_clusters=5):
    descriptions, dids = get_profile_descriptions(es)  # descriptionと対応するdidのリストを取得

    # 前処理：文字列で、空でなく、2語以上のものに限定
    docs = [d for d in descriptions if isinstance(d, str) and d.strip() and len(d.strip().split()) > 1]
    dids = [dids[i] for i in range(len(descriptions)) if isinstance(descriptions[i], str) and descriptions[i].strip() and len(descriptions[i].strip().split()) > 1]

    if not docs:
        return pd.DataFrame(columns=["did", "description", "cluster", "cluster_str", "x", "y"])

    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["did", "description", "cluster", "cluster_str", "x", "y"])

    # PCAで次元削減（2次元で可視化）
    X_dense = X.toarray()
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_dense)
    x_coords = X_reduced[:, 0]
    y_coords = X_reduced[:, 1]

    # クラスタリング
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X)

    # DataFrame 作成
    profile_cluster_df = pd.DataFrame({
        "did": dids,
        "description": docs,
        "cluster": labels,
        "cluster_str": [str(label) for label in labels],
        "x": x_coords,
        "y": y_coords
    })

    return profile_cluster_df

def get_profile_descriptions(es, index="profileindex-*"):
    body = {
        "query": {
            "exists": {
                "field": "description"
            }
        },
        "_source": ["did", "description"],
        "size": 1000
    }

    res = es.search(index=index, body=body, scroll="1m", request_timeout=60)
    scroll_id = res["_scroll_id"]
    hits = res["hits"]["hits"]

    descriptions = []
    dids = []

    while hits:
        for hit in hits:
            source = hit["_source"]
            description = source.get("description")
            did = source.get("did")
            if description and did:
                descriptions.append(description)
                dids.append(did)

        res = es.scroll(scroll_id=scroll_id, scroll="1m")
        scroll_id = res["_scroll_id"]
        hits = res["hits"]["hits"]

    return descriptions, dids

def generate_profile_wordclouds(df, cluster_col="cluster_str", text_col="description"):
    """
    各クラスタごとに、対象のプロフィール文を結合し、
    WordCloud オブジェクトを生成して返す。
    """
    from wordcloud import WordCloud

    # 各クラスタごとにテキストをまとめる
    cluster_texts = {}
    for cluster in df[cluster_col].unique():
        # texts = df[df[cluster_col] == cluster][text_col].tolist()
        # 修正後
        texts = [remove_urls(text) for text in df[df[cluster_col] == cluster][text_col].tolist()]

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

#########################
# 8. プロフィールクラスタと投稿クラスタのヒートマップ
#########################
def create_cluster_heatmap(post_df, profile_df):
    # ここでは例として、両クラスタの組み合わせごとの件数をランダムに生成するサンプルです。
    # 実際は、ユーザごとの投稿クラスタとプロフィールクラスタをマージする必要があります。
    heat_data = np.random.randint(0, 50, (5, 5))  # 5×5 のヒートマップ用サンプルデータ
    return heat_data

#########################
# 9. LIKE のネットワーク表示（likeindex）
#########################
def create_like_network(es, query, start, end):
    # 1) クエリに合致する投稿URIを取得
    posts = get_post_uris_and_texts(es, query, start, end) # ★修正箇所
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
    res = es.search(index="likeindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
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

#########################
# 10. リポストのネットワーク表示（repostindex）
#########################
def create_repost_network(es, query, start, end):
    # 1) クエリに合致する投稿URIを取得
    posts = get_post_uris_and_texts(es, query, start, end) # ★修正箇所
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
    res = es.search(index="repostindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
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

#########################
## ユーザのDIDからユーザ名をAPIを用いて取得する関数
#########################

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_usernames_parallel(did_list, token, max_workers=10):
    """
    DIDリストに対してBluesky APIを並列実行してユーザ名を取得する。

    Parameters:
        did_list (list): ユニークなDIDのリスト
        token (str): JWTトークン
        max_workers (int): 並列スレッド数

    Returns:
        dict: {did: username} のマッピング
    """
    result = {}
    cache = {}

    def task(did):
        if did in cache:
            return did, cache[did]
        username = get_username_from_did(did, token)
        cache[did] = username
        return did, username

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_did = {executor.submit(task, did): did for did in set(did_list)}
        for future in as_completed(future_to_did):
            did, username = future.result()
            result[did] = username
    return result

##############################
# ユーザのDIDからディスプレイネームをAPIを用いて取得する関数
##############################

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def fetch_display_info_parallel(did_list, token, max_workers=10):
    """
    各DIDに対応する handle と displayName を並列で取得する。
    
    Parameters:
        did_list (list of str): DIDのリスト
        token (str): JWTトークン
        max_workers (int): スレッド数

    Returns:
        dict: {did: {"handle": ..., "displayName": ...}} の辞書
    """
    cache = {}

    def fetch(did):
        if did in cache:
            return did, cache[did]
        url = f"https://bsky.social/xrpc/app.bsky.actor.getProfile?actor={did}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            info = {
                "handle": data.get("handle"),
                "displayName": data.get("displayName")
            }
            cache[did] = info
            return did, info
        except Exception as e:
            return did, {"handle": did, "displayName": "取得失敗"}

    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_did = {executor.submit(fetch, did): did for did in set(did_list)}
        for future in as_completed(future_to_did):
            did, info = future.result()
            result[did] = info
    return result
#################################
# テキストの中の無駄なURLの単語を除去する関数 
#################################

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

#################################
# 投稿データから単語を抽出し、単語ランキングを作成
#################################
def get_word_frequency_ranking(es, query, start, end, lang_filter=None, top_n=25):
    # 1. Elasticsearch から投稿を取得
    must_clauses = [
        {"match": {"commit.record.text": query}},
        {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
    ]
    if lang_filter:
        must_clauses.append({"term": {"commit.record.langs": lang_filter}})
    
    body = {
        "size": 1000,
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }
    res = es.search(index="postindex-*", body=body, request_timeout=es_request_timeout, timeout=es_timeout)
    texts = [hit["_source"]["commit"]["record"].get("text", "") for hit in res["hits"]["hits"]]
    
    # 2. 前処理（URL除去 + 空でないもの）
    cleaned_texts = [remove_urls(t) for t in texts if t.strip()]
    
    # 3. CountVectorizerでストップワードを除いた頻度集計
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(cleaned_texts)
    vocab = vectorizer.get_feature_names_out()
    counts = matrix.sum(axis=0).A1  # flatten sparse matrix
    
    # 4. 頻度辞書 → ランキング形式
    word_freq = list(zip(vocab, counts))
    word_freq.sort(key=lambda x: x[1], reverse=True)
    
    return word_freq[:top_n]  # 上位N件を返す
    
#################################
# 投稿データからトピックモデリングを行い、そのクエリの所属する話題を調べる
#################################

def perform_topic_modeling(texts, n_topics=5, n_words=10):
    """
    LDAを使ってトピックモデリングを行い、各トピックの上位語を返す

    Parameters:
        texts (list of str): 投稿テキスト
        n_topics (int): トピック数
        n_words (int): 各トピックで表示する語数

    Returns:
        topic_words (list of tuples): [(topic_index, [word1, word2, ...]), ...]
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # 前処理（URL除去、ストップワード除外）
    cleaned_texts = [remove_urls(t) for t in texts]
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(cleaned_texts)

    lda_model = LDA(n_components=n_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [words[i] for i in top_words_idx]
        topic_words.append((topic_idx, top_words))

    return topic_words

#################################
# 投稿データからトピックモデリングを行う。（BERTopic）
#################################
    
def run_bertopic_on_dataframe(df, text_column="text", language="english", model_name="intfloat/multilingual-e5-large"):
    """
    DataFrame（texts列含む）にBERTopicを適用し、各投稿にトピック番号と確信度を付与したDataFrameを返す。
    前処理には以下を含む：
    - URL除去
    - HTMLタグ除去
    - 記号・数字除去
    - 小文字化
    - ストップワード除去（CountVectorizerにより）
    """

    # --- チェック ---
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' 列が DataFrame に存在しません。")

    # --- 前処理関数 ---
    def clean_text(text):
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # URL除去
        text = re.sub(r"<.*?>", '', text)                    # HTMLタグ除去
        text = re.sub(r"[^a-zA-Z\u4e00-\u9fa5\u3040-\u309F\u30A0-\u30FF\s]", '', text)  # 記号・数字除去
        return text.lower().strip()                          # 小文字化＋空白除去

    # --- テキスト抽出＆前処理 ---
    cleaned_texts = df[text_column].fillna("").astype(str).apply(clean_text).tolist()

    # --- モデル準備 ---
    embedding_model = SentenceTransformer(model_name)
        # --- ストップワード対応 Vectorizer ---
    if language == "english":
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.10, max_df=0.50)
    else:
        vectorizer_model = CountVectorizer(stop_words=None, ngram_range=(1, 2), min_df=0.10, max_df=0.50)  # 多言語用に除去なし（必要なら後でカスタム）
    hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    # --- BERTopic 実行 ---
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        ctfidf_model=ctfidf_model,
        language=language,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(cleaned_texts)

    # --- 元のDataFrameにトピック番号・確信度を追加 ---
    result_df = df.copy()
    result_df["topic"] = topics
    topic_probs_1d = []
    
    for i in range(len(topics)):
        topic_id = topics[i]
        if topic_id == -1 or probs[i] is None:
            topic_probs_1d.append(np.nan)
        else:
            # 明示的に float に変換（防御的）
            topic_probs_1d.append(float(probs[i][topic_id]))
    
    # これで確実に 1D リスト → シリーズに変換可能
    result_df["topic_prob"] = topic_probs_1d

    return topic_model, result_df, embedding_model

#################################
# 投稿データからトピックモデリングを行ったものに対して可視化を行う。機能部部は別関数。（BERTopic）
#################################
def show_topic_summary_streamlit(topic_model):
    """
    トピック概要をStreamlit上に表形式で表示する。
    """
    if not topic_model:
        st.error("❌ topic_model が未定義です。BERTopic モデルを渡してください。")
        return

    topic_info_df = topic_model.get_topic_info()

    # 列名調整（見やすく）
    topic_info_df = topic_info_df.rename(columns={
        "Topic": "トピックID",
        "Count": "投稿数",
        "Name": "名前",
        "Representation": "代表語（上位）"
    })

    st.subheader("🧾 トピック概要テーブル")
    st.dataframe(topic_info_df, use_container_width=True)

def show_topic_scatter_streamlit(df, topic_model, embedding_model, text_column="text"):
    """
    投稿を2次元にマッピングし、トピックごとに色分けしてStreamlit上に表示（Plotly使用）
    """
    from umap import UMAP
    import pandas as pd
    import plotly.express as px

    if text_column not in df.columns or "topic" not in df.columns:
        st.error(f"❌ '{text_column}' または 'topic' 列が df に存在しません。")
        return

    st.subheader("📊 2次元トピック分布プロット")

    # テキスト埋め込み
    with st.spinner("🔍 テキストをベクトル化中..."):
        embeddings = embedding_model.encode(df[text_column].tolist(), show_progress_bar=False)

    # 2次元へ圧縮（UMAP）
    reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
    reduced = reducer.fit_transform(embeddings)

    # Plotly 用の DataFrame を作成（x/y列を追加）
    plot_df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "topic": df["topic"].astype(str),
        "text": df[text_column]
    })

    # プロット
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="topic",
        hover_data=["text"],
        title="トピック分布（2次元マッピング）",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bertopic_visualizations(model, docs):
    """
    BERTopic の可視化図を Streamlit に表示する
    """
    plots = [
        ("トピックの頻度", model.visualize_barchart()),
        ("トピックの階層構造", model.visualize_hierarchy()),
        ("トピック同士の関係（ネットワーク）", model.visualize_topics()),
        ("トピック間の類似度マップ（ヒートマップ）", model.visualize_heatmap()),
        ("各文書のトピック可視化（UMAP）", model.visualize_documents(docs)),
        ("用語の重要度の比較", model.visualize_term_rank())
    ]

    for title, fig in plots:
        st.subheader(title)
        components.html(fig.to_html(), height=600, scrolling=True)
######################################################
# 得られた投稿群から、jaccard係数による閾値のもと、Grivan-Newmanクラスタリングによって単語共起ネットワークを形成する
######################################################
def build_cooccurrence_network_with_jaccard_from_es(es, query, start_date, end_date, lang_filter=None, 
                                                     threshold=2, threshold_jaccard=0.1, 
                                                     clustering_method="Girvan-Newman"):
    """
    Elasticsearch から投稿を取得し、共起ネットワークを構築し可視化する関数
    """
    # 投稿データを取得
    posts = get_post_uris_and_texts(es, query, start_date, end_date)
    texts = [p['text'] for p in posts if p['text']]

    if not texts:
        st.warning("⚠️ 投稿が見つかりませんでした。クエリを変更して試してください。")
        return None, None, None

    # トークン化（簡易版: 空白区切り）
    tokenized_texts = [t.strip() for t in texts if t.strip()]

    # 単語→投稿インデックス
    word_docs = defaultdict(set)
    for idx, text in enumerate(tokenized_texts):
        for w in set(text.split()):
            word_docs[w].add(idx)

    if not word_docs:
        st.warning("⚠️ 共起する単語が見つかりませんでした。クエリを変更して試してください。")
        return None, None, None

    # 共起情報とネットワーク構築
    G = nx.Graph()
    cooccurrence_list = []
    for w1, w2 in combinations(word_docs.keys(), 2):
        shared = word_docs[w1] & word_docs[w2]
        union = word_docs[w1] | word_docs[w2]
        freq = len(shared)
        if len(union) == 0:
            continue
        jaccard_score = freq / len(union)
        if freq >= threshold and jaccard_score >= threshold_jaccard:
            G.add_edge(w1, w2, weight=jaccard_score)
            cooccurrence_list.append({
                "単語ペア": f"{w1} & {w2}",
                "共起数": freq,
                "Jaccard係数": jaccard_score
            })

    if not cooccurrence_list:
        st.warning("⚠️ 共起関係が見つかりませんでした。クエリを変更して試してください。")
        return None, None, None

    cooccurrence_df = pd.DataFrame(cooccurrence_list).sort_values(
        ["共起数", "Jaccard係数"], ascending=False).reset_index(drop=True)
    st.markdown("### 💡 共起ランキング（頻度×Jaccard係数）")
    st.dataframe(cooccurrence_df, hide_index=True)

    if G.number_of_nodes() == 0:
        st.warning("⚠️ ネットワークが作成できませんでした。クエリを変更して試してください。")
        return None, None, None

    G_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    degree_centrality = nx.degree_centrality(G_cc)
    degree_df = pd.DataFrame({
        "単語": list(degree_centrality.keys()),
        "中心性": list(degree_centrality.values())
    }).sort_values("中心性", ascending=False).reset_index(drop=True)
    st.markdown("### 🔝 次数中心性ランキング（最大連結成分）")
    st.dataframe(degree_df, hide_index=True)

    st.markdown(f"🔍 クラスタリング手法: **{clustering_method}** を使用します")

    if clustering_method == "Louvain":
        import community as community_louvain
        partition = community_louvain.best_partition(G_cc)
    elif clustering_method == "Label Propagation":
        communities = nx.community.label_propagation_communities(G_cc)
        partition = {node: idx for idx, comm in enumerate(communities) for node in comm}
    elif clustering_method == "Girvan-Newman":
        comp = nx.community.girvan_newman(G_cc)
        for _ in range(10):
            communities = next(comp)
        partition = {node: idx for idx, comm in enumerate(communities) for node in comm}
    else:
        st.warning(f"⚠️ 不正なクラスタリング手法が指定されました: {clustering_method}")
        return None, None, None

    color_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Alphabet
    node_colors = [color_palette[partition[node] % len(color_palette)] for node in G_cc.nodes()]

    pos = nx.kamada_kawai_layout(G_cc)
    edge_traces = []
    for u, v in G_cc.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = G_cc[u][v]["weight"]
        edge_traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                      line=dict(width=0.5 + weight * 5, color='gray'), hoverinfo='none'))

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G_cc.nodes()],
        y=[pos[n][1] for n in G_cc.nodes()],
        mode='markers+text',
        text=list(G_cc.nodes()),
        textposition="top center",
        hovertext=[f"{n}<br>中心性: {degree_centrality[n]:.4f}<br>クラスタ: {partition[n]}" for n in G_cc.nodes()],
        marker=dict(size=[5 + degree_centrality[n] * 50 for n in G_cc.nodes()],
                    color=node_colors, opacity=0.8),
        textfont=dict(size=8, color="black"),
        hoverinfo="text"
    )

    fig_net = go.Figure(data=edge_traces + [node_trace])
    fig_net.update_layout(title="単語共起ネットワーク（最大連結成分）",
                          showlegend=False, hovermode='closest',
                          margin=dict(l=0, r=0, b=0, t=40),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_net, use_container_width=True)

    return degree_df, cooccurrence_df, G_cc
    
######################################################
# リポストネットワーク
######################################################

def create_repost_user_network(es, query, start, end):
    st.markdown("リポストネットワーク作成")
    index = "repostindex-*"
    size = 1000
    html_path = "/tmp/repost_user_network.html"

    # Elasticsearch クエリ
    body = {
        "size": size,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"commit.operation": "create"}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}},
                    {"match": {"commit.record.type_": "app.bsky.feed.repost"}}
                ]
            }
        }
    }

    try:
        res = es.search(index=index, body=body)
    except Exception as e:
        st.error(f"Elasticsearch エラー: {e}")
        return

    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        st.warning("該当するリポストデータが見つかりませんでした。")
        return

    edges = set()
    nodes = set()

    for hit in hits:
        src = hit.get("_source", {})
        commit = src.get("commit", {})
        record = commit.get("record", {})
        subject = record.get("subject", {})
        subject_uri = subject.get("uri")
        user_did = src.get("did")

        if subject_uri and user_did:
            nodes.add(user_did)
            nodes.add(subject_uri)  # 投稿もノードとして追加
            edges.add((subject_uri, user_did))  # 投稿 → リポストユーザー

    if not nodes or not edges:
        st.warning("ネットワークを描画するための十分なデータがありません。")
        return

    # ネットワーク生成
    net = Network(height="800px", width="100%", directed=True, notebook=False)

    for node in nodes:
        label = node.split("/")[-1] if "at://" in node else node[-6:]  # DIDやURIの一部をラベルに
        net.add_node(node, label=label)

    for src, dst in edges:
        net.add_edge(src, dst)

    try:
        net.save_graph(html_path)
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=800, scrolling=True)
    except Exception as e:
        st.error(f"ネットワークの描画に失敗しました: {e}")

######################################################
# リプライネットワーク
######################################################
def create_reply_user_network(es, query, start, end):
    index = "postindex-*"
    html_path = "output/reply_user_network.html"
    size = 2000

    # クエリ本体
    body = {
        "size": size,
        "_source": [
            "did",
            "commit.record.reply.parent.uri"
        ],
        "query": {
            "bool": {
                "must": [
                    {"query_string": {"query": query, "default_field": "commit.record.text"}},
                    {"range": {"commit.record.createdAt": {"gte": start, "lte": end}}}
                ],
                "filter": [
                    {"term": {"commit.operation": "create"}},
                    {"term": {"commit.record.type_": "app.bsky.feed.post"}}
                ]
            }
        }
    }

    try:
        res = es.search(index=index, body=body, request_timeout=60)
    except Exception as e:
        st.error(f"Elasticsearch エラー: {e}")
        return

    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        st.warning("該当する投稿データが見つかりませんでした。")
        return

    nodes = set()
    edges = set()

    for hit in hits:
        src = hit.get("_source", {})
        user_did = src.get("did")
        parent_uri = src.get("commit", {}).get("record", {}).get("reply", {}).get("parent", {}).get("uri")

        if user_did:
            nodes.add(user_did)

        # 親投稿の URI が存在する場合（リプライである場合）
        if user_did and parent_uri:
            try:
                parent_rkey = parent_uri.split("/")[-1]
                parent_did_uri = parent_uri.split("/")[2]

                # Elasticsearchで親投稿のDIDを逆引き
                parent_res = es.search(index=index, body={
                    "size": 1,
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"did.keyword": parent_did_uri}},
                                {"term": {"commit.rkey.keyword": parent_rkey}}
                            ]
                        }
                    }
                })

                parent_hit = parent_res.get("hits", {}).get("hits", [])
                if parent_hit:
                    parent_did = parent_hit[0]["_source"].get("did")
                    if parent_did:
                        nodes.add(parent_did)
                        edges.add((user_did, parent_did))
            except Exception as e:
                # 無視して継続
                continue

    # ネットワーク可視化（pyvis）
    net = Network(height="750px", width="100%", notebook=False, directed=True)

    for node in nodes:
        net.add_node(node, label=node.split(":")[-1])  # DIDの一部を表示

    for source, target in edges:
        net.add_edge(source, target)

    if not os.path.exists(os.path.dirname(html_path)):
        os.makedirs(os.path.dirname(html_path))

    try:
        net.show(html_path)
        st.success(f"✅ リプライネットワークを生成しました: `{html_path}`")
        with open(html_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        st.components.v1.html(html_code, height=600, width=800)
    except Exception as e:
        st.error(f"❌ ネットワーク描画に失敗しました: {e}")

######################################################
# 引用ポストのネットワーク
######################################################

def create_quote_embed_network(es, start_date, end_date, max_nodes=200):
    import streamlit as st
    import json
    from pyvis.network import Network
    import os

    st.subheader("🔗 引用ネットワークの可視化")

    def fetch_embeds(limit=1000):
        """ESから event.original を含む post を検索し、embed.uri 情報を抽出する"""
        query = {
            "size": limit,
            "_source": ["did", "event.original"],
            "query": {
                "bool": {
                    "must": [
                        {"range": {"commit.record.createdAt": {"gte": start_date, "lte": end_date}}},
                        {"term": {"commit.collection": "app.bsky.feed.post"}},
                        {"term": {"commit.operation": "create"}}
                    ]
                }
            }
        }
        try:
            res = es.search(index="postindex-*", body=query)
            return res.get("hits", {}).get("hits", [])
        except Exception as e:
            st.error(f"検索失敗: {e}")
            return []

    def parse_embed_uri(event_str):
        """event.original の文字列から引用された URI を取り出す"""
        try:
            obj = json.loads(event_str)
            embed = obj.get("commit", {}).get("record", {}).get("embed", {})
            if embed.get("$type") == "app.bsky.embed.record":
                return embed.get("record", {}).get("uri")
            return None
        except json.JSONDecodeError:
            return None

    edges = set()
    nodes = set()

    hits = fetch_embeds(limit=max_nodes)
    st.write(f"対象投稿数: {len(hits)} 件")

    for hit in hits:
        did = hit["_source"].get("did")
        event_str = hit["_source"].get("event.original", "")
        quote_uri = parse_embed_uri(event_str)
        if quote_uri and did:
            edges.add((did, quote_uri))
            nodes.add(did)
            nodes.add(quote_uri)

    if not edges:
        st.warning("引用関係が見つかりませんでした。")
        return

    # 可視化
    net = Network(height="750px", width="100%", notebook=False, directed=True)
    for node in nodes:
        net.add_node(node, label=node[-6:], title=node)
    for src, tgt in edges:
        net.add_edge(src, tgt, title="quoted")

    os.makedirs("output", exist_ok=True)
    html_path = "output/quote_embed_network.html"
    try:
        net.show(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=750, scrolling=True)
    except Exception as e:
        st.error(f"ネットワーク描画失敗: {e}")


######################################################
# 引用ポストのネットワーク2
######################################################
def create_recursive_quote_network(df_posts, df_quotes, html_path="output/quote_network.html"):
    # URI → text マップ
    uri_to_text = {
        row["uri"]: row["text"]
        for _, row in df_posts.iterrows()
        if row.get("uri") and row.get("text")
    }

    # 引用投稿: uri → quoted_uri
    quote_map = {
        row["uri"]: row["quoted_uri"]
        for _, row in df_quotes.iterrows()
        if row.get("uri") and row.get("quoted_uri")
    }

    # URI → 投稿行 (全投稿を検索対象にするため)
    all_uri_rows = {
        row["uri"]: row
        for _, row in df_posts.iterrows()
        if row.get("uri")
    }

    # 一時的にノードとエッジを蓄積
    edges = set()
    node_connection_count = defaultdict(int)
    visited_uri = set()

    def traverse_chain(current_uri):
        if current_uri in visited_uri:
            return
        visited_uri.add(current_uri)

        quoted_uri = quote_map.get(current_uri)
        if not quoted_uri:
            return

        # エッジ登録
        edges.add((current_uri, quoted_uri))
        node_connection_count[current_uri] += 1
        node_connection_count[quoted_uri] += 1

        # 再帰的にたどる
        if quoted_uri in quote_map:
            traverse_chain(quoted_uri)
        elif quoted_uri in all_uri_rows:
            node_connection_count[quoted_uri] += 0  # ノードとしては登録するが末端

    for start_uri in df_quotes["uri"].dropna().unique():
        traverse_chain(start_uri)

    # フィルタリング: 双方の接続数が1のみ（孤立2ノード）を除外
    filtered_edges = {
        (src, tgt)
        for src, tgt in edges
        if not (node_connection_count[src] == 1 and node_connection_count[tgt] == 1)
    }

    if not filtered_edges:
        st.warning("3つ以上つながった投稿が見つかりませんでした。")
        return

    # pyvis ネットワーク構築
    net = Network(height="750px", width="100%", notebook=False, directed=True)
    added_nodes = set()

    for src, tgt in filtered_edges:
        for uri in [src, tgt]:
            if uri not in added_nodes:
                label = uri_to_text.get(uri, "")[:50] + "..."
                net.add_node(uri, label=label)
                added_nodes.add(uri)
        net.add_edge(src, tgt)

    # HTML保存・描画
    try:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        net.save_graph(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=750, scrolling=True)
        st.success(f"引用ネットワークを生成しました: {html_path}")
    except Exception as e:
        st.error(f"ネットワーク描画に失敗しました: {e}")

######################################################
# データが取得できているかテストする関数
######################################################

def test_repost_data_fetch(es, index_pattern="repostindex-*", sample_size=5):
    st.subheader("🧪 Repost データ取得テスト")
    st.write("Elasticsearch からサンプルデータを取得しています...")

    try:
        body = {
            "size": sample_size,
            "query": {
                "match_all": {}
            },
            "sort": [{"@timestamp": {"order": "desc"}}]
        }

        res = es.search(index=index_pattern, body=body)
        hits = res.get("hits", {}).get("hits", [])

        if not hits:
            st.warning("⚠ データが見つかりませんでした。")
            return

        st.success(f"✅ {len(hits)} 件のサンプルデータを取得しました。")

        for i, hit in enumerate(hits):
            src = hit.get("_source", {})
            commit = src.get("commit", {})
            record = commit.get("record", {})
            subject = record.get("subject", {})
            via = record.get("via", {})

            st.markdown(f"### 🔹 Sample {i + 1}")
            st.json({
                "commit.operation": commit.get("operation"),
                "commit.record.createdAt": record.get("createdAt"),
                "commit.record.type_": record.get("type_"),
                "did (user)": src.get("did"),
                "subject.uri": subject.get("uri"),
                "via.uri": via.get("uri"),
                "@timestamp": src.get("@timestamp")
            })

    except Exception as e:
        st.error(f"❌ Elasticsearch クエリ中にエラーが発生しました: {e}")

######################################################
# 雪だるま式のリポストネットワーク
######################################################

def create_recursive_repost_user_network_from_df_parallel(
    es,
    start_date,
    end_date,
    iterations=3,
    users_per_iteration=5,
    posts_df=None,
    max_workers=10
):
    st.subheader("☃️ 並列高速・雪だるま式リポストネットワーク（投稿データから）")

    if posts_df is None or posts_df.empty or "uri" not in posts_df.columns:
        st.error("posts_dfが空または 'uri' 列がありません")
        return

    popular_posts = posts_df[posts_df["repost_count"] >= 20]
    st.write(f"🎯 条件に合致した投稿数（repost_count ≥ 20）: {len(popular_posts)}")

    if popular_posts.empty:
        st.error("リポスト20件以上の投稿が見つかりませんでした")
        return

    root_post_uri = random.choice(popular_posts["uri"].dropna().tolist())
    st.success(f"初期投稿 URI: `{root_post_uri}`")

    def get_users_who_reposted(post_uri):
        query = {
            "size": users_per_iteration,
            "_source": ["did"],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"commit.record.subject.uri.keyword": post_uri}},
                        {"term": {"commit.collection": "app.bsky.feed.repost"}},
                        {"term": {"commit.operation": "create"}},
                        {"range": {"commit.record.createdAt": {"gte": start_date, "lte": end_date}}}
                    ]
                }
            }
        }
        res = es.search(index="repostindex-*", body=query)
        return [hit["_source"]["did"] for hit in res["hits"]["hits"] if "did" in hit["_source"]]

    def get_popular_post_uris_by_user(user_did):
        q = {
            "size": users_per_iteration,
            "_source": ["did", "commit.rkey", "commit.record.createdAt"],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"did": user_did}},
                        {"term": {"commit.collection": "app.bsky.feed.post"}},
                        {"term": {"commit.operation": "create"}},
                        {"range": {"commit.record.createdAt": {"gte": start_date, "lte": end_date}}}
                    ]
                }
            }
        }
        res = es.search(index="postindex-*", body=q)
        uri_map = {}
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            rkey = src["commit"]["rkey"]
            did = src["did"]
            uri = f"at://{did}/app.bsky.feed.post/{rkey}"
            uri_map[uri] = True
        if not uri_map:
            return []
        terms_q = {
            "size": 0,
            "query": {"terms": {"commit.record.subject.uri.keyword": list(uri_map.keys())}},
            "aggs": {"reposts": {"terms": {"field": "commit.record.subject.uri.keyword", "size": len(uri_map)}}}
        }
        reposts = es.search(index="repostindex-*", body=terms_q)
        counts = {b["key"]: b["doc_count"] for b in reposts["aggregations"]["reposts"]["buckets"]}
        return [uri for uri, cnt in counts.items() if cnt >= 5]

    G = nx.DiGraph()
    current_post_uris = [root_post_uri]

    for i in range(iterations):
        st.write(f"🔁 Iteration {i+1}/{iterations} - 投稿数: {len(current_post_uris)}")
        next_post_uris = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_uri = {
                pool.submit(get_users_who_reposted, uri): uri
                for uri in current_post_uris
            }
            for fut in as_completed(future_to_uri):
                post_uri = future_to_uri[fut]
                try:
                    users = fut.result(timeout=30)
                except Exception as e:
                    st.write(f"⚠️ {post_uri} の取得中エラー: {e}")
                    continue
                if not users:
                    continue
                sample = random.sample(users, min(users_per_iteration, len(users)))
                original = post_uri.split("/")[2]
                for user in sample:
                    G.add_edge(user, original)
                    # 投稿取得を並列で
                # 投稿URI収集
                post_futs = {
                    pool.submit(get_popular_post_uris_by_user, u): u
                    for u in sample
                }
                for pf in as_completed(post_futs):
                    try:
                        uris = pf.result(timeout=30)
                        next_post_uris.extend(uris)
                    except Exception as e:
                        continue

        if not next_post_uris:
            st.warning("📉 次の投稿が見つかりませんでした。終了します。")
            break
        current_post_uris = next_post_uris

    if G.number_of_edges() == 0:
        st.warning("⚠️ ネットワークが構築されませんでした。")
        return

    st.success(f"✅ 完成ネットワーク：{G.number_of_nodes()} ノード, {G.number_of_edges()} エッジ")
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(G)

    html_path = "output/repost_recursive_network_parallel.html"
    os.makedirs("output", exist_ok=True)
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=750, scrolling=True)

    return G

######################################################
# 雪だるま式のインタラクションネットワーク
######################################################

def create_recursive_interaction_user_network_parallel(
    es,
    start_date,
    end_date,
    iterations=3,
    users_per_iteration=10,
    posts_df=None,
    max_workers=10,
    label=None
):
    st.subheader("🌐 並列高速マルチインタラクションネットワーク")

    G = nx.MultiDiGraph()

    if posts_df is None or posts_df.empty or "uri" not in posts_df.columns:
        st.error("posts_dfが空または 'uri' 列がありません")
        return

    candidate_posts = posts_df[(posts_df["like_count"] >= 10) & (posts_df["repost_count"] >= 10)]
    if candidate_posts.empty:
        st.error("likeとrepost両方10件以上の投稿が見つかりませんでした")
        return

    root_post_uri = random.choice(candidate_posts["uri"].dropna().tolist())
    # 1. 初期投稿の投稿者（did）を取得
    initial_row = posts_df[posts_df["uri"] == root_post_uri].head(1)
    if not initial_row.empty and "label" in initial_row.columns:
        initial_did = initial_row["did"].values[0]
        initial_label = initial_row["label"].values[0]
        G.add_node(initial_did, label=initial_label)  # ラベルを明示的にノード属性に付加
        st.info(f"🎯 初期ユーザ `{initial_did}` にラベル `{initial_label}` を付加しました")

    st.success(f"初期投稿 URI: `{root_post_uri}`")

    def get_users_interaction(post_uri):
        futures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures['like'] = pool.submit(
                lambda: es.search(index="likeindex-*", body={"size": users_per_iteration,
                    "query":{"bool":{"must":[{"term":{"commit.record.subject.uri.keyword": post_uri}},
                                               {"range":{"commit.record.createdAt":{"gte": start_date, "lte": end_date}}}]}}}
                )
            )
            futures['repost'] = pool.submit(
                lambda: es.search(index="repostindex-*", body={"size": users_per_iteration,
                    "query":{"bool":{"must":[{"term":{"commit.record.subject.uri.keyword": post_uri}},
                                               {"range":{"commit.record.createdAt":{"gte": start_date, "lte": end_date}}}]}}}
                )
            )
            futures['reply'] = pool.submit(
                lambda: es.search(index="postindex-*", body={"size": users_per_iteration,
                    "query":{"bool":{"must":[{"term":{"commit.collection":"app.bsky.feed.post"}},
                                               {"term":{"commit.record.reply.parent.uri.keyword": post_uri}},
                                               {"range":{"commit.record.createdAt":{"gte": start_date, "lte": end_date}}}]}}}
                )
            )
        result = {}
        for key, fut in futures.items():
            try:
                res = fut.result(timeout=30)
                result[key] = [h["_source"]["did"] for h in res["hits"]["hits"] if "did" in h["_source"]]
            except Exception as e:
                result[key] = []
                st.write(f"⚠️ {post_uri} の {key} 集計でエラー: {e}")
        return result['like'], result['repost'], result['reply']

    def get_user_posts(did):
        res = es.search(index="postindex-*", body={
            "size": users_per_iteration,
            "_source": ["did", "commit.rkey", "commit.record.text"],
            "query": {
                "bool": {
                    "must": [
                        {"term":{"did":did}},
                        {"term":{"commit.collection":"app.bsky.feed.post"}},
                        {"term":{"commit.operation":"create"}},
                        {"range":{"commit.record.createdAt":{"gte":start_date, "lte":end_date}}}
                    ]
                }
            }
        })
        posts = []
        for h in res["hits"]["hits"]:
            src = h["_source"]
            if "commit" in src and "rkey" in src["commit"]:
                uri = f"at://{src['did']}/app.bsky.feed.post/{src['commit']['rkey']}"
                text = src.get("commit", {}).get("record", {}).get("text", "")
                posts.append({"did": src["did"], "text": text, "uri": uri, "label": label})
        return posts

    seen_users = set()
    current_post_uris = [root_post_uri]
    all_posts = []

    for i in range(iterations):
        st.write(f"🔁 Iteration {i+1}/{iterations} - 投稿数: {len(current_post_uris)}")
        next_post_uris = []
        futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for post_uri in current_post_uris:
                futures.append(pool.submit(get_users_interaction, post_uri))

            for fut, post_uri in zip(as_completed(futures), current_post_uris):
                like_users, repost_users, reply_users = fut.result()
                original = post_uri.split("/")[2]

                for u in like_users:
                    G.add_edge(u, original, color="rgba(0, 0, 255, 0.2)", interaction="like")
                for u in repost_users:
                    G.add_edge(u, original, color="rgba(255, 0, 0, 0.2)", interaction="repost")
                for u in reply_users:
                    G.add_edge(u, original, color="rgba(0, 128, 0, 0.2)", interaction="reply")

                seen_users.update(like_users + repost_users + reply_users)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            post_futs = {pool.submit(get_user_posts, u): u for u in random.sample(seen_users, min(users_per_iteration, len(seen_users)))}
            for fut in as_completed(post_futs):
                user_posts = fut.result()
                all_posts.extend(user_posts)
                next_post_uris.extend(p["uri"] for p in user_posts)

        if not next_post_uris:
            st.warning("📉 次の投稿が見つかりませんでした。終了します。")
            break

        current_post_uris = next_post_uris

    if G.number_of_edges() == 0:
        st.warning("⚠️ ネットワークが構築されませんでした。")
        return

    
    attached_count = defaultdict(int)
    leaf_nodes = []
    for node in G.nodes():
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        if len(neighbors) == 1:
            parent = list(neighbors)[0]
            attached_count[parent] += 1
            leaf_nodes.append(node)
    
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)
        G.nodes[node]["size"] = attached_count.get(node, 1) * 5
        G.nodes[node]["color"] = "#cccccc"
    
    for leaf in leaf_nodes:
        G.nodes[leaf]["size"] = 5
        G.nodes[leaf]["color"] = "rgba(200,200,200,0.5)"  # 薄く表示
        G.nodes[leaf]["label"] = ""  # ラベルなしで主張を減らす

        for u, v, k in G.in_edges(leaf, keys=True):
            G[u][v][k]["color"] = "rgba(200,200,200,0.5)"
        for u, v, k in G.out_edges(leaf, keys=True):
            G[u][v][k]["color"] = "rgba(200,200,200,0.5)"

    st.success(f"✅ 完成ネットワーク：{G.number_of_nodes()} ユーザ, {G.number_of_edges()} エッジ")

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    for node, data in G.nodes(data=True):
        label = data.get("label", node)
        net.add_node(
            node,
            label=label,
            color=data.get("color", "#cccccc"),
            size=data.get("size", 10)
        )

    # 🔗 エッジも手動で追加
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, color=data.get("color", "gray"), title=data.get("interaction", ""))
    # legend = """
    #   <div style="position:absolute; bottom:10px; left:10px; background:white; padding:5px;
    #         border:1px solid #888; z-index:999;">
    #     <div><span style='color:red;'>■</span> Repost</div>
    #     <div><span style='color:blue;'>■</span> Like</div>
    #     <div><span style='color:green;'>■</span> Reply</div>
    #   </div>
    # """
    # net.html = net.html.replace("</body>", legend + "</body>")
    # os.makedirs("output", exist_ok=True)
    # path = "output/interaction_network_parallel.html"
    # net.save_graph(path)
    # with open(path, "r", encoding="utf-8") as f:
    #     st.components.v1.html(f.read(), height=750, scrolling=True)

    df = pd.DataFrame(all_posts)
    return G, df

######################################################
# 二つのネットワークを引数として、同じユーザで結合する関数
######################################################

def merge_graphs_on_common_nodes(G_a, G_b):
    # グラフAとBのノード集合
    nodes_a = set(G_a.nodes)
    nodes_b = set(G_b.nodes)

    # 共通ノード
    common_nodes = nodes_a & nodes_b
    st.write(f"🔗 共通ノード数: {len(common_nodes)}")

    # 統合グラフ（MultiDiGraph）
    G_merged = nx.MultiDiGraph()
    G_merged.add_nodes_from(G_a.nodes(data=True))
    G_merged.add_edges_from(G_a.edges(data=True))
    G_merged.add_nodes_from(G_b.nodes(data=True))
    G_merged.add_edges_from(G_b.edges(data=True))

    # 各共通ノードで明示的に "bridge" 接続（エッジは追加のみ、ノードはすでに追加済み）
    for did in common_nodes:
        G_merged.add_edge(did, did, interaction="bridge", color="purple")

    st.success(f"✅ 合体後ネットワーク：{G_merged.number_of_nodes()} ノード, {G_merged.number_of_edges()} エッジ")

    # 可視化（凡例含む）
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(G_merged)

    legend = """
      <div style="position:absolute; bottom:10px; left:10px; background:white; padding:5px;
            border:1px solid #888; z-index:999;">
        <div><span style='color:red;'>■</span> Repost</div>
        <div><span style='color:blue;'>■</span> Like</div>
        <div><span style='color:green;'>■</span> Reply</div>
        <div><span style='color:purple;'>■</span> 共通ノード（ブリッジ）</div>
      </div>
    """
    html_path = "output/merged_graph.html"
    os.makedirs("output", exist_ok=True)
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read().replace("</body>", legend + "</body>")
        st.components.v1.html(html_content, height=750, scrolling=True)

    return G_merged

######################################################
# 複数のネットワークのリストを引数として、同じユーザで結合する関数
######################################################
def merge_graphs_on_common_nodes_multi(graph_list):
    """
    複数の NetworkX グラフを統合し、共通ノードにブリッジを接続する。
    
    Parameters:
        graph_list (list of nx.DiGraph or nx.MultiDiGraph): 統合対象のグラフのリスト
    
    Returns:
        nx.MultiDiGraph: 統合されたグラフ
    """

    if len(graph_list) < 2:
        raise ValueError("最低でも2つのグラフが必要です。")

    # 全ノードの出現記録
    node_occurrences = {}
    for idx, G in enumerate(graph_list):
        for node in G.nodes:
            node_occurrences.setdefault(node, set()).add(idx)

    # 共通ノード（2つ以上のグラフに登場するノード）
    common_nodes = {node for node, idx_set in node_occurrences.items() if len(idx_set) > 1}
    st.write(f"🔗 共通ノード数: {len(common_nodes)}")

    # 空の MultiDiGraph に統合
    G_merged = nx.MultiDiGraph()
    for G in graph_list:
        G_merged.add_nodes_from(G.nodes(data=True))
        G_merged.add_edges_from(G.edges(data=True))

    # 共通ノードに "bridge" エッジを追加（自己ループ）
    for node in common_nodes:
        G_merged.add_edge(node, node, interaction="bridge", color="purple")

    st.success(f"✅ 合体後ネットワーク：{G_merged.number_of_nodes()} ノード, {G_merged.number_of_edges()} エッジ")

    # 可視化（pyvis + 凡例）
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(G_merged)

    # legend = """
    #   <div style="position:absolute; bottom:10px; left:10px; background:white; padding:5px;
    #         border:1px solid #888; z-index:999;">
    #     <div><span style='color:red;'>■</span> Repost</div>
    #     <div><span style='color:blue;'>■</span> Like</div>
    #     <div><span style='color:green;'>■</span> Reply</div>
    #     <div><span style='color:purple;'>■</span> 共通ノード（ブリッジ）</div>
    #   </div>
    # """

    # html_path = "output/merged_graph.html"
    # os.makedirs("output", exist_ok=True)
    # net.save_graph(html_path)

    # with open(html_path, "r", encoding="utf-8") as f:
    #     html_content = f.read().replace("</body>", legend + "</body>")
    #     st.components.v1.html(html_content, height=750, scrolling=True)

    return G_merged

######################################################
# 二つのリポストネットワークを引数として、同じユーザで結合する関数
######################################################
def merge_directed_graphs_on_common_nodes(G_a: nx.DiGraph, G_b: nx.DiGraph) -> nx.DiGraph:
    # 新しいDiGraphを作成
    G_merged = nx.DiGraph()
    
    # ノードとエッジをマージ
    G_merged.add_nodes_from(G_a.nodes(data=True))
    G_merged.add_nodes_from(G_b.nodes(data=True))
    G_merged.add_edges_from(G_a.edges(data=True))
    G_merged.add_edges_from(G_b.edges(data=True))
    
    # 共通ノードで連結する
    common_nodes = set(G_a.nodes()).intersection(G_b.nodes())
    for node in common_nodes:
        # ノードを介して双方向エッジで結びます
        G_merged.add_edge(node, node, interaction="bridge", color="purple")
    
    return G_merged

######################################################
# トピック別でネットワークを可視化する関数
######################################################
def visualize_network_with_topics(G, df, topic_col="topic", prob_col="topic_prob"):

    st.subheader("🧠 トピック別ネットワーク可視化（ぶら下がり処理＋信頼度付き）")

    if "did" not in df.columns or topic_col not in df.columns or prob_col not in df.columns:
        st.error("データフレームに 'did' またはトピック列または信頼度列が含まれていません")
        return

    # 1. 各ユーザーの代表トピック＆信頼度を決定
    user_topics = (
        df.groupby("did")[topic_col]
        .agg(lambda x: Counter(x).most_common(1)[0][0])
        .to_dict()
    )
    user_probs = (
        df.groupby("did")[prob_col]
        .agg("mean")  # 複数レコードがある場合は平均値
        .to_dict()
    )

    # 2. トピックごとに色を割り当て
    topic_ids = sorted(set(user_topics.values()))
    color_map = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    topic_colors = {topic: color_map[i % len(color_map)] for i, topic in enumerate(topic_ids)}

    # 3. ぶら下がりノード（接続先が1つだけ）を見つけて透明化する準備
    attached_count = defaultdict(int)
    leaf_nodes = []
    for node in G.nodes():
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        if len(neighbors) == 1:
            parent = list(neighbors)[0]
            attached_count[parent] += 1
            leaf_nodes.append(node)

    # 4. ノード属性の設定（色・サイズ・透明化処理＋信頼度）
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)
        prob = user_probs.get(node, 0.1)
        if node in leaf_nodes:
            G.nodes[node]["size"] = 5
            G.nodes[node]["color"] = "rgba(200,200,200,0.1)"
            G.nodes[node]["label"] = ""
        else:
            topic = user_topics.get(node, None)
            G.nodes[node]["topic"] = topic if topic is not None else "unknown"
            G.nodes[node]["topic_prob"] = prob
            G.nodes[node]["color"] = topic_colors.get(topic, "#cccccc")
            G.nodes[node]["size"] = max(prob * 40, 10)  # 信頼度に応じてスケーリング
            G.nodes[node]["label"] = f"{node}\nトピック: {topic}, 信頼度: {prob:.2f}"

    # 5. エッジの色変更（ぶら下がりノードを含むものを薄く）
    for u, v, k in G.edges(keys=True):
        if u in leaf_nodes or v in leaf_nodes:
            G[u][v][k]["color"] = "rgba(200,200,200,0.1)"

    # 6. 可視化（pyvis）
    net = Network(height="750px", width="100%", directed=True, notebook=False)

    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=data.get("label", node),
            color=data.get("color", "#cccccc"),
            size=data.get("size", 10),
            title=f"信頼度: {data.get('topic_prob', 0):.2f}"
        )

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, color=data.get("color", "rgba(128,128,128,0.8)"), title=data.get("interaction", ""))

    # 7. 凡例（トピック + 薄ノード説明付き）
    legend_items = "".join([
        f"<div><span style='color:{color}; font-weight:bold;'>■</span> Topic {tid}</div>"
        for tid, color in topic_colors.items()
    ])
    legend_html = f"""
    <div style="position:absolute; bottom:10px; left:10px; background:white;
                padding:10px; border:1px solid #888; z-index:999;">
        <b>📘 トピック凡例</b>
        {legend_items}
        <div><span style='color:rgba(200,200,200,0.5);'>■</span> ぶら下がりノード</div>
    </div>
    """
    net.html = net.html.replace("</body>", legend_html + "</body>")

    # 8. 保存・表示
    os.makedirs("output", exist_ok=True)
    path = "output/topic_colored_network.html"
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=750, scrolling=True)

######################################################
# ネガティブな感情分析が伝搬するようにした関数
######################################################
def normalize_text(text: str) -> str:
    """
    日本語SNS向けの軽量正規化処理：
    - 絵文字を除去
    - 全角記号の統一
    - 重複文字の縮小（例：すごーーーい→すごい）
    - 半角カナ・記号の正規化（オプション）

    Returns:
        str: 正規化済みテキスト
    """
    # 1. 絵文字除去
    text = emoji.replace_emoji(text, replace='')

    # 2. 全角記号→半角へ（例：？ → ?）
    text = re.sub(r'[？]', '?', text)
    text = re.sub(r'[！]', '!', text)

    # 3. 連続する記号や文字の簡略化（例：わーーーい → わい）
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # 4. 余分な空白を除去
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 感情分析パイプライン（英語）
# sentiment_pipeline = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
sentiment_pipeline = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
# tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
# def is_negative(text, threshold=0.2):
#     result = sentiment_pipeline(text)[0]
#     return result["label"].lower() == "negative" and result["score"] > threshold

def is_negative(text, threshold=0.3):
    try:
        text = normalize_text(text)
        text = text[:1000]
        result = sentiment_pipeline(text)[0]

        label = result["label"].lower()
        score = result["score"]

        # tabularisaiモデルの5段階評価に対応
        return label in ["very negative", "negative"] and score >= threshold

    except Exception as e:
        print(f"[⚠️ 感情分析エラー]: {e}")
        return False


def generate_neg_inter_matrix(network: nx.MultiDiGraph, post_df, topic_attr="topic", text_col="text"):
    inter_matrix = defaultdict(lambda: defaultdict(int))
    neg_inter_matrix = defaultdict(lambda: defaultdict(int))

    # 投稿者ごとのテキストリストをまとめる
    user_texts = post_df.groupby("did")[text_col].apply(list).to_dict()
    user_topics = post_df.groupby("did")["topic"].first().to_dict()  # 投稿単位ではなくユーザー単位

    for u, v, edge_data in network.edges(data=True):
        # トピック番号取得
        topic_u = network.nodes[u].get(topic_attr)
        topic_v = network.nodes[v].get(topic_attr)

        if topic_u is None or topic_v is None:
            continue

        inter_matrix[topic_u][topic_v] += 1

        # uの投稿に否定的なものがあるか判定
        texts = user_texts.get(u, [])
        if any(is_negative(text) for text in texts):
            neg_inter_matrix[topic_u][topic_v] += 1

    return inter_matrix, neg_inter_matrix

######################################################
# SPINを計算する関数
######################################################
def calculate_spin_score(inter_matrix, neg_inter_matrix, alpha=0.5, beta=0.5):
    """
    高速かつ型安全な SPIN スコア計算関数。
    """

    # 1. キーをすべて文字列に変換し統一
    all_keys = set(map(str, inter_matrix.keys())) | set(map(str, neg_inter_matrix.keys()))
    topics = sorted(all_keys)
    n = len(topics)
    topic_index = {topic: i for i, topic in enumerate(topics)}

    # 2. 空の NumPy 行列（交流数・ネガ数）
    inter_arr = np.zeros((n, n), dtype=np.float64)
    neg_arr = np.zeros((n, n), dtype=np.float64)

    # 3. 辞書から行列に変換（キーを str に変換）
    for c1, row in inter_matrix.items():
        for c2, value in row.items():
            i, j = topic_index[str(c1)], topic_index[str(c2)]
            inter_arr[i, j] = value

    for c1, row in neg_inter_matrix.items():
        for c2, value in row.items():
            i, j = topic_index[str(c1)], topic_index[str(c2)]
            neg_arr[i, j] = value

    # 4. マスク処理
    intra_mask = np.eye(n, dtype=bool)
    inter_mask = ~intra_mask

    # 5. 集計処理
    intra_total = inter_arr[intra_mask].sum()
    inter_total = inter_arr[inter_mask].sum()
    intra_neg_total = neg_arr[intra_mask].sum()
    inter_neg_total = neg_arr[inter_mask].sum()

    # 6. 比率・スコア
    intra_neg_ratio = intra_neg_total / intra_total if intra_total > 0 else 0
    inter_neg_ratio = inter_neg_total / inter_total if inter_total > 0 else 0
    spin_score = alpha * intra_neg_ratio + beta * inter_neg_ratio

    return {
        "SPIN": spin_score,
        "intra_neg_ratio": intra_neg_ratio,
        "inter_neg_ratio": inter_neg_ratio,
        "intra_total": intra_total,
        "inter_total": inter_total,
        "intra_neg_total": intra_neg_total,
        "inter_neg_total": inter_neg_total
    }

def calculate_spin_score_by_label(inter_matrix, neg_inter_matrix, alpha=0.25, beta=0.75):
    """
    ラベルベースのSPINスコア計算関数。

    Parameters:
        inter_matrix (dict): {label1: {label2: interaction_count, ...}, ...}
        neg_inter_matrix (dict): 同上（ネガティブインタラクション数）
        alpha (float): 内部ネガ比の重み
        beta (float): 外部ネガ比の重み

    Returns:
        dict: 各種SPINメトリクス
    """

    # 1. ラベルのセットを統一
    all_labels = set(map(str, inter_matrix.keys())) | set(map(str, neg_inter_matrix.keys()))
    for row in inter_matrix.values():
        all_labels.update(map(str, row.keys()))
    for row in neg_inter_matrix.values():
        all_labels.update(map(str, row.keys()))
    labels = sorted(all_labels)
    n = len(labels)
    label_index = {label: i for i, label in enumerate(labels)}

    # 2. 行列の初期化
    inter_arr = np.zeros((n, n), dtype=np.float64)
    neg_arr = np.zeros((n, n), dtype=np.float64)

    # 3. 辞書 → 行列
    for l1, row in inter_matrix.items():
        l1_str = str(l1)
        if l1_str not in label_index:
            continue
        for l2, value in row.items():
            l2_str = str(l2)
            if l2_str not in label_index:
                continue
            i, j = label_index[l1_str], label_index[l2_str]
            inter_arr[i, j] = value

    for l1, row in neg_inter_matrix.items():
        l1_str = str(l1)
        if l1_str not in label_index:
            continue
        for l2, value in row.items():
            l2_str = str(l2)
            if l2_str not in label_index:
                continue
            i, j = label_index[l1_str], label_index[l2_str]
            neg_arr[i, j] = value

    # 4. 内部 vs 外部マスク（同じラベルなら内部）
    intra_mask = np.eye(n, dtype=bool)
    inter_mask = ~intra_mask

    # 5. 集計
    intra_total = inter_arr[intra_mask].sum()
    inter_total = inter_arr[inter_mask].sum()
    intra_neg_total = neg_arr[intra_mask].sum()
    inter_neg_total = neg_arr[inter_mask].sum()

    # 6. スコア計算
    intra_neg_ratio = intra_neg_total / intra_total if intra_total > 0 else 0
    inter_neg_ratio = inter_neg_total / inter_total if inter_total > 0 else 0
    spin_score = alpha * intra_neg_ratio + beta * inter_neg_ratio

    return {
        "SPIN": spin_score,
        "intra_neg_ratio": intra_neg_ratio,
        "inter_neg_ratio": inter_neg_ratio,
        "intra_total": intra_total,
        "inter_total": inter_total,
        "intra_neg_total": intra_neg_total,
        "inter_neg_total": inter_neg_total,
        "labels": labels
    }

def display_spin_result(result, neg_inter_matrix):
    st.header("🧠 SPINスコア結果")

    # 数値出力
    st.metric("SPINスコア", f"{result['SPIN']:.3f}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("内部ネガ率", f"{result['intra_neg_ratio']:.3f}")
    with col2:
        st.metric("外部ネガ率", f"{result['inter_neg_ratio']:.3f}")

    with st.expander("📋 詳細データ"):
        st.write({
            "内部交流数": result["intra_total"],
            "内部ネガ交流数": result["intra_neg_total"],
            "外部交流数": result["inter_total"],
            "外部ネガ交流数": result["inter_neg_total"],
        })

    # ネガティブ交流行列をDataFrameに変換
    df_neg = pd.DataFrame(neg_inter_matrix).fillna(0).astype(int)

    # 🔢 ラベルを昇順に揃える（行と列両方）
    all_labels = sorted(set(df_neg.index) | set(df_neg.columns))
    df_neg_sorted = df_neg.reindex(index=all_labels, columns=all_labels, fill_value=0)

    # 表示
    st.subheader("📊 ネガティブ交流行列（ラベル昇順）")
    st.dataframe(df_neg_sorted)

    # seabornヒートマップ
    if df_neg_sorted.shape[0] > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_neg_sorted, annot=True, fmt="d", cmap="Reds", ax=ax)
        st.pyplot(fig)
    else:
        st.info("十分なラベル数がないため、ヒートマップは表示されません。")

######################################################
# ラベル伝搬関数・所属するコミュニティを特定する
######################################################

def label_propagation_community_detection(
    G: nx.MultiDiGraph,
    max_iter=10,
    num_classes=None,
    tau=0.5,
    create_others=True,
    label_attr="label"
):
    """
    アルゴリズム3に基づくラベル伝搬によるコミュニティ検出（SPIN用）

    Parameters:
        G: nx.MultiDiGraph - 入力ネットワーク（初期ラベル付きノードあり）
        max_iter: int - 反復回数
        num_classes: int or None - クラス数（None の場合、初期ラベルから自動決定）
        tau: float - コミュニティ所属しきい値（e.g., 0.5）
        create_others: bool - 所属が不明確なノードに "その他" ラベルを付与する
        label_attr: str - 初期ラベルに使用するノード属性名

    Returns:
        dict: ノード → コミュニティラベル の辞書
    """
    # 初期ラベル付きノードの取得
    labeled_nodes = {
        n: G.nodes[n][label_attr] for n in G.nodes if label_attr in G.nodes[n]
    }

    if not labeled_nodes:
        raise ValueError("初期ラベル付きノードが存在しません。")

    unique_labels = sorted(set(labeled_nodes.values()))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    num_classes = num_classes or len(unique_labels)

    # 初期ラベルベクトルの準備
    final_labels = {}
    for node in G.nodes:
        if node in labeled_nodes:
            vec = np.zeros(num_classes)
            vec[label_to_index[labeled_nodes[node]]] = 1.0
            final_labels[node] = vec
        else:
            final_labels[node] = np.ones(num_classes) / num_classes  # 一様分布

    for _ in range(max_iter):
        nodes = list(G.nodes)
        random.shuffle(nodes)
        for node in nodes:
            if node in labeled_nodes:
                continue  # 固定されたラベルは変えない

            new_label = np.zeros(num_classes)
            weight_sum = 0.0

            for neighbor, _, data in G.in_edges(node, data=True):
                weight = data.get("weight", 1.0)
                new_label += weight * final_labels[neighbor]
                weight_sum += weight

            for _, neighbor, data in G.out_edges(node, data=True):
                weight = data.get("weight", 1.0)
                new_label += weight * final_labels[neighbor]
                weight_sum += weight

            if weight_sum > 0:
                final_labels[node] = new_label / weight_sum

    # 最終ラベルの決定（しきい値ベース）
    node_communities = {}
    for node, probs in final_labels.items():
        partitions = [
            i for i, p in enumerate(probs) if p >= tau
        ]
        if create_others and not partitions:
            node_communities[node] = [num_classes]  # "others"
        else:
            node_communities[node] = partitions

    return node_communities
    
######################################################
# ラベル伝搬関数・所属するトピックを特定する
######################################################
def topic_label_propagation(
    G: nx.MultiDiGraph,
    topic_df: pd.DataFrame,
    topic_col="topic",
    prob_col="topic_prob",
    max_iter=10,
    tau=0.5,
    create_others=True
):
    """
    トピックとその確率に基づいてラベルを伝搬する関数。

    Parameters:
        G (nx.MultiDiGraph): 入力ネットワーク（ノードIDは'did'に対応）
        topic_df (pd.DataFrame): 各ノードのトピックとその確率を含むデータフレーム
        topic_col (str): トピックラベルが格納されている列名
        prob_col (str): トピックの確率が格納されている列名
        max_iter (int): ラベル伝搬の最大反復回数
        tau (float): トピックに所属とみなす確率しきい値
        create_others (bool): 所属なしノードに"others"ラベルを付与するか

    Returns:
        dict: ノードID → トピックインデックスリスト の辞書
    """
    # 初期トピック付きノードを取得
    labeled_nodes = topic_df.dropna(subset=["did", topic_col, prob_col])
    labeled_nodes = labeled_nodes.groupby("did").agg({topic_col: "first", prob_col: "mean"}).reset_index()
    labeled_dict = labeled_nodes.set_index("did").to_dict(orient="index")

    # トピック集合とインデックス化
    unique_topics = sorted(set(v[topic_col] for v in labeled_dict.values()))
    topic_to_index = {label: i for i, label in enumerate(unique_topics)}
    num_classes = len(unique_topics)

    # 初期ラベルベクトルの準備
    final_labels = {}
    for node in G.nodes:
        if node in labeled_dict:
            vec = np.zeros(num_classes)
            topic_index = topic_to_index[labeled_dict[node][topic_col]]
            prob = labeled_dict[node][prob_col]
            vec[topic_index] = prob
            final_labels[node] = vec
        else:
            final_labels[node] = np.ones(num_classes) / num_classes

    # ラベル伝搬の反復
    for _ in range(max_iter):
        nodes = list(G.nodes)
        random.shuffle(nodes)
        for node in nodes:
            if node in labeled_dict:
                continue  # 初期ラベルは固定

            new_label = np.zeros(num_classes)
            weight_sum = 0.0

            for neighbor, _, data in G.in_edges(node, data=True):
                weight = data.get("weight", 1.0)
                new_label += weight * final_labels[neighbor]
                weight_sum += weight

            for _, neighbor, data in G.out_edges(node, data=True):
                weight = data.get("weight", 1.0)
                new_label += weight * final_labels[neighbor]
                weight_sum += weight

            if weight_sum > 0:
                final_labels[node] = new_label / weight_sum

    # 最終ラベルの決定
    node_topics = {}
    for node, probs in final_labels.items():
        partitions = [i for i, p in enumerate(probs) if p >= tau]
        if create_others and not partitions:
            node_topics[node] = [num_classes]  # "others" クラス
        else:
            node_topics[node] = partitions

    return node_topics

######################################################
# ラベル伝搬関数・所属するコミュニティを可視化する
######################################################

def visualize_community_network_streamlit(G, node_communities, label_palette="tab10", notebook=False):
    """
    Streamlit上でコミュニティごとにノードを色分けしたネットワークを可視化（pyvis）

    Parameters:
        G: NetworkX グラフ
        node_communities: dict - {ノード: [コミュニティID]}
        label_palette: str - seabornカラーパレット名
        notebook: bool - pyvis notebook モード（通常FalseでOK）
    """

    # 色マップの準備
    unique_coms = sorted({c for coms in node_communities.values() for c in coms})
    palette = sns.color_palette(label_palette, len(unique_coms)).as_hex()
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_coms)}
    color_map["others"] = "#bbbbbb"

    # Pyvisネットワーク初期化
    net = Network(height="750px", width="100%", directed=True, notebook=notebook)

    # ノード追加
    for node, data in G.nodes(data=True):
        coms = node_communities.get(node, ["others"])
        main_com = coms[0] if coms else "others"
        color = color_map.get(main_com, "#bbbbbb")

        label = data.get("label", str(node))
        size = data.get("size", 10)

        net.add_node(node, label=label, color=color, size=size)

    # エッジ追加
    for u, v, edge_data in G.edges(data=True):
        title = edge_data.get("interaction", "")
        color = edge_data.get("color", "#cccccc")
        net.add_edge(u, v, color=color, title=title)

    # HTML 出力と埋め込み
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp_file.name)  # 一時ファイル削除

    st.components.v1.html(html, height=750, scrolling=True)

######################################################
# ネガティブ交流行列を、グラフから作成する。
######################################################

def generate_interaction_matrices_from_graph(G, node_communities, post_df, text_col="text"):
    """
    ラベル伝搬後のノードに基づいて、内部・外部の交流数とネガティブ交流数の行列を生成する（高速版）

    Parameters:
        G: nx.MultiDiGraph - 入力グラフ（ラベル伝搬後）
        node_communities: dict - ノード → [コミュニティID] のマッピング
        post_df: pd.DataFrame - 各ユーザの投稿テキストが含まれるデータフレーム
        text_col: str - テキストカラム名

    Returns:
        inter_matrix: defaultdict - 通常の交流数行列
        neg_inter_matrix: defaultdict - ネガティブ交流数行列
    """
    inter_matrix = defaultdict(lambda: defaultdict(int))
    neg_inter_matrix = defaultdict(lambda: defaultdict(int))

    # ユーザーごとの投稿リスト
    user_texts = post_df.groupby("did")[text_col].apply(list).to_dict()

    # 事前にネガティブ投稿を持つユーザーをキャッシュ
    negative_users = set()
    for did, texts in user_texts.items():
        try:
            if any(is_negative(text) for text in texts):
                negative_users.add(did)
        except Exception as e:
            continue  # 何か問題があれば無視

    # エッジごとの処理（交流数とネガティブ交流数）
    for u, v, data in G.edges(data=True):
        com_u = node_communities.get(u, ["others"])[0]
        com_v = node_communities.get(v, ["others"])[0]

        inter_matrix[com_u][com_v] += 1

        if u in negative_users:
            neg_inter_matrix[com_u][com_v] += 1

    return inter_matrix, neg_inter_matrix

