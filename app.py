import sys
import streamlit as st
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

import analyze

# このスクリプトと同じディレクトリの icons/logo.png をアイコンにしたい例
icon_path = os.path.join(os.path.dirname(__file__), "icons", "logo.png")

st.set_page_config(
    page_title="Blue Sky Analysis App",
    page_icon=icon_path
)

# ログイン処理
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)


# ログインメソッドで入力フォームを配置
authenticator.login(location='main')

if st.session_state.get('authentication_status'):
    authenticator.logout()
    st.write(f'Welcome *{st.session_state.get("name")}*')
    analyze.analyze(config)
elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')

