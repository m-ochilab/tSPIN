import streamlit as st
import pandas as pd
from run_spin_cli import main as cli_main

def run_spin_streamlit():
    st.title("SPIN Snowball 分析ダッシュボード")

    # 1. CLI main を実行
    result = cli_main()   # ← 元の main() をそのまま利用

    # 2. 結果を Streamlit に表示
    st.subheader("Seed Accounts")
    st.write(result["seed_accounts"])

    st.subheader("サンプルされたユーザ数 / エッジ数")
    st.write(f"ユーザ数: {len(result['common_list'])}")
    st.write(f"エッジ数: {len(result['sampled_edges'])}")

    # エッジ DataFrame
    edges_df = pd.DataFrame(result["sampled_edges"], columns=["src", "dst"])
    st.subheader("Sampled Edges")
    st.dataframe(edges_df)

    # --- ここにネットワーク可視化や BERTopic の結果なども Streamlit 上へ ---
    # 例:
    # st.plotly_chart(network_fig)
    # st.dataframe(topic_df)

if __name__ == "__main__":
    run_spin_streamlit()
