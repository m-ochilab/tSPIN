# 卒業研究：Blueskyにおける情報拡散ネットワークと負の情報流（tSPIN）解析システム
本リポジトリは、分散型SNS「Bluesky」を対象に、特定のトピック（クエリ）に関連する投稿を収集し、情報の拡散構造およびユーザー間の負の影響度（NNIF/tSPIN）を定量化・可視化するための解析システムです。
1. システム概要
本システムは、Elasticsearchに蓄積されたBlueskyのデータを用いて、以下のステップで解析を行います。
データ抽出: 指定したキーワードと期間に基づき、投稿データを取得。
SPIN (Snowball Sampling): アルゴリズムに基づき、シードユーザーから拡散ネットワークをサンプリング。
コミュニティ・トピック抽出: ラベル伝搬によるコミュニティ分割と、BERTopicによる話題抽出。
NNIF (Non-negative Information Flow): ユーザー間の投稿の時系列・トークン類似度から情報の流れを計算。
tSPIN 解析: 特定のトピックやネガティブな文脈が、コミュニティ内外にどのように波及しているかをスコアリング。
2. ディレクトリ構成と主要ファイル主要なスクリプトとその役割は以下の通りです。
ファイル名役割
run_spin_cli.py: メイン実行スクリプト。コンソールから一連の解析フローを回します。research_matae_tokenizer_updated.py: 共通ユーティリティ。データ取得、ネットワーク構築、可視化関数群。
snowball_spin.py: SPINアルゴリズム（Algorithm 1）の本体実装。
define_seed_snowball.py: シードユーザーの選定およびSPINの実行管理。topic_pipeline_tokenizer_updated.py: BERTopicを用いたトピック抽出およびユーザープロファイリング。
Calculate_NNIF_tokenizer_updated.py: ユーザー間の情報流（NNIF）の計算。時間同期クロスエントロピーを使用。
Calculate_tSPIN.py: コミュニティとトピックを組み合わせたtSPIN指数の算出。
negative_judge.py: 日本語辞書等を用いたネガティブ投稿の判定。
3. セットアップ必要要件
Python 3.9以上、Elasticsearch（解析対象データが格納されていること）
pip install -r requirements-core.txt requirements-topic.txt

4. 実行方法コンソールからの実行
run_spin_cli.py 内の query_str_a, query_str_b および start_day, end_day を書き換えて実行します。
python run_spin_cli.py
実行が完了すると、解析結果のサマリが標準出力に表示され、詳細なデータが export_all_results 関数を通じて出力されます。Streamlitによる可視化research_matae_tokenizer_updated.py に含まれる関数群は、Streamlit上でのインタラクティブな描画（ネットワークグラフや時系列比較）に対応しています。
5. 後任者への引き継ぎ事項独自アルゴリズム：SPINとNNIF
SPIN: ネットワーク全体を追うのではなく、拡散の「一枝」をランダムに辿ることで、巨大なSNSデータから効率的に拡散構造を抽出しています。
NNIF: ユーザーAの投稿後にユーザーBが似た内容を投稿するまでの「時間」と「内容の類似性（圧縮効率）」から、情報の伝播を推定しています。注意点Elasticsearch接続: ES_ADDR は研究室の環境に合わせて適宜修正してください。メモリ消費: BERTopicやNNIFの計算はメモリを大量に消費します。特に relevant_users の数が増える場合は、max_tokens_per_user などの制限パラメータを調整してください。
トークナイザ: 日本語の分かち書き精度が解析結果に直結します。環境に合わせて Sudachi や GiNZA が正しくロードされているか確認してください。
