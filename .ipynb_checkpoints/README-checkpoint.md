# tSPIN Project: OSNEM Submission & System Handover

## 1. プロジェクト概要 (Project Overview)
本プロジェクトは、SNS上における政治的対立（分極化）を、単なる「構造的な距離」ではなく**「トピックを介した有向の情報流（Directed Information Flow）」**として定量化する新指標 **tSPIN (topic-aware Social Polarization Information Network)** の開発と実装システムです。

現在、本研究は国際誌 **OSNEM (Online Social Networks and Media)** の特集号 *"Disinformation, Toxicity, Harms in Online Social Networks and Media"* への投稿に向けて最適化されています。従来の分極化指標では捉えきれなかった「有害な言説（Toxic discourse）が、どのコミュニティからどのトピックに乗って攻撃されているか」を高解像度で可視化します。

### CRediT authorship contribution statement
* **Masanao Ochi:** Supervision, Project administration, Writing - review & editing. 
* **Yusei Matae:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization.

---

## 2. 実行手順 (Execution Guide)
数百万件規模のデータ処理を行うため、処理中にSSH接続等が切断されても計算が継続できるよう、仮想端末（`byobu`）上で実行します。

### Step-by-Step 実行コマンド
```bash
# 1. プロジェクトディレクトリに移動
cd bsky_es_analysis_app

# 2. 仮想端末（byobu）を起動（接続切れ対策）
byobu new -t (仮想環境名)

# 3. 環境変数の再読み込み（byobu内でcondaコマンドを有効にするため）
source ~/.bashrc

# 4. 分析用の仮想環境（spin）をアクティベート
conda activate spin

# 5. メインパイプラインの実行
python run_spin_cli_en.py
(※ byobu からデタッチ（バックグラウンドに回して抜ける）する場合は F6 キー、または Ctrl + a の後に d を押してください。)

3. システム構成と主要スクリプト (Codebase Architecture)
本システムは、数百万件のテキストデータとネットワークデータを現実的な時間で処理するため、高度な並列処理とメモリ最適化が施されています。

📊 データ抽出・前処理
research_matae_tokenizer_updated.py
Elasticsearchからのデータ抽出、Bluesky API連携、および高度なトークナイズ（SpaCy等）を担当。ノイズ除去やネガティブユーザーの判定基盤を含みます。

🧠 トピックモデル (BERTopic)
topic_pipeline_tokenizer_updated_en.py
抽出したテキスト群をBERTopicで分類します。

【最適化】ランダムサンプリング学習: メモリ枯渇を防ぐため、15万件のデータをランダム抽出して空間モデルを学習（Fit）し、その空間に全データを当てはめる（Transform）構造を採用しています。

【最適化】デッドロック対策: PyTorch(GPU)とUMAPのマルチスレッド衝突を防ぐ環境変数設定が組み込まれています。

🌊 情報流計算 (NNIF)
Calculate_NNIF_tokenizer_updated_en.py
South et al. の情報流エントロピー（NNIF）を計算します。

【最適化】メモリ爆発(Pickle Bomb)の防止: ProcessPoolExecutor の initializer を用い、巨大なトークン辞書をワーカープロセス間で「メモリ共有」することで、CPUの全コアを使った安全な爆速並列計算を実現しています。

🧮 tSPINスコア算出と可視化
Calculate_tSPIN_en.py
NNIFとBERTopicの結果を統合し、内部循環と外部攻撃のテンソル演算をPyTorch(torch.einsum)を用いてGPUで高速実行します。

research_output_en.py / Topic_heatmap.py
結果の出力と可視化。20×20のフル・ブロックマトリックスを作成し、Spectral Co-clusteringを用いて陣営（Ukr/Rus）の境界を維持したまま類似トピックを並べ替える階層的ヒートマップ（Outside-In Layout）を自動生成します。

4. 出力ファイル (Output Artifacts)
run_spin_cli_en.py が正常に完了すると、タイムスタンプ付きのディレクトリ（例: results/YYYYMMDD_HHMMSS/）に以下のファイル群が出力されます。

heatmap_coclustering_hierarchical_Ukr_Rus.pdf / .png:
論文のメインとなる20×20の共クラスタリング・ヒートマップ。

table_4_1_dataset_statistics.csv / .json:
ノード数、エッジ数、ネガティブユーザー数などのデータセット統計。

topic_details_summary.md / table_topic_details.csv:
抽出された各トピックの代表語（Keywords）と代表的な投稿テキスト（Ground Truthing用）。

table_4_2_asymmetry_analysis.csv:
双方向のスコア差分（非対称性）が大きいトピックペアのランキング。

5. 今後の展望・既知の課題 (Future Work)
データ疎性（Data Sparsity）への対応: 投稿数が極端に少ないトピックペアでスコアが不安定になる場合があるため、ベイズ推定等を用いた平滑化（Smoothing）の導入余地があります。

時系列分析（Longitudinal Analysis）: 現状の静的なスナップショットから、NNIFのタイムスタンプ情報を活かした「対立の波及プロセス」の動的モデル化への拡張。