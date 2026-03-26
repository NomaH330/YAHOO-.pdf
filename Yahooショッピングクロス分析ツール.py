import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import io
from io import BytesIO
import re
import numpy as np
from janome.tokenizer import Tokenizer
from itertools import combinations
from collections import Counter
import datetime
from sentence_transformers import SentenceTransformer
# --- グラフ日本語対応ライブラリ ---
import japanize_matplotlib
# --- ワードクラウド & 階層的クラスタリング関連ライブラリ ---
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as shc
# --- インタラクティブグラフ ライブラリ ---
import plotly.graph_objects as go
import plotly.express as px
# --- 画像処理ライブラリ (エラー修正用) ---
import PIL.Image

# 【画像エラー修正】Pillow 10.0.0以降で削除されたANTIALIASをLANCZOSに紐付けるパッチ
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- ネットワーク図関連 ---
import networkx as nx

# --- ページの初期設定 ---
st.set_page_config(
    page_title="Yahoo！ショッピングの購買記録の多機能分析ツール",
    page_icon="👑",
    layout="wide"
)

# --- セッションステートの初期化 ---
if 'result_df' not in st.session_state: st.session_state.result_df = None
if 'ranking_df' not in st.session_state: st.session_state.ranking_df = None
if 'selected_months' not in st.session_state: st.session_state.selected_months = []
if 'sim_base_col' not in st.session_state: st.session_state.sim_base_col = None

if 'distance_threshold' not in st.session_state: st.session_state.distance_threshold = 0.5

if 'vectors_for_clustering' not in st.session_state: st.session_state.vectors_for_clustering = None
if 'vector_source_col' not in st.session_state: st.session_state.vector_source_col = None
if 'news_data' not in st.session_state: st.session_state.news_data = {}
if 'highlight_terms' not in st.session_state: st.session_state.highlight_terms = [""] * 5
if 'highlight_cols' not in st.session_state: st.session_state.highlight_cols = [""] * 5
if 'show_wiki_news' not in st.session_state: st.session_state.show_wiki_news = False
if 'cluster_param_method' not in st.session_state: st.session_state.cluster_param_method = 'コサイン距離'
if 'n_clusters_input' not in st.session_state: st.session_state.n_clusters_input = 5

if 'comment_uploader_key' not in st.session_state: st.session_state.comment_uploader_key = 0
if 'exp_uploader_key' not in st.session_state: st.session_state.exp_uploader_key = 0
if 'news_uploader_key' not in st.session_state: st.session_state.news_uploader_key = 0

if 'active_distance_threshold' not in st.session_state: st.session_state.active_distance_threshold = 0.5
if 'active_n_clusters' not in st.session_state: st.session_state.active_n_clusters = 5
if 'active_filter_method' not in st.session_state: st.session_state.active_filter_method = 'コサイン距離'

if 'news_outputs' not in st.session_state: st.session_state.news_outputs = {}
if 'expenditure_output' not in st.session_state: st.session_state.expenditure_output = None
if 'specific_ranking_df' not in st.session_state: st.session_state.specific_ranking_df = None
if 'output2_ranking_output' not in st.session_state: st.session_state.output2_ranking_output = None

# 分析用データのキャッシュ（ボタン更新用）
if 'active_plot_df' not in st.session_state: st.session_state.active_plot_df = None

# 分析実行時の状態保存
if 'frozen_df_filtered' not in st.session_state: st.session_state.frozen_df_filtered = None
if 'frozen_sidebar_range' not in st.session_state: st.session_state.frozen_sidebar_range = (None, None)
if 'frozen_filter_mode' not in st.session_state: st.session_state.frozen_filter_mode = None

# --- コールバック関数 ---
def update_users_from_groups():
    selected_groups = st.session_state.target_group_multiselect
    if st.session_state.result_df is not None and not st.session_state.result_df.empty:
        temp_df = st.session_state.result_df.copy()
        if 'UserID' not in temp_df.columns and 'ソースファイル' in temp_df.columns:
            temp_df['UserID'] = temp_df['ソースファイル'].str[:3]
        if 'UserID' in temp_df.columns:
            if selected_groups:
                users = temp_df[temp_df['分類グループ'].isin(selected_groups)]['UserID'].unique().tolist()
                st.session_state.target_user_multiselect = sorted(users)
            else:
                st.session_state.target_user_multiselect = []

# --- 分析用関数 ---

def create_wordcloud(frequencies, font_path=None, max_font_scale_factor=1.0):
    """ワードクラウドを生成する関数（日本語フォント対応版）"""
    if not frequencies: return None
    max_font_size = min(200, int(100 * max_font_scale_factor))
    
    # フォントパスの動的取得
    try:
        font_path = japanize_matplotlib.get_font_ttf_path()
    except:
        font_path = None

    wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white', max_font_size=max_font_size, prefer_horizontal=1, colormap='viridis').generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(12, 6)); ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off')
    return fig

@st.cache_resource
def get_tokenizer(): return Tokenizer()

@st.cache_resource
def load_st_model(): return SentenceTransformer('all-MiniLM-L6-v2')

EMOTION_DICT = {'好き': ['好き', '最高', '素晴らしい', '愛用', 'お気に入り', 'リピート', '大満足', '素敵'], '喜び': ['嬉しい', '楽しい', '満足', '美味しい', 'うまい', '快適', '気持ちい', '癒される', '面白い'], '悲しみ': ['悲しい', '残念', 'がっかり', '寂しい', '切ない'], '怒り': ['ひどい', '最悪', '怒り', '不満', 'ありえない', '許せない', 'ムカつく'], '恐れ': ['不安', '怖い', '心配']}

def calculate_emotion_scores(text):
    if not isinstance(text, str): return {e: 1 for e in EMOTION_DICT.keys()}
    scores = {};
    for e, kws in EMOTION_DICT.items():
        c = sum(kw in text for kw in kws); score = min(c * 2, 5); scores[e] = score if score > 0 else 1
    return scores

@st.cache_data
def get_common_words_by_group(df, text_col, group_col, top_n=3):
    if text_col not in df.columns or group_col not in df.columns: return {}
    t = get_tokenizer(); common_words_map = {}; unique_groups = df[group_col].unique()
    for group_id in unique_groups:
        texts = df[df[group_col] == group_id][text_col].dropna().astype(str).tolist()
        if not texts: common_words_map[group_id] = "---"; continue
        all_words = []
        for text in texts:
            tokens = [tok.base_form for tok in t.tokenize(text) if tok.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞'] and len(tok.surface) > 1]; all_words.extend(tokens)
        if not all_words: common_words_map[group_id] = "---"
        else: count = Counter(all_words); top_n_words = [word for word, freq in count.most_common(top_n)]; common_words_map[group_id] = ', '.join(top_n_words)
    return common_words_map

@st.cache_data
def process_expenditure_files(files, sheet_name):
    all_data = []
    
    # シート名ごとの読み込み設定
    sheet_settings = {
        '二人': {'rows': 155 - 9, 'cols': "L,M,O:AF"}, 
        '勤労': {'rows': 246 - 9, 'cols': "L,M,O:AF"}, 
        '無職': {'rows': 246 - 9, 'cols': "L,M,O:Y"}   
    }
    
    settings = sheet_settings.get(sheet_name)
    if not settings:
        st.error(f"シート名 {sheet_name} の設定が見つかりません。")
        return pd.DataFrame()
        
    nrows_to_read = settings['rows']
    
    for file in files:
        try:
            try: 
                date_df = pd.read_excel(file, sheet_name=sheet_name, header=None, usecols="H", skiprows=6, nrows=1, engine='openpyxl')
                date_str = date_df.iloc[0, 0] if not date_df.empty else f"不明({file.name})"
            except ValueError: 
                continue
            
            try:
                # 【エラー対策】列指定をあえて行わず、ヘッダー位置のみ指定して読み込む
                df_exp_raw = pd.read_excel(file, sheet_name=sheet_name, header=8, engine='openpyxl')
            except Exception as e:
                st.warning(f"ファイル {file.name} のシート読み込み中にエラーが発生しました: {e}")
                continue

            if len(df_exp_raw.columns) < 15:
                st.warning(f"ファイル {file.name} の列数が不足しています。期待される形式ではありません。")
                continue

            # 動的に列を抽出 (L, M, O以降)
            target_col_indices = [11, 12] + list(range(14, len(df_exp_raw.columns)))
            df_exp = df_exp_raw.iloc[:, target_col_indices].copy()

            df_exp.rename(columns={df_exp.columns[0]: '用途分類', df_exp.columns[1]: '単位'}, inplace=True)
            df_exp.dropna(subset=['用途分類'], inplace=True)
            df_exp['報告年月'] = date_str
            
            all_data.append(df_exp)
                
        except Exception as e: 
            st.error(f"ファイル {file.name} の処理中に予期せぬエラー: {e}")
            return pd.DataFrame()
            
    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    cols = ['報告年月', '用途分類', '単位'] + [col for col in final_df.columns if col not in ['報告年月', '用途分類', '単位']]
    return final_df[cols]

@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='analysis_results')
    processed_data = output.getvalue(); return processed_data

def create_cooccurrence_network_graph(texts, tokenizer, top_n_words=10, k_for_ranking=3):
    word_counts = Counter(); all_words_list = []
    for doc in texts:
        if not isinstance(doc, str): continue
        doc = re.sub(r'[\[\]「」。]', '', doc); words = [token.surface for token in tokenizer.tokenize(doc) if token.part_of_speech.split(',')[0] in ['名詞', '形容詞'] and len(token.surface) > 1]
        if words: word_counts.update(words); all_words_list.append(words)
    co_occurrence_2 = Counter()
    for words in all_words_list:
        for w1, w2 in combinations(sorted(list(set(words))), 2): co_occurrence_2[(w1, w2)] += 1
    co_occurrence_k = Counter()
    if len(texts) > 0 and k_for_ranking > 1:
        for words in all_words_list:
            unique_words_in_doc = list(set(words))
            if len(unique_words_in_doc) >= k_for_ranking:
                for combo in combinations(sorted(unique_words_in_doc), k_for_ranking): co_occurrence_k[combo] += 1
    df_word_rank = pd.DataFrame(word_counts.most_common(), columns=['単語', '出現回数']); df_word_rank['順位'] = df_word_rank.index + 1; df_cooc_k = pd.DataFrame(co_occurrence_k.most_common(), columns=['combo', 'count']); fig = go.Figure()
    if not word_counts: fig.update_layout(title='共起ネットワーク (表示可能なデータがありません)', xaxis={"visible": False}, yaxis={"visible": False}); return fig, df_word_rank, df_cooc_k
    top_words_data = word_counts.most_common(top_n_words); network_nodes = [word for word, count in top_words_data]; top_edges = co_occurrence_2.most_common(200); G = nx.Graph()
    for node in network_nodes: G.add_node(node)
    for (w1, w2), weight in top_edges:
        if w1 in network_nodes and w2 in network_nodes: G.add_edge(w1, w2, weight=weight)
    G.remove_nodes_from(list(nx.isolates(G)))
    if not G.nodes(): fig.update_layout(title='共起ネットワーク (表示可能な繋がりがありません)', xaxis={"visible": False}, yaxis={"visible": False}); return fig, df_word_rank, df_cooc_k
    top_k_coocs = co_occurrence_k.most_common(3); cluster_top_coocs_str_list = [f"<b>{i+1}位：{count}回：</b>{'，'.join(combo)}" for i, (combo, count) in enumerate(top_k_coocs)]; cluster_top_coocs_str = "<br>".join(cluster_top_coocs_str_list) if cluster_top_coocs_str_list else "なし"
    pos = nx.spring_layout(G, k=0.8, seed=42); edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'); node_trace = go.Scatter(x=[], y=[], textfont=dict(size=10), mode='text', text=[], hoverinfo='text', hovertext=[], marker=dict(size=[], color='skyblue', line_width=2))
    for edge in G.edges(): x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_trace['x'] += tuple([x0, x1, None]); edge_trace['y'] += tuple([y0, y1, None])
    word_counts_dict = dict(word_counts); rank_map = {word: rank + 1 for rank, (word, count) in enumerate(top_words_data)}
    for node in G.nodes(): x, y = pos[node]; node_trace['x'] += tuple([x]); node_trace['y'] += tuple([y]); rank = rank_map.get(node, ''); count = word_counts_dict.get(node, 0); node_label = f"{rank}. {node}<br>({count}回)"; node_trace['text'] += tuple([node_label]); hover_text = f'<b>{node} ({count}回)</b><br>--- {k_for_ranking}単語の共起ランキング ---<br>{cluster_top_coocs_str}'; node_trace['hovertext'] += tuple([hover_text]); node_trace['marker']['size'] += tuple([0])
    layout = go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)); fig = go.Figure(data=[edge_trace, node_trace], layout=layout); return fig, df_word_rank, df_cooc_k

# --- リセット関数 ---
def reset_comment_uploader(): 
    st.session_state.comment_uploader_key += 1; st.session_state.result_df = None; st.session_state.ranking_df = None; 
    st.session_state.sim_base_col = None; st.session_state.vectors_for_clustering = None; st.session_state.output2_ranking_output = None; 
    st.session_state.active_plot_df = None; st.session_state.frozen_df_filtered = None; st.session_state.frozen_sidebar_range = (None, None); st.session_state.frozen_filter_mode = None
def reset_exp_uploader(): st.session_state.exp_uploader_key += 1; st.session_state.expenditure_output = None
def reset_news_uploader(): st.session_state.news_uploader_key += 1; st.session_state.news_data = {}; st.session_state.news_outputs = {}

# --- サイドバー ---
with st.sidebar:
    st.title("絞り込みと分析対象の設定")
    st.header("1. Yahoo！ショッピングの購買記録をアップロード")
    uploaded_files = st.file_uploader("購買記録データ", type=['csv', 'xlsx'], accept_multiple_files=True, key=f"comment_uploader_{st.session_state.comment_uploader_key}")
    if uploaded_files: st.button("アップロードを一斉削除", on_click=reset_comment_uploader, key="clear_comments", use_container_width=True)
    df = None
    if uploaded_files:
        try:
            df_list = []
            for f in uploaded_files: temp_df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f); temp_df['ソースファイル'] = f.name; df_list.append(temp_df)
            df = pd.concat(df_list, ignore_index=True); st.success(f"{len(uploaded_files)}個のファイルを読み込みました。")
        except Exception as e: st.error(f"ファイル読み込みエラー: {e}")
    else: st.info("分析を開始するには、購買記録データをアップロードしてください。")

if df is not None:
    if 'コメント' in df.columns:
        with st.spinner('コメントの感情を分析中です...'):
            emotion_scores_list = [calculate_emotion_scores(text) for text in df['コメント']]; emotion_df = pd.DataFrame(emotion_scores_list)
            for col in emotion_df.columns: df[col] = emotion_df[col]
    df_filtered = df.copy()
    with st.sidebar:
        st.header("2. データの絞り込み条件")
        with st.expander("期間で絞り込む", expanded=False):
            if '購買日' in df.columns:
                try:
                    date_column = '購買日'; df_filtered[date_column] = pd.to_datetime(df_filtered[date_column], errors='coerce'); df_filtered.dropna(subset=[date_column], inplace=True); filter_type = st.radio("指定方法:", ('期間を指定', '月を選択'), horizontal=True, key="date_filter_type")
                    if filter_type == '期間を指定':
                        # 【固定期間設定】
                        start_limit = datetime.date(2010, 1, 1)
                        end_limit = datetime.date(2035, 1, 31)

                        # データから抽出した最小/最大値（デフォルト値として利用）
                        min_date_val_default = df_filtered[date_column].min().date() if not df_filtered[date_column].empty else datetime.date.today()
                        max_date_val_default = df_filtered[date_column].max().date() if not df_filtered[date_column].empty else datetime.date.today()
                        
                        # デフォルト値を強制範囲内に収める
                        min_date_val = max(min_date_val_default, start_limit)
                        max_date_val = min(max_date_val_default, end_limit)
                        
                        if min_date_val > max_date_val:
                             min_date_val = start_limit
                             max_date_val = end_limit
                        
                        # 【指示1対応: 開始日を先に配置】
                        start_date = st.date_input(
                            '開始日', 
                            min_date_val, 
                            min_value=start_limit,
                            max_value=end_limit 
                        )
                        
                        end_date = st.date_input(
                            '終了日', 
                            max_date_val, 
                            min_value=start_limit, 
                            max_value=end_limit
                        )
                        
                        # 【指示2対応: 期間の順序チェックとフィルタリング】
                        if start_date and end_date:
                            if start_date > end_date:
                                # 開始日が終了日より後にある場合、データを空にする (0件表示を実現)
                                df_filtered = pd.DataFrame(columns=df_filtered.columns)
                                st.warning("期間指定が不正です: 開始日は終了日以前である必要があります。")
                            else:
                                # 期間が正しい場合のみフィルタリングを適用
                                df_filtered = df_filtered[ (df_filtered[date_column].dt.date >= start_date) & (df_filtered[date_column].dt.date <= end_date) ]
                        
                    else:
                        st.write("毎年、以下の月を対象にする:"); cols = st.columns(6); selected_months = []
                        for i in range(1, 13):
                            with cols[(i-1)%6]:
                                if st.checkbox(f"{i}月", key=f"month_{i}"): selected_months.append(i)
                        if selected_months: df_filtered = df_filtered[df_filtered[date_column].dt.month.isin(selected_months)]
                except Exception as e: st.error(f"「購買日」列を日付に変換できませんでした: {e}")
            else: st.info("絞り込みに必要な「購買日」列がデータにありません。")

        with st.expander("キーワードで絞り込む", expanded=False):
            kw_options = ['未選択', 'コメント', '商品名']; st.markdown("---"); st.markdown("**検索条件グループ 1**"); valid_kw_cols_1 = [opt for opt in kw_options if opt in df.columns or opt == '未選択']; comment_col_kw_1 = st.selectbox('検索対象 1', valid_kw_cols_1, key="kw_col_1")
            if comment_col_kw_1 != '未選択':
                search_logic_1 = st.radio("キーワードの条件 (グループ1)", ('いずれかを含める (OR)', '全て含める (AND)'), key='logic_1', horizontal=True); keywords_1 = [st.text_input(f"キーワード1-{i+1}", key=f"kw1_{i}") for i in range(5)]; keywords_1 = [kw.strip() for kw in keywords_1 if kw.strip()]
                if keywords_1:
                    if search_logic_1 == 'いずれかを含める (OR)': pat_1 = '|'.join(map(re.escape, keywords_1)); df_filtered = df_filtered[df_filtered[comment_col_kw_1].astype(str).str.contains(pat_1, na=False)]
                    else:
                        for kw in keywords_1: df_filtered = df_filtered[df_filtered[comment_col_kw_1].astype(str).str.contains(re.escape(kw), na=False)]
            st.markdown("---"); st.markdown("**検索条件グループ 2**"); valid_kw_cols_2 = [opt for opt in kw_options if opt in df.columns or opt == '未選択']; comment_col_kw_2 = st.selectbox('検索対象 2', valid_kw_cols_2, key="kw_col_2")
            if comment_col_kw_2 != '未選択':
                search_logic_2 = st.radio("キーワードの条件 (グループ2)", ('いずれかを含める (OR)', '全て含める (AND)'), key='logic_2', horizontal=True); keywords_2 = [st.text_input(f"キーワード2-{i+1}", key=f"kw2_{i}") for i in range(5)]; keywords_2 = [kw.strip() for kw in keywords_2 if kw.strip()]
                if keywords_2:
                    if search_logic_2 == 'いずれかを含める (OR)': pat_2 = '|'.join(map(re.escape, keywords_2)); df_filtered = df_filtered[df_filtered[comment_col_kw_2].astype(str).str.contains(pat_2, na=False)]
                    else:
                        for kw in keywords_2: df_filtered = df_filtered[df_filtered[comment_col_kw_2].astype(str).str.contains(re.escape(kw), na=False)]

        with st.expander("評価星で絞り込む"):
            if '評価星' in df.columns and not df['評価星'].dropna().empty: 
                min_v, max_v = 1, 5 # スライダーの範囲を整数に設定
                # --- 修正指示 2: 評価星のスライダーを整数値（1-5）に変更 ---
                rating_range = st.slider("スコア範囲", min_v, max_v, (min_v, max_v), step=1)
                # ----------------------------------------------------
                df_filtered = df_filtered[df_filtered['評価星'].between(rating_range[0], rating_range[1])]
            else: st.info("絞り込みに必要な「評価星」列がデータにないか、有効なデータがありません。")

        with st.expander("感情値で絞り込む"):
            emotion_cols_found = []
            for emotion in EMOTION_DICT.keys():
                if emotion in df.columns and pd.api.types.is_numeric_dtype(df[emotion]): 
                    emotion_cols_found.append(emotion); 
                    min_e, max_e = 1, 5
                    # --- 修正指示 2: 感情値のスライダーを整数値（1-5）に変更 ---
                    emo_range = st.slider(f"{emotion}レベル", min_e, max_e, (min_e, max_e), step=1, key=f"emo_{emotion}")
                    # ----------------------------------------------------
                    if 'emo_range' in locals() and emo_range != (min_e, max_e): 
                        df_filtered = df_filtered[df_filtered[emotion].between(emo_range[0], emo_range[1])]
            if not emotion_cols_found: st.info("この機能を利用するには、データに「コメント」列が必要です。")

        st.header("3. 年収別の出費情報をアップロード")
        expenditure_files = st.file_uploader("年収別出費情報ファイル", type=['xlsx'], accept_multiple_files=True, key=f"exp_uploader_{st.session_state.exp_uploader_key}")
        if expenditure_files: st.button("アップロードを一斉削除", on_click=reset_exp_uploader, key="clear_exp", use_container_width=True)

        st.header("4. Wikipediaニュースのアップロード")
        news_files = st.file_uploader("Wikipediaニュースファイル", type=['csv', 'xlsx'], accept_multiple_files=True, key=f"news_uploader_{st.session_state.news_uploader_key}")
        if news_files:
            st.button("アップロードを一斉削除", on_click=reset_news_uploader, key="clear_news", use_container_width=True)
            st.session_state.news_data = {}
            for news_file in news_files:
                try:
                    year_match = re.search(r'(\d{4})', news_file.name)
                    if year_match:
                        year = year_match.group(1)
                        temp_news_df = pd.read_csv(news_file, header=0) if news_file.name.endswith('.csv') else pd.read_excel(news_file, header=0)
                        temp_news_df.columns = [re.sub(r'^[A-Z]列: ', '', col) for col in temp_news_df.columns]
                        if '月日' in temp_news_df.columns and '情報' in temp_news_df.columns:
                            temp_news_df['日付'] = year + '年' + temp_news_df['月日'].astype(str)
                            temp_news_df['日付'] = pd.to_datetime(temp_news_df['日付'], format='%Y年%m月%d日', errors='coerce')
                            st.session_state.news_data[news_file.name] = temp_news_df.dropna(subset=['日付'])
                        else:
                            st.warning(f"ファイル「{news_file.name}」に「月日」列または「情報」列が見つかりませんでした。")
                except Exception as e:
                    st.error(f"ニュースファイル {news_file.name} の読み込みエラー: {e}")

    st.header("絞り込み結果")
    
    # 【修正: 期間不正時の0件表示対応】
    total_person_count = df['ソースファイル'].nunique() if 'ソースファイル' in df.columns else 0
    
    if df_filtered.empty:
         # df_filteredが空の場合、常に0件・0人として表示
         st.write(f"（0件・0人 / 全{len(df)}件・全{total_person_count}人）")
    elif 'ソースファイル' in df.columns:
        filtered_person_count = df_filtered['ソースファイル'].nunique()
        st.write(f"（{len(df_filtered)}件・{filtered_person_count}人 / 全{len(df)}件・全{total_person_count}人）")
    else: 
        st.write(f"（{len(df_filtered)}件 / 全{len(df)}件）")
        
    df_display_filtered = df_filtered.copy(); date_cols_filtered = [col for col in ['購買日', '報告年月', '日付'] if col in df_display_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_display_filtered[col])]
    for col in date_cols_filtered: df_display_filtered[col] = df_display_filtered[col].dt.date
    st.dataframe(df_display_filtered)

    # タブ設定
    tab_names = ["📊 1. 購買グループ・クラスター分析", "💰 2. 年収別出費分析"]
    tabs = st.tabs(tab_names); tab1, tab2 = tabs[0], tabs[1]
    analysis_options = [opt for opt in ['商品名', 'コメント'] if opt in df.columns]

    with tab1:
        st.header("1. 購買グループ・クラスター分析")
        if len(df_filtered) > 1 and analysis_options:
            with st.expander("分析設定（基準列の選択とクラスタリング）", expanded=True):
                c1, c2 = st.columns(2)
                with c1: sim_base_col = st.selectbox("類似度を測る基準列を選択:", analysis_options, key="sim_base_col_selector")
                with c2: clustering_col = st.selectbox("クラスタリングを行う分析列を選択:", analysis_options, index=min(1, len(analysis_options)-1), key="clustering_col_selector")
                st.write("**基準テキストを検索・選択**"); search_keyword = st.text_input("キーワードを入力して基準の候補を絞り込めます", key="search_axis_item"); source_for_selectbox = df_filtered[sim_base_col].dropna().unique().tolist()
                if search_keyword: filtered_source = [item for item in source_for_selectbox if search_keyword in str(item)]
                else: filtered_source = source_for_selectbox
                
                # --- 修正箇所：基準が「候補がありません」の場合も考慮 ---
                axis_item = st.selectbox("基準を選択:", filtered_source if filtered_source else ["候補がありません"], key="axis_item_selector")
                
                st.markdown("---"); c1, c2 = st.columns(2)
                with c1: analysis_method = st.radio("分析手法を選択してください:", ('TF-IDF (速度重視)', 'SentenceTransformer (精度重視)'), captions=["特定文章中における頻出単語と、多くの文章における頻出単語によって、全文章を高速で特徴付け・分析できます。", "文脈まで考慮した上で高精度分析ができますが、処理に時間がかかります。"], horizontal=True, key="analysis_method_selector")
                
                with c2:
                    st.write("**クラスタリングの感度調整**")
                    cluster_param_method = st.radio(
                        "クラスタリングの基準:", 
                        ('コサイン距離', 'コサイン距離(特別版)', 'クラスター数'),
                        captions=[
                            "基準との類似度が一定以上のデータだけでクラスタリングできます。",
                            "基準との類似度は集計せず、クラスター内部の要素同士の類似度検証に有効。",
                            "基準との類似度は集計せず、特定のクラスター数でクラスタリングできます。"
                        ],
                        horizontal=False, key="cluster_param_method"
                    )
                    
                    is_cluster_count_mode = 'クラスター数' in cluster_param_method
                    n_clusters_input = st.number_input("クラスター数(分類グループ)の数", min_value=2, value=st.session_state.n_clusters_input, step=1, key="n_clusters_input", disabled=not is_cluster_count_mode, help="分類したいグループ의 수를 半角英数で指定します。")
                    
                    is_threshold_mode = 'コサイン距離' in cluster_param_method
                    st.write("###### コサイン距離 (最低ライン)の設定")
                    st.caption("グループ化する際の「許容できる最大距離」を設定します。0に近いほど厳密に（似ているものだけ）、2に近いほど緩やかに（異なるものも）グループ化します。")
                    
                    distance_threshold = st.slider(
                        "コサイン距離 (最小値)",
                        0.00, 2.00, 
                        st.session_state.distance_threshold, 
                        0.05, 
                        key="distance_threshold", 
                        disabled=not is_threshold_mode
                    )

                # --- 修正箇所：条件式の緩和（特別版・クラスター数の場合は基準不問） ---
                exec_condition = False
                if 'コサイン距離' == cluster_param_method: # 通常版
                    if axis_item and axis_item != "候補がありません":
                        exec_condition = True
                else: # 特別版 or クラスター数
                    exec_condition = True

                if st.button("📈 類似度・クラスタリング実行") and exec_condition:
                    with st.spinner('計算中...'):
                        # 【重要】実行ボタンが押されたタイミングのdf_filteredをフリーズ保存
                        st.session_state.frozen_df_filtered = df_filtered.copy()
                        # 【重要】期間バリデーション用の範囲もこの時点でフリーズ保存
                        if not df_filtered.empty and '購買日' in df_filtered.columns:
                            min_d = df_filtered['購買日'].min().date()
                            max_d = df_filtered['購買日'].max().date()
                            st.session_state.frozen_sidebar_range = (min_d, max_d)
                        else:
                            st.session_state.frozen_sidebar_range = (None, None)
                        
                        # 【追加】絞り込みモードのフリーズ保存
                        st.session_state.frozen_filter_mode = st.session_state.date_filter_type

                        # 以下、フリーズされたデータを使用して計算
                        df_cleaned = df_filtered.dropna(subset=[sim_base_col, clustering_col]).copy()
                        if not df_cleaned.empty:
                            texts_for_clustering = df_cleaned[clustering_col].astype(str).tolist()
                            if analysis_method == 'TF-IDF (速度重視)': vectors_for_clustering = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit_transform(texts_for_clustering).toarray()
                            else: vectors_for_clustering = load_st_model().encode(texts_for_clustering, show_progress_bar=True)
                            st.session_state.vectors_for_clustering = vectors_for_clustering
                            
                            # 類似度計算用のベクトル作成（基準が選択されている場合のみ）
                            has_valid_axis = axis_item and axis_item != "候補がありません"
                            if has_valid_axis:
                                texts_for_similarity = df_cleaned[sim_base_col].astype(str).tolist()
                                if analysis_method == 'TF-IDF (速度重視)': 
                                    sim_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit(texts_for_similarity); 
                                    vectors_for_similarity = sim_vectorizer.transform(texts_for_similarity); 
                                    axis_item_vector = sim_vectorizer.transform([str(axis_item)])
                                else: 
                                    model = load_st_model(); 
                                    vectors_for_similarity = model.encode(texts_for_similarity, show_progress_bar=True); 
                                    axis_item_vector = model.encode([str(axis_item)], show_progress_bar=False)

                            current_method = st.session_state.cluster_param_method; current_n_clusters = st.session_state.n_clusters_input
                            current_dist = st.session_state.distance_threshold 

                            if 'クラスター数' in current_method: n_clusters_param = current_n_clusters; distance_threshold_param = None
                            else: 
                                distance_threshold_param = current_dist; n_clusters_param = None
                            
                            clustering = AgglomerativeClustering(n_clusters=n_clusters_param, distance_threshold=distance_threshold_param, linkage='average', metric='cosine').fit(vectors_for_clustering)
                            df_cleaned['分類グループ'] = clustering.labels_; 
                            
                            # --- 修正箇所：基準ラベルと類似度計算の制御 ---
                            if has_valid_axis:
                                df_cleaned['コサイン類似度'] = cosine_similarity(axis_item_vector, vectors_for_similarity).flatten()
                                df_cleaned[f'基準の{sim_base_col}'] = axis_item
                            else:
                                if 'コサイン類似度' not in df_cleaned.columns:
                                    df_cleaned['コサイン類似度'] = 0.0 # ダミー値
                                df_cleaned[f'基準の{sim_base_col}'] = "（基準なし）"

                            st.session_state.result_df, st.session_state.sim_base_col, st.session_state.clustering_col = df_cleaned, sim_base_col, clustering_col
                            
                            st.session_state.active_distance_threshold = current_dist
                            st.session_state.active_n_clusters = current_n_clusters; st.session_state.active_filter_method = current_method
                            
                            # 初期表示用のactive_plot_dfもリセット（または更新）
                            st.session_state.active_plot_df = None
                            
                            st.success("計算が完了しました。")
                        else: st.error("分析対象のデータがありません。")
                st.caption("分析条件を変更した場合、もう一度このボタンをクリック")

            with st.expander("最適なクラスター数の算出", expanded=False):
                st.info("""**【分析の流れ（「コサイン距離」基準を使う場合）】**\n
1. コサイン距離を変化させるとクラスター数がどう変わるか、下のグラフで確認します。\n
2. グラフのカーブが急激に変わり始める点（エルボー点）が、最適な距離（しきい値）の候補です。\n
3. グラフの赤い点線を参考に、上の分析設定にある**コサイン距離 (最低ライン)スライダー**を調整し、最終的な分析を実行します。""")
                if len(df_filtered) > 1 and analysis_options:
                    st.markdown("---"); st.subheader("コサイン距離とクラスター数の関係グラフ"); plot_c1, plot_c2 = st.columns(2)
                    with plot_c1: plot_clustering_col = st.selectbox("グラフの生成対象列:", analysis_options, index=min(1, len(analysis_options)-1), key="plot_clustering_col_selector")
                    with plot_c2: plot_analysis_method = st.radio("分析手法:", ('TF-IDF (速度重視)', 'SentenceTransformer (精度重視)'), captions=["特定文章中における頻出単語と、多くの文章における頻出単語によって、全文章を高速でクラスター分けできます。", "文脈まで考慮した上で高精度にクラスター分析できますが、処理に時間がかかります。"], key="plot_analysis_method_selector")
                    if st.button("📊 グラフを生成"):
                        df_for_plot = df_filtered.dropna(subset=[plot_clustering_col]).copy()
                        with st.spinner(f"「{plot_clustering_col}」のベクトルを計算し、グラフを生成中..."):
                            texts_for_plot = df_for_plot[plot_clustering_col].astype(str).tolist()
                            if plot_analysis_method == 'TF-IDF (速度重視)': vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit_transform(texts_for_plot).toarray()
                            else: vectors = load_st_model().encode(texts_for_plot, show_progress_bar=True)
                            st.session_state.vectors_for_clustering, st.session_state.vector_source_col = vectors, plot_clustering_col
                            linkage_matrix = shc.linkage(vectors, method='average', metric='cosine')
                            
                            dist_thresholds = np.arange(0.0, 2.05, 0.05)
                            cluster_counts = [len(np.unique(shc.fcluster(linkage_matrix, t=t, criterion='distance'))) for t in dist_thresholds]

                            x_vals = dist_thresholds 
                            y_vals = np.array(cluster_counts)
                            
                            elbow_idx = 0
                            if y_vals.max() != y_vals.min():
                                x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
                                y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
                                vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
                                vec = vec / np.linalg.norm(vec)
                                vec_pts = np.stack([x_norm - x_norm[0], y_norm - y_norm[0]], axis=1)
                                dists = np.abs(vec_pts[:, 0] * vec[1] - vec_pts[:, 1] * vec[0])
                                elbow_idx = np.argmax(dists)
                            
                            optimal_dist = dist_thresholds[elbow_idx]
                            optimal_cluster_count = cluster_counts[elbow_idx]

                            fig, ax = plt.subplots(figsize=(12, 7))
                            ax.plot(dist_thresholds, cluster_counts, marker='o', linestyle='-')
                            
                            ax.set_title('コサイン距離とクラスター数の関係'); 
                            ax.set_xlabel('コサイン距離'); 
                            ax.set_ylabel('クラスター数'); ax.grid(True, linestyle='--', alpha=0.6)
                            ax.set_xlim(-0.05, 2.05)

                            ax.axvline(x=optimal_dist, color='r', linestyle='--', label=f'最適なポイント（エルボー点）: 距離 {optimal_dist:.2f}')
                            ax.axhline(y=optimal_cluster_count, color='r', linestyle=':', label=f'（クラスター数: {optimal_cluster_count}）')
                            ax.legend()

                            if cluster_counts: ax.set_ylim(0, max(cluster_counts) * 1.1)
                            st.pyplot(fig)
                            
                            start_dist = dist_thresholds[0]
                            end_dist = dist_thresholds[-1]
                            flat_start_idx = len(dist_thresholds) - 1
                            for i in range(elbow_idx + 1, len(dist_thresholds)):
                                if cluster_counts[i] <= 1:
                                    flat_start_idx = i; break
                            flat_dist = dist_thresholds[flat_start_idx]

                            st.markdown(f"""
                            このクラスタリンググラフにおいて最適なポイント（エルボー点）と言えるのは、
                            コサイン距離が **{optimal_dist:.2f}** 付近です。

                            **急な坂道（{start_dist:.2f} ～ {optimal_dist:.2f}あたり）：**
                            少し距離のしきい値を変えるだけで、クラスター数が激減しています。
                            ここはまだ分類が細かすぎる状態です。

                            **地面に着く直前・カーブの部分（{optimal_dist:.2f} ～ {flat_dist:.2f}あたり）：**
                            急降下が落ち着き、なだらかになり始める場所です。ここがエルボー（肘）と呼ばれる最適なポイント候補です。
                            これ以上統合すると、全然違うもの同士を無理やりくっつけることになる手前のラインです。

                            **完全に平らな部分（{flat_dist:.2f} ～ {end_dist:.2f}あたり）：**
                            クラスター数がほぼ1や0に近い状態です。これは全部まとめて1つのグループにしてしまった状態ですので、ここを選んではいけません。
                            """)

            if st.session_state.get('result_df') is not None:
                st.subheader("分析結果")
                result_df_to_show = st.session_state.result_df.copy(); sim_base_col_res = st.session_state.sim_base_col
                
                active_method = st.session_state.get('active_filter_method', 'コサイン距離')
                active_dist = st.session_state.get('active_distance_threshold', 0.5)
                active_n = st.session_state.get('active_n_clusters', 5)
                
                is_special_mode = (active_method == 'コサイン距離(特別版)' or active_method == 'クラスター数')

                if active_method == 'コサイン距離':
                    similarity_threshold = 1 - active_dist
                    result_df_to_show = result_df_to_show[result_df_to_show['コサイン類似度'] >= similarity_threshold]
                    st.info(f"コサイン類似度が {similarity_threshold:.2f} 以上のデータを表示中。（補足：コサイン類似度とは、文章の内容がどれだけ似ているかを表す数値です。最大値は1.00で完全一致、最小値は-1.00で全然異なります)")
                elif active_method == 'コサイン距離(特別版)': 
                    st.info("補足：グループ中身のコサイン類似度とは、グループ中身の各文章がどれだけ似ているかを表す数値です。最大値は1.00で完全一致、最小値は-1.00で全然異なります。")
                elif active_method == 'クラスター数':
                    st.info("補足：グループ中身のコサイン類似度とは、グループ中身の各文章がどれだけ似ているかを表す数値です。最大値は1.00で完全一致、最小値は-1.00で全然異なります。")
                else: # fallback
                    st.info(f"（表示中の結果）基準: {active_method}。")

                result_df_to_show.rename(columns={sim_base_col_res: f'比較対象_{sim_base_col_res}'}, inplace=True)
                if 'ソースファイル' in result_df_to_show.columns: result_df_to_show['UserID'] = result_df_to_show['ソースファイル'].str[:3]

                if not result_df_to_show.empty:
                    total_records = len(df)
                    
                    group_counts = result_df_to_show['分類グループ'].value_counts(); result_df_to_show['分類グループの存在割合'] = result_df_to_show['分類グループ'].map((group_counts / total_records) * 100).apply(lambda x: f"{x:.2f}%")
                    if st.session_state.vectors_for_clustering is not None:
                        full_res_df = st.session_state.result_df; full_vectors = st.session_state.vectors_for_clustering; group_sim_map = {}; groups_to_calc = result_df_to_show['分類グループ'].unique()
                        for g_id in groups_to_calc:
                            indices = np.where(full_res_df['分類グループ'] == g_id)[0]
                            if len(indices) > 1: group_vecs = full_vectors[indices]; sim_matrix = cosine_similarity(group_vecs); n = len(indices); sum_sim = np.sum(sim_matrix); avg_sim = (sum_sim - n) / (n * (n - 1)); group_sim_map[g_id] = avg_sim
                            else: group_sim_map[g_id] = 1.000
                        result_df_to_show['グループ中身のコサイン類似度'] = result_df_to_show['分類グループ'].map(group_sim_map).apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

                st.markdown("---"); st.write("#### ⚙️ 分類グループの共通単語を分析")
                c1, c2 = st.columns(2)
                with c1:
                    common_analysis_options = []; 
                    if '商品名' in analysis_options: common_analysis_options.append('商品名')
                    if 'コメント' in analysis_options: common_analysis_options.append('コメント')
                    if '商品名' in analysis_options and 'コメント' in analysis_options: common_analysis_options.append('商品名とコメント')
                    default_index = 0
                    if common_analysis_options:
                        if "商品名とコメント" in common_analysis_options: default_index = common_analysis_options.index("商品名とコメント")
                        elif "コメント" in common_analysis_options: default_index = common_analysis_options.index("コメント")
                    common_word_source_col = st.selectbox("共通単語の分析対象とする列:", common_analysis_options, key="common_word_source_selector", index=default_index)
                with c2: top_n_common = st.number_input("共通単語の表示数", min_value=3, max_value=6, value=3, step=1, key="top_n_common_selector")
                
                base_cols = ['UserID', f'基準の{sim_base_col_res}', f'比較対象_{sim_base_col_res}', 'コサイン類似度', '分類グループ']; 
                if is_special_mode and 'コサイン類似度' in base_cols: base_cols.remove('コサイン類似度')
                if 'グループ中身のコサイン類似度' in result_df_to_show.columns: base_cols.append('グループ中身のコサイン類似度')
                if '分類グループの存在割合' in result_df_to_show.columns: base_cols.append('分類グループの存在割合')

                with st.spinner("共通単語を計算中..."):
                    if common_word_source_col == '商品名とコメント':
                        product_col_for_common_words = f'比較対象_商品名' if f'比較対象_商品名' in result_df_to_show.columns else '商品名'; comment_col_for_common_words = f'比較対象_コメント' if f'比較対象_コメント' in result_df_to_show.columns else 'コメント'
                        map_product = get_common_words_by_group(result_df_to_show, product_col_for_common_words, '分類グループ', top_n=top_n_common); result_df_to_show[f'商品名の共通単語TOP{top_n_common}'] = result_df_to_show['分類グループ'].map(map_product); base_cols.append(f'商品名の共通単語TOP{top_n_common}')
                        map_comment = get_common_words_by_group(result_df_to_show, comment_col_for_common_words, '分類グループ', top_n=top_n_common); result_df_to_show[f'コメントの共通単語TOP{top_n_common}'] = result_df_to_show['分類グループ'].map(map_comment); base_cols.append(f'コメントの共通単語TOP{top_n_common}')
                    elif common_word_source_col:
                        text_col_for_common_words = f'比較対象_{common_word_source_col}' if f'比較対象_{common_word_source_col}' in result_df_to_show.columns else common_word_source_col
                        common_words_map = get_common_words_by_group(result_df_to_show, text_col_for_common_words, '分類グループ', top_n=top_n_common); result_df_to_show[f'共通単語TOP{top_n_common}'] = result_df_to_show['分類グループ'].map(common_words_map); base_cols.append(f'共通単語TOP{top_n_common}')
                other_cols = [c for c in df.columns if c not in [sim_base_col_res, st.session_state.clustering_col, 'ソースファイル']]
                if st.session_state.clustering_col != sim_base_col_res:
                    if st.session_state.clustering_col not in result_df_to_show.columns: other_cols.insert(0, st.session_state.clustering_col)
                final_cols = base_cols + sorted(list(set(other_cols)))
                if 'UserID' not in result_df_to_show.columns: final_cols.remove('UserID')
                final_cols = [col for col in final_cols if col in result_df_to_show.columns]
                sort_keys = ['分類グループ']; sort_orders = [True]
                if 'コサイン類似度' in final_cols: sort_keys.append('コサイン類似度'); sort_orders.append(False)
                result_df_to_show = result_df_to_show[final_cols].sort_values(by=sort_keys, ascending=sort_orders).reset_index(drop=True)
                df_display_result = result_df_to_show.copy()
                for col in [c for c in ['購買日', '報告年月', '日付'] if c in df_display_result.columns]: df_display_result[col] = pd.to_datetime(df_display_result[col]).dt.date
                st.dataframe(df_display_result); st.divider()

                st.header("補足1：ユーザー購買記録とWikipediaニュース")
                required_cols = ['商品名', '購買日', 'ソースファイル', 'コメント', '評価星']
                if all(col in df.columns for col in required_cols):
                    # 【重要】ここでは常に「実行ボタンを押した時点のデータ(frozen_df_filtered)」を使用する
                    if st.session_state.frozen_df_filtered is not None:
                        plot_df = st.session_state.frozen_df_filtered.copy()
                    else:
                        plot_df = pd.DataFrame(columns=df.columns) # まだ実行されていない場合

                    if not plot_df.empty:
                        plot_df['UserID'] = plot_df['ソースファイル'].str[:3]
                        plot_df['購買日'] = pd.to_datetime(plot_df['購買日'], errors='coerce')
                        plot_df.dropna(subset=['購買日', '商品名'], inplace=True)
                        
                        st.subheader("ユーザー購買行動の深掘り分析")
                        with st.container(border=True):
                            # --- STEP 1 ---
                            with st.expander("STEP 1: 分類グループまたはUserIDでユーザーを絞り込む"):
                                st.write("分析したいUserIDを選択 (複数選択可)で入力された全ユーザーを分析対象にします。\nなお分類グループを指定すれば、そのグループの全ユーザーが、分析したいUserIDを選択 (複数選択可)に自動入力されます。")
                                group_options = []
                                if st.session_state.result_df is not None and not st.session_state.result_df.empty: 
                                    group_options = sorted(st.session_state.result_df['分類グループ'].unique().tolist())
                                
                                target_group_ids = st.multiselect(
                                    "分類グループIDを選択 (複数選択可)", 
                                    options=group_options, 
                                    key="target_group_multiselect", 
                                    on_change=update_users_from_groups, 
                                    help="「1. 購買グループ・クラスター分析」実行後に選択可能になります。選択すると、そのグループに属するUserIDが自動的に下のボックスに入力されます。"
                                )
                                
                                user_options = sorted(plot_df['UserID'].unique().tolist())
                                
                                target_user_ids = st.multiselect(
                                    "分析したいUserIDを選択 (複数選択可)", 
                                    options=user_options, 
                                    key="target_user_multiselect"
                                )
                                
                                force_raw_mode = st.checkbox(
                                    "選択したユーザーの全購買記録を分析する（サイドバーの絞り込み条件・クラスタリング結果を無視）",
                                    help="チェックを入れると、「1. Yahoo！ショッピングの購買記録をアップロード」された全データから、選択されたユーザーの記録を抽出して分析します。チェックを外すと、「⚙️ 分類グループの共通単語を分析」の集計結果（サイドバーで絞り込まれたデータ）をベースに分析します。"
                                )

                            # --- STEP 2 ---
                            with st.expander("STEP 2: 追加キーワードで購買記録を絞り込む"):
                                highlight_logic = st.radio("ハイライト条件のロジック:", ('未選択', 'いずれかを満たす', 'すべてを満たす'), horizontal=True, key="highlight_logic"); highlight_conditions = []
                                for i in range(1, 6):
                                    st.markdown(f"**ハイライト条件 {i}**"); c1, c2 = st.columns(2); highlight_term = c1.text_input(f"ハイライトする単語 {i}", key=f"highlight_term_{i}"); search_col = c2.selectbox(f"検索対象の列 {i}", [""] + ['商品名', 'コメント'], key=f"search_col_{i}")
                                    if highlight_term and search_col: highlight_conditions.append({'term': highlight_term, 'col': search_col})
                            
                            # --- STEP 3 ---
                            with st.expander("STEP 3: 分析期間・対象月を調整"):
                                date_filter_mode = st.radio("期間指定モード:", ('未選択', '期間を指定', '特定月（毎年集計）'), horizontal=True)
                                
                                # デフォルト値はフリーズされたデータから取得
                                default_min = st.session_state.frozen_sidebar_range[0] if st.session_state.frozen_sidebar_range[0] else datetime.date.today()
                                default_max = st.session_state.frozen_sidebar_range[1] if st.session_state.frozen_sidebar_range[1] else datetime.date.today()
                                
                                step3_start_date = None; step3_end_date = None; step3_selected_months = []
                                is_out_of_bounds = False
                                
                                valid_months_in_sidebar = set()
                                valid_sidebar_min = None
                                valid_sidebar_max = None
                                
                                if not force_raw_mode and st.session_state.frozen_df_filtered is not None and not st.session_state.frozen_df_filtered.empty:
                                    valid_months_in_sidebar = set(st.session_state.frozen_df_filtered['購買日'].dt.month.unique())
                                    valid_sidebar_min = st.session_state.frozen_sidebar_range[0]
                                    valid_sidebar_max = st.session_state.frozen_sidebar_range[1]

                                if date_filter_mode == '期間を指定':
                                    c1, c2 = st.columns(2)
                                    start_limit_s3 = datetime.date(2010, 1, 1)
                                    end_limit_s3 = datetime.date(2035, 1, 31)

                                    with c1: 
                                        step3_start_date = st.date_input('開始日', default_min, key="s3_start", min_value=start_limit_s3, max_value=end_limit_s3)
                                    with c2: 
                                        step3_end_date = st.date_input('終了日', default_max, key="s3_end", min_value=start_limit_s3, max_value=end_limit_s3)
                                    
                                    if not force_raw_mode:
                                        if st.session_state.get('frozen_filter_mode') == '期間を指定':
                                            if valid_sidebar_min and valid_sidebar_max and step3_start_date and step3_end_date:
                                                if step3_start_date < valid_sidebar_min or step3_end_date > valid_sidebar_max:
                                                    is_out_of_bounds = True
                                        elif st.session_state.get('frozen_filter_mode') == '月を選択':
                                            if valid_months_in_sidebar and step3_start_date and step3_end_date:
                                                curr_date = step3_start_date.replace(day=1)
                                                while curr_date <= step3_end_date:
                                                    if curr_date.month not in valid_months_in_sidebar:
                                                        is_out_of_bounds = True; break
                                                    if curr_date.month == 12: curr_date = datetime.date(curr_date.year + 1, 1, 1)
                                                    else: curr_date = datetime.date(curr_date.year, curr_date.month + 1, 1)

                                elif date_filter_mode == '特定月（毎年集計）': 
                                    step3_selected_months = st.multiselect("対象月を選択 (複数可)", list(range(1, 13)), default=[datetime.date.today().month])
                                    if not step3_selected_months: st.warning("月を選択してください。")
                                    elif not force_raw_mode and valid_months_in_sidebar:
                                        selected_set = set(step3_selected_months)
                                        if not selected_set.issubset(valid_months_in_sidebar): is_out_of_bounds = True

                            # --- 集計 ---
                            if is_out_of_bounds:
                                st.error("「⚙️ 分類グループの共通単語を分析」の指定期間に合わせて下さい。もしくは該当データがありません")
                            else:
                                if force_raw_mode and (target_group_ids or target_user_ids):
                                    df_base_for_step3 = df.copy()
                                    df_base_for_step3['UserID'] = df_base_for_step3['ソースファイル'].str[:3]
                                    df_base_for_step3['購買日'] = pd.to_datetime(df_base_for_step3['購買日'], errors='coerce')
                                    df_base_for_step3.dropna(subset=['購買日', '商品名'], inplace=True)
                                else:
                                    df_base_for_step3 = plot_df.copy()
                                
                                count_before_step123 = len(df_base_for_step3)
                                df_for_outputs = df_base_for_step3.copy()
                                all_target_users = set()
                                
                                if target_group_ids:
                                    if st.session_state.result_df is not None and not st.session_state.result_df.empty:
                                        target_df_subset = st.session_state.result_df[st.session_state.result_df['分類グループ'].isin(target_group_ids)].copy()
                                        if 'UserID' not in target_df_subset.columns and 'ソースファイル' in target_df_subset.columns:
                                            target_df_subset['UserID'] = target_df_subset['ソースファイル'].str[:3]
                                        if 'UserID' in target_df_subset.columns:
                                            users_from_group = target_df_subset['UserID'].unique()
                                            if len(users_from_group) > 0: all_target_users.update(users_from_group)
                                    else: st.info("分類グループIDでの絞り込みは、「1. 購買グループ・クラスター分析」を実行した後に有効になります。")

                                if target_user_ids: all_target_users.update(target_user_ids)

                                if not target_group_ids and not target_user_ids:
                                    final_user_list_step1 = df_for_outputs['UserID'].unique().tolist() if 'UserID' in df_for_outputs.columns else []
                                else:
                                    final_user_list_step1 = sorted(list(all_target_users))

                                if final_user_list_step1 and not df_for_outputs.empty: 
                                    df_for_outputs = df_for_outputs[df_for_outputs['UserID'].isin(final_user_list_step1)]; 
                                
                                final_plot_df = df_for_outputs.copy()
                                if not final_plot_df.empty and highlight_conditions and highlight_logic != '未選択':
                                    if highlight_logic == 'いずれかを満たす': mask = np.logical_or.reduce([df_for_outputs[cond['col']].astype(str).str.contains(re.escape(cond['term']), na=False) for cond in highlight_conditions])
                                    else: mask = np.logical_and.reduce([df_for_outputs[cond['col']].astype(str).str.contains(re.escape(cond['term']), na=False) for cond in highlight_conditions])
                                    final_plot_df = df_for_outputs[mask]; st.success(f"STEP 2の結果: キーワード条件に一致する {len(final_plot_df)} 件の購買記録に絞り込みました。")
                                
                                if not final_plot_df.empty:
                                    if date_filter_mode == '期間を指定' and step3_start_date and step3_end_date:
                                        if step3_start_date <= step3_end_date:
                                            final_plot_df = final_plot_df[(final_plot_df['購買日'].dt.date >= step3_start_date) & (final_plot_df['購買日'].dt.date <= step3_end_date)]
                                        else: final_plot_df = pd.DataFrame(columns=final_plot_df.columns)
                                    elif date_filter_mode == '特定月（毎年集計）' and step3_selected_months:
                                        final_plot_df = final_plot_df[final_plot_df['購買日'].dt.month.isin(step3_selected_months)]
                                
                                count_after_step123 = len(final_plot_df)
                                st.info(f"STEP 1~3 適用後のデータ件数: {count_before_step123}件 ➡ {count_after_step123}件")

                                if st.session_state.active_plot_df is None:
                                    st.session_state.active_plot_df = final_plot_df

                                # --- OUTPUT 1 ---
                                with st.expander("OUTPUT 1: ユーザー別の商品購買記録", expanded=False):
                                    if st.session_state.active_plot_df is not None and not st.session_state.active_plot_df.empty:
                                        plot_df_display = st.session_state.active_plot_df.copy()
                                        st.subheader("ユーザー別の商品購買記録")
                                        user_ids = sorted(plot_df_display['UserID'].unique()); user_map = {uid: i for i, uid in enumerate(user_ids)}; plot_df_for_viz = plot_df_display.copy(); plot_df_for_viz['y_axis'] = plot_df_for_viz['UserID'].map(user_map); hover_cols = ['商品名', 'コメント', '評価星'] + list(EMOTION_DICT.keys())
                                        fig = px.line(plot_df_for_viz.sort_values(by='購買日'), x='購買日', y='y_axis', color='UserID', markers=True, hover_name='商品名', hover_data=hover_cols); st.plotly_chart(fig, use_container_width=True)
                                        
                                        excel_buffer_1 = BytesIO()
                                        with pd.ExcelWriter(excel_buffer_1, engine='openpyxl') as writer:
                                            export_df1 = plot_df_for_viz[['UserID', '購買日', '商品名', 'コメント', '評価星']].copy()
                                            export_df1['購買日'] = export_df1['購買日'].dt.date
                                            export_df1.to_excel(writer, index=False)
                                        st.download_button(label="📥 購買記録データをExcelでダウンロード", data=excel_buffer_1.getvalue(), file_name="output1_user_purchases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_op1_1")

                                        st.subheader("日別の総購買回数")
                                        daily_counts_df = plot_df_display.groupby(plot_df_display['購買日'].dt.date).size().reset_index(name='購買回数'); st.plotly_chart(px.bar(daily_counts_df, x='購買日', y='購買回数', title='日別の総購買回数（絞り込み結果）'), use_container_width=True)

                                    st.markdown("---")
                                    if st.button("🔄 情報更新", help="STEP 1~3の変更をグラフに反映します"):
                                        st.session_state.active_plot_df = final_plot_df; st.rerun()

                                # --- OUTPUT 2 ---
                                is_expanded_op2 = st.session_state.output2_ranking_output is not None and not st.session_state.output2_ranking_output.empty
                                with st.expander("OUTPUT 2: 頻出単語解析", expanded=is_expanded_op2):
                                    # 【修正】セッションステート(active_plot_df)ではなく、現在のStep1-3適用結果(final_plot_df)が存在すれば優先的に使用する
                                    target_df_for_op2 = final_plot_df if 'final_plot_df' in locals() and final_plot_df is not None and not final_plot_df.empty else st.session_state.active_plot_df

                                    if target_df_for_op2 is not None and not target_df_for_op2.empty:
                                        plot_df_display = target_df_for_op2.copy()
                                        op2_options = [c for c in ['商品名', 'コメント'] if c in plot_df_display.columns]
                                        if op2_options:
                                            op2_target = st.selectbox("解析対象の列を選択", op2_options, key="op2_select")
                                            if st.button("頻出単語ランキングを表示", key="btn_op2_ranking"):
                                                with st.spinner("解析中..."):
                                                    texts = plot_df_display[op2_target].dropna().astype(str).tolist()
                                                    t = get_tokenizer(); docs, all_words = [], []
                                                    for text in texts:
                                                        tokens = [tok.base_form for tok in t.tokenize(text) if tok.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞'] and len(tok.surface) > 1]
                                                        docs.append(tokens); all_words.extend(tokens)
                                                    if not all_words: st.session_state.output2_ranking_output = pd.DataFrame()
                                                    else:
                                                        word_counts = Counter(all_words); top_words = word_counts.most_common(10); ranking_data = []
                                                        for rank, (target_word, count) in enumerate(top_words, 1):
                                                            co_counter = Counter()
                                                            for doc in docs:
                                                                if target_word in doc: co_counter.update([w for w in doc if w != target_word])
                                                            top3_co = co_counter.most_common(3); co_str = ", ".join([f"{w}({c})" for w, c in top3_co]) if top3_co else "なし"
                                                            ranking_data.append({'順位': rank, '単語': target_word, '出現回数': count, '共起単語TOP3 (共起回数)': co_str})
                                                        st.session_state.output2_ranking_output = pd.DataFrame(ranking_data)
                                            if st.session_state.output2_ranking_output is not None:
                                                if not st.session_state.output2_ranking_output.empty: st.dataframe(st.session_state.output2_ranking_output, use_container_width=True)
                                                else: st.info("頻出単語が見つかりませんでした。")

                        # --- Wikipediaニュース（旧版の機能・ヴィジュアルに書き換え） ---
                        with st.expander("Wikipediaニュース", expanded=True):
                            if 'news_data' in st.session_state and st.session_state.news_data:
                                valid_months_in_sidebar_wiki = set(); valid_sidebar_min_wiki = None; valid_sidebar_max_wiki = None
                                active_filter_mode_sidebar = st.session_state.get('date_filter_type', '期間を指定')
                                if active_filter_mode_sidebar == '月を選択':
                                    current_selected_months = [i for i in range(1, 13) if st.session_state.get(f"month_{i}", False)]
                                    if current_selected_months: valid_months_in_sidebar_wiki = set(current_selected_months)
                                else:
                                    if df_filtered is not None and not df_filtered.empty and '購買日' in df_filtered.columns:
                                        valid_sidebar_min_wiki = df_filtered['購買日'].min().date(); valid_sidebar_max_wiki = df_filtered['購買日'].max().date()

                                valid_news_files = [filename for filename, news_df in st.session_state.news_data.items() if '日付' in news_df.columns]
                                if valid_news_files:
                                    tabs_list = valid_news_files
                                    news_tabs = st.tabs(tabs_list)
                                    for i, filename in enumerate(valid_news_files):
                                        with news_tabs[i]:
                                            news_df = st.session_state.news_data[filename].copy()
                                            if '日付' in news_df.columns: news_df['日付'] = news_df['日付'].dt.date
                                            st.subheader(f"該当期間のニュース ({filename})")
                                            current_filtered_news = news_df.sort_values(by='日付', na_position='last').reset_index(drop=True)
                                            mask = pd.Series([True] * len(current_filtered_news))
                                            if active_filter_mode_sidebar == '月を選択':
                                                if valid_months_in_sidebar_wiki: mask = pd.to_datetime(current_filtered_news['日付']).dt.month.isin(valid_months_in_sidebar_wiki)
                                            else:
                                                if valid_sidebar_min_wiki and valid_sidebar_max_wiki: mask = (current_filtered_news['日付'] >= valid_sidebar_min_wiki) & (current_filtered_news['日付'] <= valid_sidebar_max_wiki)
                                            current_filtered_news.loc[~mask, ['日付', '情報']] = None
                                            base_display_cols = [col for col in news_df.columns if col not in ['月日', '作成日時']]
                                            final_display_cols = (['日付'] + [c for c in base_display_cols if c != '日付']) if '日付' in base_display_cols else base_display_cols
                                            st.dataframe(current_filtered_news[final_display_cols], use_container_width=True)
                                            
                                            st.markdown(f"##### **`{filename}`** の共起ネットワーク分析")
                                            news_texts = current_filtered_news['情報'].dropna().astype(str).tolist()
                                            if news_texts:
                                                c1, c2 = st.columns(2)
                                                with c1: top_n_to_display = st.slider("表示単語数", 3, 10, 10, 1, key=f"net_top_n_{filename}")
                                                with c2: k_for_ranking = st.number_input("組み合わせ数", 2, 4, 3, 1, key=f"net_k_{filename}")
                                                if st.button("共起ネットワークを生成", key=f"gen_news_net_{filename}"):
                                                    fig, df_word_rank, df_cooc_k = create_cooccurrence_network_graph(news_texts, get_tokenizer(), top_n_to_display, k_for_ranking)
                                                    st.session_state.news_outputs[filename] = {'fig': fig, 'df_word_rank': df_word_rank, 'df_cooc_k': df_cooc_k, 'top_n': top_n_to_display, 'k': k_for_ranking}
                                                if filename in st.session_state.news_outputs:
                                                    cached_output = st.session_state.news_outputs[filename]; st.plotly_chart(cached_output['fig'], use_container_width=True)
                                                    excel_rows = []; df_word_rank_target = cached_output['df_word_rank'].head(cached_output['top_n'])
                                                    for _, row in df_word_rank_target.iterrows():
                                                        main_word = row['単語']; related_combos = cached_output['df_cooc_k'][cached_output['df_cooc_k']['combo'].apply(lambda c: main_word in c)].head(10)
                                                        if not related_combos.empty:
                                                            is_first, combo_rank = True, 1
                                                            for _, combo_row in related_combos.iterrows():
                                                                combo_str, combo_count = ', '.join(combo_row['combo']), combo_row['count']
                                                                if is_first: excel_rows.append([row['順位'], main_word, row['出現回数'], combo_rank, combo_str, combo_count]); is_first = False
                                                                else: excel_rows.append(['', '', '', combo_rank, combo_str, combo_count])
                                                                combo_rank += 1
                                                        else: excel_rows.append([row['順位'], main_word, row['出現回数'], '', '', ''])
                                                    df_excel = pd.DataFrame(excel_rows, columns=['順位', '単語', '出現回数', f"{cached_output['k']}単語共起 順位", f"{cached_output['k']}単語 組み合わせ", '共起回数'])
                                                    output_excel = io.BytesIO(); df_excel.to_excel(output_excel, index=False); st.download_button(label="📥 共起ネットワークデータをExcelでダウンロード", data=output_excel.getvalue(), file_name=f"news_{filename}_co-occurrence.xlsx", key=f"dl_btn_{filename}")
                                            else: st.info("表示可能なニュース情報がありません。")

                                    
                            else: st.info("サイドバーの「4. Wikipediaニュースのアップロード」からニュースファイルをアップロードしてください。")

            elif not analysis_options: st.warning("分析の軸となる「商品名」または「コメント」列がありません。")

    with tab2:
        st.header("2. 年収別出費分析")
        if expenditure_files:
            sheet_label_map = {'二人以上の世帯': '二人', '二人以上の世帯のうち勤労者世帯': '勤労', '二人以上の世帯のうち無職世帯': '無職'}
            selected_label = st.selectbox("分析対象のシートを選択:", list(sheet_label_map.keys()), key="sheet_choice_label")
            sheet_choice_internal = sheet_label_map[selected_label]
            expenditure_df = process_expenditure_files(expenditure_files, sheet_choice_internal)
            if expenditure_df is not None and not expenditure_df.empty:
                date_options = sorted(expenditure_df['報告年月'].unique().tolist()); selected_date = st.selectbox('表示する年月を選択', date_options, index=len(date_options)-1 if date_options else 0)
                expenditure_df_filtered = expenditure_df[expenditure_df['報告年月'] == selected_date]
                selectable_categories = expenditure_df_filtered['用途分類'].dropna().unique().tolist() if not expenditure_df_filtered.empty else []
                selected_categories = st.multiselect("分析したい用途分類を選択してください", selectable_categories)
                if st.button("📈 グラフ/表を表示", key="show_exp_chart"):
                    if not expenditure_df_filtered.empty and selected_categories:
                        df_display = expenditure_df_filtered[expenditure_df_filtered['用途分類'].isin(selected_categories)].copy()
                        if not df_display.empty:
                            display_table = df_display.set_index('用途分類')
                            target_keywords = ['万円', '未満', '以上', '世帯数', '分布', '調整', '集計', '平均', '人員', '有業', '年齢']
                            income_cols = [col for col in expenditure_df.columns if any(keyword in str(col) for keyword in target_keywords)]
                            final_table = display_table[income_cols]; final_table.insert(0, '単位', display_table['単位'])
                            st.session_state.expenditure_output = {'table': final_table, 'header': f"分析結果：{selected_date} ({selected_label})"}
                if st.session_state.expenditure_output:
                    st.markdown("---"); st.subheader(st.session_state.expenditure_output['header']); st.dataframe(st.session_state.expenditure_output['table'], use_container_width=True)
            else: st.warning("データを読み込めませんでした。")
        else: st.info("サイドバーで「年収別出費情報ファイル」をアップロードしてください。")