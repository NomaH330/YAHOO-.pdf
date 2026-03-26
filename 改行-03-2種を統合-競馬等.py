import re
import pandas as pd
from datetime import datetime
import os
from collections import OrderedDict

def process_events_from_text(file_content: str) -> list[tuple[str, str]]:
    """
    【依頼1】の要件に基づき、出来事テキスト（2024競馬出来事.txt）を整形する関数。
    月と日にちを結合し、指定されたルールで改行や整形を行う。
    """
    # 連続する空行を削除
    text = re.sub(r'\n{2,}', '\n', file_content)
    lines = text.strip().split('\n')
    
    events = []
    current_month = ''
    
    # 正規表現パターンの定義
    # 条件3: 「・」や「～」を含む日付形式
    date_pattern = re.compile(
        r'^(?P<date>\d{1,2}月\d{1,2}日(?:(?:～|・)\d{1,2}月\d{1,2}日)?(?:～\d{1,2}日)?|\d{1,2}日)'
    )
    month_pattern = re.compile(r'^(\d{1,2})月$')

    temp_info_lines = []
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line:
            continue

        # 月のヘッダーを検出
        month_match = month_pattern.match(clean_line)
        if month_match:
            current_month = month_match.group(1) + '月'
            # 月の行自体は出力しない
            continue

        # 日付で始まる行を検出
        date_match = date_pattern.match(clean_line)
        if date_match:
            date_str = date_match.group('date')
            
            # 月情報がない場合は、保持している月の情報を付与
            full_date = date_str if '月' in date_str else current_month + date_str
            
            # 条件1, 6, 7: 日付の後の整形
            info_part = clean_line[len(date_str):].strip()
            info_part = re.sub(r'^\s*-\s*', '', info_part)
            
            # 次の行からが情報の続き
            temp_info_lines = [info_part] if info_part else []
            for next_line in lines[i+1:]:
                next_line_clean = next_line.strip()
                if not next_line_clean or date_pattern.match(next_line_clean) or month_pattern.match(next_line_clean):
                    break
                # 条件2: 文章の途中で改行しない（ここでは情報を結合している）
                temp_info_lines.append(next_line_clean)

            # 条件4, 5: 文末の日付形式に対応
            full_info = ' '.join(temp_info_lines)
            full_info = re.sub(r'(\([^\)]*\d{1,2}日[^\)]*\))。?', r'\1\n', full_info)
            
            # 条件8: 同じ日付が連続する場合の重複を避ける（このロジックでは発生しにくいが念のため）
            if not (events and events[-1][0] == full_date):
                 events.append((full_date, full_info.strip()))

    return events

def process_sources_from_text(file_content: str) -> dict[str, list[tuple[str, str]]]:
    """
    【依頼2】の要件に基づき、出典テキスト（2024競馬出典.txt）を整形する関数。
    指定されたセクションごとに`^`を連番に変換し、改行する。
    """
    # セクションのリスト（順序を保持）
    SECTIONS = ["注釈", "出典", "報道発表", "公式発表", "速報", "一次文献", "個人", "参考文献"]
    
    # re.splitでセクション毎に分割するためのパターン
    pattern = re.compile(r'^\s*(' + '|'.join(SECTIONS) + r')\s*$', re.MULTILINE)
    
    split_content = pattern.split(file_content)
    
    processed_data = OrderedDict()
    # 最初の要素はヘッダー前の不要な部分なので無視
    content_iter = iter(split_content[1:])
    
    for header in content_iter:
        content_block = next(content_iter, "")
        header = header.strip()
        
        if header not in processed_data:
            processed_data[header] = []
        
        # '^'で分割し、空の要素を除外
        parts = [p for p in content_block.split('^') if p.strip()]
        
        for i, part in enumerate(parts, 1):
            source_number = f"[{i}]"
            source_info = part.strip()
            processed_data[header].append((source_number, source_info))
            
    return processed_data

def create_integrated_csv(events_data: list, sources_data: dict, output_filepath: str):
    """
    【依頼3】の要件に基づき、整形された2つのデータから統合CSVファイルを作成する。
    """
    # CSVの列名と順序を定義
    COLUMN_MAPPING = OrderedDict([
        ("注釈", ("C列: 注釈番号", "D列: 注釈情報")),
        ("出典", ("E列: 出典番号", "F列: 出典情報")),
        ("報道発表", ("G列: 報道発表番号", "H列: 報道発表情報")),
        ("公式発表", ("I列: 公式発表番号", "J列: 公式発表情報")),
        ("速報", ("K列: 速報番号", "L列: 速報情報")),
        ("一次文献", ("M列: 一次文献番号", "N列: 一次文献情報")),
        ("個人", ("O列: 個人番号", "P列: 個人情報")),
        ("参考文献", ("Q列: 参考文献番号", "R列: 参考文献情報")),
    ])

    data = OrderedDict()
    
    # A列・B列のデータを準備
    data['A列: 月日'] = [item[0] for item in events_data]
    data['B列: 情報'] = [item[1] for item in events_data]
    
    # C列以降のデータを準備
    for section, (num_col, info_col) in COLUMN_MAPPING.items():
        if section in sources_data:
            section_items = sources_data[section]
            data[num_col] = [item[0] for item in section_items]
            data[info_col] = [item[1] for item in section_items]
        else:
            # 存在しないセクションは空の列として用意
            data[num_col] = []
            data[info_col] = []

    # pandas DataFrameを作成（列の長さが異なっても自動でNaNで補完される）
    df = pd.DataFrame({key: pd.Series(value) for key, value in data.items()})

    # S列: 作成日時を追加
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not df.empty:
      df['S列: 作成日時'] = now

    # CSVファイルに書き出し（文字化け対策: utf-8-sig）
    try:
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"\n✅ CSVファイル '{output_filepath}' を正常に作成しました。")
    except Exception as e:
        print(f"\n❌ CSVファイルの書き出し中にエラーが発生しました: {e}")

def get_user_choice(prompt: str, max_choice: int, exclusion: int = -1) -> int:
    """ユーザーから有効な数値入力を受け取る"""
    choice = 0
    while True:
        try:
            choice_str = input(prompt)
            choice = int(choice_str)
            if 1 <= choice <= max_choice:
                if choice == exclusion:
                    print("エラー: 1つ目と同じファイルは選択できません。")
                    continue
                break
            else:
                print(f"エラー: 1から{max_choice}の間の番号を入力してください。")
        except ValueError:
            print("エラー: 半角数値を入力してください。")
    return choice

if __name__ == '__main__':
    # --- ステップ1: フォルダ内の全TXTファイルを検索 ---
    print("現在のフォルダにあるTXTファイルを検索しています...")
    try:
        available_txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
        if len(available_txt_files) < 2:
            print(f"エラー: フォルダ内に処理対象の.txtファイルが2つ以上見つかりません。（検出数: {len(available_txt_files)}個）")
            input("Enterキーを押して終了します...")
            exit()
        print(f"-> {len(available_txt_files)}個のTXTファイルが見つかりました。")
    except Exception as e:
        print(f"ファイルの検索中にエラーが発生しました: {e}")
        input("Enterキーを押して終了します...")
        exit()

    # --- ステップ2: 処理する2つのファイルを選択 ---
    print("\n--- 【ステップ1】処理するファイルを2つ選択してください ---")
    for i, filename in enumerate(available_txt_files):
        print(f"[{i+1}] {filename}")
    
    choice1 = get_user_choice("1つ目のファイル番号を入力してください: ", len(available_txt_files))
    file1_name = available_txt_files[choice1 - 1]

    choice2 = get_user_choice("2つ目のファイル番号を入力してください: ", len(available_txt_files), exclusion=choice1)
    file2_name = available_txt_files[choice2 - 1]

    # --- 選択したファイルを読み込み ---
    uploaded_files = {}
    try:
        with open(file1_name, 'r', encoding='utf-8') as f:
            uploaded_files[file1_name] = f.read()
        print(f"\n-> ファイル1を読み込みました: {file1_name}")

        with open(file2_name, 'r', encoding='utf-8') as f:
            uploaded_files[file2_name] = f.read()
        print(f"-> ファイル2を読み込みました: {file2_name}")
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        input("Enterキーを押して終了します...")
        exit()
    
    selected_files = list(uploaded_files.keys())

    # --- ステップ3: ファイルの役割を選択 ---
    print("\n--- 【ステップ2】A列・B列（月日・情報）として処理するファイルを選択してください ---")
    for i, filename in enumerate(selected_files):
        print(f"[{i+1}] {filename}")
    
    role_choice = get_user_choice("番号を入力してください (1 or 2): ", len(selected_files))
    file_for_AB_name = selected_files[role_choice - 1]
    
    # 選ばれなかった方を自動的にC列以降用とする
    file_for_C_onward_name = selected_files[1] if role_choice == 1 else selected_files[0]
    
    content_for_AB = uploaded_files[file_for_AB_name]
    content_for_C_onward = uploaded_files[file_for_C_onward_name]

    # --- ステップ4: 出力ファイル名の設定 ---
    print("\n--- 【ステップ3】出力するCSVファイル名を入力してください ---")
    output_csv_file = input("ファイル名 (例: 2024_競馬まとめ.csv): ")
    if not output_csv_file.lower().endswith('.csv'):
        output_csv_file += '.csv'


    # --- 処理を実行 ---
    print("\n-------------------------------------------")
    print("処理を開始します...")
    print(f"A/B列（月日・情報）用ファイル: '{file_for_AB_name}'")
    print(f"C列以降（出典など）用ファイル: '{file_for_C_onward_name}'")
    print(f"出力CSVファイル名: '{output_csv_file}'")
    print("-------------------------------------------\n")

    # 各ファイルを整形
    events_data = process_events_from_text(content_for_AB)
    print(f"-> {len(events_data)}件のイベントデータを抽出しました。")
    sources_data = process_sources_from_text(content_for_C_onward)
    for section, data_list in sources_data.items():
        print(f"-> セクション '{section}' から {len(data_list)}件のデータを抽出しました。")

    # 統合CSVを作成
    create_integrated_csv(events_data, sources_data, output_csv_file)

    input("\n処理が完了しました。Enterキーを押して終了します...")