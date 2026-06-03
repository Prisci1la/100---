# Chapter 5中基于JMMLU练习的工具函数。 / Chapter 5のJMMLUベース演習用ユーティリティ関数。

import csv  # 导入CSV解析模块 / CSV解析モジュールをインポート
import os  # 导入环境变量读取模块 / 環境変数読み取り用モジュールをインポート
import re  # 导入正则表达式模块 / 正規表現モジュールをインポート
import urllib.request  # 导入URL请求模块 / URLリクエスト用モジュールをインポート
from io import StringIO  # 导入字符串流工具 / 文字列ストリーム用ツールをインポート


DEFAULT_SUBJECT = "japanese_history"  # 定义默认JMMLU科目 / 既定のJMMLU科目を定義
DEFAULT_LABELS = ("A", "B", "C", "D")  # 定义允许的选择项标签 / 使用可能な選択肢ラベルを定義
JMMLU_RAW_BASE_URL = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU"  # 定义JMMLU官方CSV基础URL / JMMLU公式CSVのベースURLを定義


def load_jmmlu_questions(subject=DEFAULT_SUBJECT, limit=None):  # 从JMMLU_CSV_PATH或官方GitHub CSV读取JMMLU问题 / JMMLU_CSV_PATHまたは公式GitHub CSVからJMMLU問題を読み込む

    csv_path = os.environ.get("JMMLU_CSV_PATH")  # 从环境变量读取本地CSV路径 / 環境変数からローカルCSVパスを読み取る
    if csv_path:  # 如果指定了本地CSV路径 / ローカルCSVパスが指定されている場合
        with open(csv_path, encoding="utf-8", newline="") as f:  # 用UTF-8打开本地CSV / UTF-8でローカルCSVを開く
            text = f.read()  # 读取CSV全文 / CSV全文を読み込む
        source = csv_path  # 记录数据源为本地路径 / データソースをローカルパスとして記録
    else:
        url = f"{JMMLU_RAW_BASE_URL}/{subject}.csv"  # 生成官方CSV下载URL / 公式CSVのダウンロードURLを生成
        try:  # 尝试从网络获取CSV / ネットワークからCSV取得を試行
            with urllib.request.urlopen(url, timeout=30) as response:  # 打开URL并设置超时 / URLを開きタイムアウトを設定
                text = response.read().decode("utf-8-sig")  # 读取并解码CSV内容 / CSV内容を読み込んでデコード
            source = url  # 记录数据源为URL / データソースをURLとして記録
        except OSError as exc:
            raise RuntimeError(  # 获取失败时抛出易懂错误 / 取得失敗時に分かりやすいエラーを投げる
                "JMMLU CSVを取得できませんでした。ネットワークを確認するか、"
                "JMMLU_CSV_PATHにローカルCSVのパスを指定してください。"
            ) from exc

    questions = []  # 初始化问题列表 / 問題リストを初期化
    for row in csv.reader(StringIO(text)):  # 将CSV文本作为流逐行解析 / CSVテキストをストリームとして行ごとに解析
        if len(row) < 6:  # 跳过字段不足的问题行 / フィールド数が不足する行をスキップ
            continue

        question, choice_a, choice_b, choice_c, choice_d, answer = row[:6]  # 取出问题、四个选项和答案 / 問題、4つの選択肢、解答を取り出す
        answer = answer.strip().upper()  # 清理并统一答案标签为大写 / 解答ラベルを整形して大文字に統一
        if answer == "ANSWER" or question.strip().lower() == "question":  # 跳过表头行 / ヘッダー行をスキップ
            continue
        if answer not in DEFAULT_LABELS:  # 跳过答案标签不合法的行 / 解答ラベルが不正な行をスキップ
            continue

        questions.append({  # 将当前行整理成问题字典 / 現在の行を問題辞書に整形して追加
            "question": question.strip(),  # 保存问题正文 / 問題文を保存
            "choices": {  # 保存四个选项 / 4つの選択肢を保存
                "A": choice_a.strip(),  # 保存A选项 / A選択肢を保存
                "B": choice_b.strip(),  # 保存B选项 / B選択肢を保存
                "C": choice_c.strip(),  # 保存C选项 / C選択肢を保存
                "D": choice_d.strip(),  # 保存D选项 / D選択肢を保存
            },
            "answer": answer,  # 保存正确答案标签 / 正解ラベルを保存
            "source": source,  # 保存数据来源 / データソースを保存
        })

        if limit is not None and len(questions) >= limit:  # 达到读取上限时停止 / 読み込み上限に達したら停止
            break

    return questions  # 返回问题列表 / 問題リストを返す


def format_jmmlu_prompt(item):  # 将一个JMMLU条目格式化为多选题提示词 / 1件のJMMLU項目を多肢選択プロンプトに整形する

    choices = item["choices"]  # 取出选项字典 / 選択肢辞書を取り出す
    return f"""次の多肢選択問題に答えてください。
回答は A, B, C, D のいずれか1文字だけにしてください。

問題:
{item['question']}

選択肢:
A. {choices['A']}
B. {choices['B']}
C. {choices['C']}
D. {choices['D']}
"""


def extract_choice(text):  # 从模型回答中提取第一个A-D选择项 / モデル応答から最初のA-D選択肢を抽出する

    normalized = (text or "").upper().translate(str.maketrans("ＡＢＣＤ", "ABCD"))  # 将回答转大写并把全角ABCD转半角 / 回答を大文字化し全角ABCDを半角に変換
    match = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", normalized)  # 搜索独立的A-D选项字母 / 独立したA-D選択肢文字を検索
    return match.group(1) if match else ""  # 找到则返回选项，否则返回空字符串 / 見つかれば選択肢を返し、なければ空文字列を返す


def move_correct_choice_to_d(item):  # 创建正确选项始终为D的变体 / 正解選択肢が常にDになる変種を作成する

    correct_label = item["answer"]  # 取出原正确答案标签 / 元の正解ラベルを取り出す
    correct_choice = item["choices"][correct_label]  # 取出原正确选项内容 / 元の正解選択肢の内容を取り出す
    wrong_choices = [  # 收集所有错误选项内容 / すべての誤答選択肢の内容を集める
        item["choices"][label]  # 按标签取出选项文本 / ラベルに対応する選択肢テキストを取り出す
        for label in DEFAULT_LABELS
        if label != correct_label
    ]

    return {  # 返回正答被移动到D的新问题 / 正答をDへ移動した新しい問題を返す
        "question": item["question"],  # 保持问题正文不变 / 問題文はそのまま保持
        "choices": {  # 重新构造选项 / 選択肢を再構成
            "A": wrong_choices[0],  # 将第一个错误选项放到A / 1つ目の誤答をAに配置
            "B": wrong_choices[1],  # 将第二个错误选项放到B / 2つ目の誤答をBに配置
            "C": wrong_choices[2],  # 将第三个错误选项放到C / 3つ目の誤答をCに配置
            "D": correct_choice,  # 将正确选项放到D / 正解選択肢をDに配置
        },
        "answer": "D",  # 新正确答案固定为D / 新しい正解ラベルをDに固定
        "source": item["source"],  # 保持数据来源不变 / データソースはそのまま保持
    }
