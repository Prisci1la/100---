# Chapter 5中基于JMMLU练习的工具函数。 / Chapter 5のJMMLUベース演習用ユーティリティ関数。

import csv
import os
import re
import urllib.request
from io import StringIO


DEFAULT_SUBJECT = "japanese_history"
DEFAULT_LABELS = ("A", "B", "C", "D")
JMMLU_RAW_BASE_URL = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU"


def load_jmmlu_questions(subject=DEFAULT_SUBJECT, limit=None):  # 从JMMLU_CSV_PATH或官方GitHub CSV读取JMMLU问题 / JMMLU_CSV_PATHまたは公式GitHub CSVからJMMLU問題を読み込む

    csv_path = os.environ.get("JMMLU_CSV_PATH")
    if csv_path:
        with open(csv_path, encoding="utf-8", newline="") as f:
            text = f.read()
        source = csv_path
    else:
        url = f"{JMMLU_RAW_BASE_URL}/{subject}.csv"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8-sig")
            source = url
        except OSError as exc:
            raise RuntimeError(
                "JMMLU CSVを取得できませんでした。ネットワークを確認するか、"
                "JMMLU_CSV_PATHにローカルCSVのパスを指定してください。"
            ) from exc

    questions = []
    for row in csv.reader(StringIO(text)):
        if len(row) < 6:
            continue

        question, choice_a, choice_b, choice_c, choice_d, answer = row[:6]
        answer = answer.strip().upper()
        if answer == "ANSWER" or question.strip().lower() == "question":
            continue
        if answer not in DEFAULT_LABELS:
            continue

        questions.append({
            "question": question.strip(),
            "choices": {
                "A": choice_a.strip(),
                "B": choice_b.strip(),
                "C": choice_c.strip(),
                "D": choice_d.strip(),
            },
            "answer": answer,
            "source": source,
        })

        if limit is not None and len(questions) >= limit:
            break

    return questions


def format_jmmlu_prompt(item):  # 将一个JMMLU条目格式化为多选题提示词 / 1件のJMMLU項目を多肢選択プロンプトに整形する

    choices = item["choices"]
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

    normalized = (text or "").upper().translate(str.maketrans("ＡＢＣＤ", "ABCD"))
    match = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", normalized)
    return match.group(1) if match else ""


def move_correct_choice_to_d(item):  # 创建正确选项始终为D的变体 / 正解選択肢が常にDになる変種を作成する

    correct_label = item["answer"]
    correct_choice = item["choices"][correct_label]
    wrong_choices = [
        item["choices"][label]
        for label in DEFAULT_LABELS
        if label != correct_label
    ]

    return {
        "question": item["question"],
        "choices": {
            "A": wrong_choices[0],
            "B": wrong_choices[1],
            "C": wrong_choices[2],
            "D": correct_choice,
        },
        "answer": "D",
        "source": item["source"],
    }
