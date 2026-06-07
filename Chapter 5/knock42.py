"""
knock42.py: 多肢選択問題の正解率 (Multiple Choice Accuracy)

JMMLUのいずれかの科目を解答させ、
その正解率を求めます。
"""

import os

from jmmlu_utils import extract_choice, format_jmmlu_prompt, load_jmmlu_questions
from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


SUBJECT = os.environ.get("JMMLU_SUBJECT", "japanese_history")
LIMIT = int(os.environ.get("JMMLU_LIMIT", "10"))


def answer_question(client, item):
    """Ask one JMMLU multiple-choice question."""

    message = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        reasoning_effort="minimal",
        verbosity="low",
        messages=[
            {
                "role": "user",
                "content": format_jmmlu_prompt(item),
            }
        ],
    )

    response = message.choices[0].message.content or ""
    return response, extract_choice(response)


def calculate_accuracy():
    """Calculate accuracy on one JMMLU subject."""

    client = create_openai_client()
    limit = LIMIT if LIMIT > 0 else None
    questions = load_jmmlu_questions(subject=SUBJECT, limit=limit)

    correct_count = 0

    print("=" * 60)
    print("knock42: JMMLU 多肢選択問題の正解率")
    print("=" * 60)
    print(f"科目: {SUBJECT}")
    print(f"問題数: {len(questions)}")
    if questions:
        print(f"データ: {questions[0]['source']}")

    for i, item in enumerate(questions, 1):
        response, prediction = answer_question(client, item)
        is_correct = prediction == item["answer"]
        correct_count += int(is_correct)

        print(f"\n【問題{i}】")
        print(f"予測: {prediction or '抽出失敗'}")
        print(f"正解: {item['answer']}")
        print(f"判定: {'正解' if is_correct else '不正解'}")
        if not prediction:
            print(f"回答全文: {response}")

    accuracy = (correct_count / len(questions)) * 100 if questions else 0.0

    print("\n" + "=" * 60)
    print(f"正解率: {correct_count}/{len(questions)} = {accuracy:.1f}%")
    print("=" * 60)

    return accuracy


if __name__ == "__main__":
    calculate_accuracy()

# 実行結果:
# ============================================================
# knock42: JMMLU 多肢選択問題の正解率
# ============================================================
# 科目: japanese_history
# 問題数: 10
# データ: https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/japanese_history.csv
#
# 【問題1】
# 予測: B
# 正解: B
# 判定: 正解
#
# 【問題2】
# 予測: B
# 正解: B
# 判定: 正解
#
# 【問題3】
# 予測: D
# 正解: D
# 判定: 正解
#
# 【問題4】
# 予測: D
# 正解: D
# 判定: 正解
#
# 【問題5】
# 予測: A
# 正解: A
# 判定: 正解
#
# 【問題6】
# 予測: A
# 正解: A
# 判定: 正解
#
# 【問題7】
# 予測: D
# 正解: D
# 判定: 正解
#
# 【問題8】
# 予測: A
# 正解: A
# 判定: 正解
#
# 【問題9】
# 予測: C
# 正解: C
# 判定: 正解
#
# 【問題10】
# 予測: A
# 正解: D
# 判定: 不正解
#
# ============================================================
# 正解率: 9/10 = 90.0%
# ============================================================
