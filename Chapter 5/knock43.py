"""
knock43.py: 応答のバイアス (Response Bias)

問題42のJMMLU評価について、実験設定を変えると正解率が
変化するかを調べます。ここでは、元の選択肢配置と、
正解の選択肢をすべてDに移動した設定を比較します。
"""

import os

from jmmlu_utils import (
    extract_choice,
    format_jmmlu_prompt,
    load_jmmlu_questions,
    move_correct_choice_to_d,
)
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
    return extract_choice(response)


def evaluate_setting(client, questions, name, transform):
    """Evaluate one experimental setting."""

    correct_count = 0
    predictions = []

    for item in questions:
        transformed = transform(item)
        prediction = answer_question(client, transformed)
        predictions.append(prediction or "抽出失敗")
        correct_count += int(prediction == transformed["answer"])

    accuracy = (correct_count / len(questions)) * 100 if questions else 0.0
    return {
        "name": name,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def investigate_response_bias():
    """Compare accuracy under different JMMLU experimental settings."""

    client = create_openai_client()
    limit = LIMIT if LIMIT > 0 else None
    questions = load_jmmlu_questions(subject=SUBJECT, limit=limit)

    settings = [
        ("元の選択肢配置", lambda item: item),
        ("正解をすべてDに移動", move_correct_choice_to_d),
    ]

    print("=" * 60)
    print("knock43: JMMLU 応答のバイアス調査")
    print("=" * 60)
    print(f"科目: {SUBJECT}")
    print(f"問題数: {len(questions)}")

    results = []
    for name, transform in settings:
        result = evaluate_setting(client, questions, name, transform)
        results.append(result)

        print(f"\n【{name}】")
        print(f"正解数: {result['correct_count']}/{len(questions)}")
        print(f"正解率: {result['accuracy']:.1f}%")
        print(f"予測: {', '.join(result['predictions'])}")

    print("\n" + "=" * 60)
    if len(results) >= 2:
        diff = results[1]["accuracy"] - results[0]["accuracy"]
        print(f"正解率の差: {diff:+.1f}ポイント")
    print("=" * 60)

    return results


if __name__ == "__main__":
    investigate_response_bias()

# 実行結果:
# ============================================================
# knock43: JMMLU 応答のバイアス調査
# ============================================================
# 科目: japanese_history
# 問題数: 10
#
# 【元の選択肢配置】
# 正解数: 9/10
# 正解率: 90.0%
# 予測: B, B, D, D, A, A, D, A, C, A
#
# 【正解をすべてDに移動】
# 正解数: 9/10
# 正解率: 90.0%
# 予測: D, D, D, D, D, D, D, D, D, A
#
# ============================================================
# 正解率の差: +0.0ポイント
# ============================================================
