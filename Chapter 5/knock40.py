"""
knock40.py: Zero-Shot推論 (Zero-Shot Reasoning)

このスクリプトは、追加の学習例を与えずに、
直接問題の解答を生成する「zero-shot推論」を実装します。

問題: 9世紀の日本史に関するできごとを年代順に並べる問題
"""

from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


def zero_shot_reasoning():
    """Zero-shot推論で歴史問題を解く"""

    # クライアント初期化
    client = create_openai_client()

    # 解く問題
    question = """
    9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

    ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
    イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
    ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。

    答えは、ア、イ、ウのいずれかの組み合わせで、年代の古い順に答えてください。
    解答のみを「ア→イ→ウ」のような形式で1行で出力してください。
    """

    # APIに質問を送信
    message = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        reasoning_effort="minimal",
        verbosity="low",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    # 結果を表示
    print("=" * 60)
    print("knock40: Zero-Shot推論")
    print("=" * 60)
    print("\n【問題】")
    print(question)
    print("\n【解答】")
    print(message.choices[0].message.content)
    print("\n" + "=" * 60)

    return message.choices[0].message.content


if __name__ == "__main__":
    zero_shot_reasoning()

# 実行結果:
# ============================================================
# knock40: Zero-Shot推論
# ============================================================
#
# 【問題】
#
#     9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。
#
#     ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
#     イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
#     ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
#
#     答えは、ア、イ、ウのいずれかの組み合わせで、年代の古い順に答えてください。
#     解答のみを「ア→イ→ウ」のような形式で1行で出力してください。
#
#
# 【解答】
# イ→ウ→ア
#
# ============================================================
