"""
knock44.py: 対話 (Dialogue)

単一ターンの対話で、複雑な質問に回答させます。

問題: 東京の電車の乗り換えに関する問題
つばめちゃんが誤った電車に乗ってしまい、
目的地を推測する必要があります。

"""

from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


def single_turn_dialogue():
    """単一ターンの対話を実行"""

    client = create_openai_client()

    question = """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
    東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
    間違えて急行に乗車してしまったことに気付きました。
    自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
    つばめちゃんの目的地でした。

    目的地の駅の名前を答えてください。
    """

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

    response = message.choices[0].message.content or ""

    print("=" * 60)
    print("knock44: 対話（単一ターン）")
    print("=" * 60)
    print("\n【質問】")
    print(question)
    print("\n【回答】")
    print(response)
    print("\n" + "=" * 60)

    return response


if __name__ == "__main__":
    single_turn_dialogue()

# 実行結果:
# ============================================================
# knock44: 対話（単一ターン）
# ============================================================
#
# 【質問】
#
#     つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
#     東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
#     間違えて急行に乗車してしまったことに気付きました。
#     自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
#     つばめちゃんの目的地でした。
#
#     目的地の駅の名前を答えてください。
#
#
# 【回答】
# 九品仏駅
#
# ============================================================
