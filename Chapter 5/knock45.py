"""
knock45.py: マルチターン対話 (Multi-turn Dialogue)

複数ターンの対話を実装します。
前のターンの会話コンテキストを保持しながら、
新しい質問に答えさせます。
"""

from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


def multi_turn_dialogue():
    """マルチターン対話を実行"""

    client = create_openai_client()

    # マルチターン対話の履歴を保持
    messages = []

    # ターン1: 初期の状況を説明
    initial_question = """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
    東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
    間違えて急行に乗車してしまったことに気付きました。
    自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
    つばめちゃんの目的地でした。

    目的地の駅の名前を答えてください。
    """

    messages.append({
        "role": "user",
        "content": initial_question
    })

    # 最初の回答を取得
    response1 = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        reasoning_effort="minimal",
        verbosity="low",
        messages=messages
    )

    first_answer = response1.choices[0].message.content or ""

    messages.append({
        "role": "assistant",
        "content": first_answer
    })

    print("=" * 60)
    print("knock45: マルチターン対話")
    print("=" * 60)
    print("\n【ターン1: 初期質問】")
    print(initial_question)
    print("\n【回答】")
    print(first_answer)
    if not first_answer.strip():
        print("\n【ターン1 デバッグ】")
        print(f"finish_reason: {response1.choices[0].finish_reason}")
        print(f"usage: {response1.usage}")

    # ターン2: 異なるシナリオで追加質問
    follow_up_question = """
    さらに、つばめちゃんが自由が丘駅で乗り換えたとき、
    先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
    目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、
    反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？

    """

    messages.append({
        "role": "user",
        "content": follow_up_question
    })

    # 2番目の回答を取得
    response2 = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        reasoning_effort="minimal",
        verbosity="low",
        messages=messages
    )

    second_answer = response2.choices[0].message.content or ""

    print("\n【ターン2: フォローアップ質問】")
    print(follow_up_question)
    print("\n【回答】")
    print(second_answer)
    if not second_answer.strip():
        print("\n【ターン2 デバッグ】")
        print(f"finish_reason: {response2.choices[0].finish_reason}")
        print(f"usage: {response2.usage}")
    print("\n" + "=" * 60)

    return {
        "first_answer": first_answer,
        "second_answer": second_answer
    }


if __name__ == "__main__":
    multi_turn_dialogue()

# 実行結果:
# ============================================================
# knock45: マルチターン対話
# ============================================================
#
# 【ターン1: 初期質問】
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
# 【ターン2: フォローアップ質問】
#
#     さらに、つばめちゃんが自由が丘駅で乗り換えたとき、
#     先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
#     目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、
#     反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
#
#
#
# 【回答】
# 2駅先
#
# ============================================================
