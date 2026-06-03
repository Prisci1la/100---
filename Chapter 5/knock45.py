"""
knock45.py: 多轮对话 / マルチターン対話

实现多轮对话。 / 複数ターンの対話を実装します。
在保留上一轮对话上下文的同时回答新问题。 / 前のターンの会話コンテキストを保持しながら、新しい質問に答えさせます。
"""

from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client  # 导入配置常量和OpenAI客户端创建函数 / 設定定数とOpenAIクライアント作成関数をインポート


def multi_turn_dialogue():  # 执行多轮对话 / マルチターン対話を実行する

    client = create_openai_client()  # 初始化OpenAI客户端 / OpenAIクライアントを初期化

    messages = []  # 保存多轮对话历史 / マルチターン対話の履歴を保持

    initial_question = (  # 第1轮：说明初始情况 / ターン1: 初期の状況を説明
        """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
    東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
    間違えて急行に乗車してしまったことに気付きました。
    自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
    つばめちゃんの目的地でした。

    目的地の駅の名前を答えてください。
    """
    )

    messages.append({  # 将第1轮用户问题加入对话历史 / ターン1のユーザー質問を対話履歴に追加
        "role": "user",  # 消息角色为用户 / メッセージの役割をユーザーに設定
        "content": initial_question,  # 消息内容为初始问题 / メッセージ内容を初期質問に設定
    })

    response1 = client.chat.completions.create(  # 获取第一次回答 / 最初の回答を取得
        model=DEFAULT_MODEL,  # 指定模型 / モデルを指定
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,  # 设置最大token数 / 最大トークン数を設定
        reasoning_effort="minimal",  # 设置推理难度为最小 / 推論の負荷を最小に設定
        verbosity="low",  # 设置详细度为低 / 出力の詳細度を低く設定
        messages=messages,  # 传入当前对话历史 / 現在の対話履歴を渡す
    )

    first_answer = response1.choices[0].message.content or ""  # 提取第1轮回答文本 / ターン1の回答文を抽出

    messages.append({  # 将第1轮模型回答加入对话历史 / ターン1のモデル回答を対話履歴に追加
        "role": "assistant",  # 消息角色为助手 / メッセージの役割をアシスタントに設定
        "content": first_answer,  # 消息内容为第1轮回答 / メッセージ内容をターン1の回答に設定
    })

    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print("knock45: マルチターン対話")  # 输出任务标题 / タスクのタイトルを出力
    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print("\n【ターン1: 初期質問】")  # 输出第1轮问题标签 / ターン1の質問ラベルを出力
    print(initial_question)  # 输出第1轮问题内容 / ターン1の質問内容を出力
    print("\n【回答】")  # 输出回答标签 / 回答ラベルを出力
    print(first_answer)  # 输出第1轮回答 / ターン1の回答を出力
    if not first_answer.strip():
        print("\n【ターン1 デバッグ】")  # 输出第1轮调试标签 / ターン1のデバッグラベルを出力
        print(f"finish_reason: {response1.choices[0].finish_reason}")  # 输出第1轮生成结束原因 / ターン1の生成終了理由を出力
        print(f"usage: {response1.usage}")  # 输出第1轮API使用信息 / ターン1のAPI使用量情報を出力

    follow_up_question = (  # 第2轮：用不同场景追加提问 / ターン2: 異なるシナリオで追加質問
        """
    さらに、つばめちゃんが自由が丘駅で乗り換えたとき、
    先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
    目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、
    反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？

    """
    )

    messages.append({  # 将第2轮用户问题加入对话历史 / ターン2のユーザー質問を対話履歴に追加
        "role": "user",  # 消息角色为用户 / メッセージの役割をユーザーに設定
        "content": follow_up_question,  # 消息内容为追加问题 / メッセージ内容を追加質問に設定
    })

    response2 = client.chat.completions.create(  # 获取第二次回答 / 2番目の回答を取得
        model=DEFAULT_MODEL,  # 指定模型 / モデルを指定
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,  # 设置最大token数 / 最大トークン数を設定
        reasoning_effort="minimal",  # 设置推理难度为最小 / 推論の負荷を最小に設定
        verbosity="low",  # 设置详细度为低 / 出力の詳細度を低く設定
        messages=messages,  # 传入包含前一轮上下文的对话历史 / 前ターンの文脈を含む対話履歴を渡す
    )

    second_answer = response2.choices[0].message.content or ""  # 提取第2轮回答文本 / ターン2の回答文を抽出

    print("\n【ターン2: フォローアップ質問】")  # 输出第2轮问题标签 / ターン2の質問ラベルを出力
    print(follow_up_question)  # 输出第2轮问题内容 / ターン2の質問内容を出力
    print("\n【回答】")  # 输出回答标签 / 回答ラベルを出力
    print(second_answer)  # 输出第2轮回答 / ターン2の回答を出力
    if not second_answer.strip():
        print("\n【ターン2 デバッグ】")  # 输出第2轮调试标签 / ターン2のデバッグラベルを出力
        print(f"finish_reason: {response2.choices[0].finish_reason}")  # 输出第2轮生成结束原因 / ターン2の生成終了理由を出力
        print(f"usage: {response2.usage}")  # 输出第2轮API使用信息 / ターン2のAPI使用量情報を出力
    print("\n" + "=" * 60)  # 输出分割线 / 区切り線を出力

    return {  # 返回两轮回答 / 2ターン分の回答を返す
        "first_answer": first_answer,  # 第1轮回答 / ターン1の回答
        "second_answer": second_answer,  # 第2轮回答 / ターン2の回答
    }


if __name__ == "__main__":
    multi_turn_dialogue()

"""
执行结果 / 実行結果:
============================================================
knock45: マルチターン対話
============================================================

【ターン1: 初期質問】

    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
    東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
    間違えて急行に乗車してしまったことに気付きました。
    自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
    つばめちゃんの目的地でした。

    目的地の駅の名前を答えてください。


【回答】
九品仏駅

【ターン2: フォローアップ質問】

    さらに、つばめちゃんが自由が丘駅で乗り換えたとき、
    先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
    目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、
    反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？



【回答】
2駅先

============================================================
"""
