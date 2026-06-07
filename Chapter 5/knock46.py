"""
knock46.py: 川柳の生成 (Senryu Generation)

川柳は日本の伝統的な短詩形で、5-7-5の音数構成です。
このスクリプトでは、適当なお題について
10個の川柳を生成します。
"""

from pathlib import Path

from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


SENRYU_OUTPUT_PATH = Path(__file__).with_name("senryu.txt")


def generate_senryu():
    """川柳を生成"""

    client = create_openai_client()

    prompt = """
    以下のお題について、川柳を10個作成してください。
    川柳は5-7-5の音数構成で、日本の伝統的な短詩形です。
    ユーモアや風刺を含めて、創意工夫のある作品をお願いします。

    【お題】「AI時代の仕事」

    形式:
    1. [川柳1]
    2. [川柳2]
    ...
    10. [川柳10]

    各川柳は、音数が正確に5-7-5になるようにしてください。
    """

    message = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        reasoning_effort="minimal",
        verbosity="low",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response = message.choices[0].message.content or ""
    SENRYU_OUTPUT_PATH.write_text(response, encoding="utf-8")

    print("=" * 60)
    print("knock46: 川柳の生成")
    print("=" * 60)
    print("\n【お題】AI時代の仕事")
    print("\n【生成された川柳】")
    print(response)
    print(f"\n保存先: {SENRYU_OUTPUT_PATH}")
    print("\n" + "=" * 60)

    return response


if __name__ == "__main__":
    generate_senryu()

# 実行結果:
# ============================================================
# knock46: 川柳の生成
# ============================================================
#
# 【お題】AI時代の仕事
#
# 【生成された川柳】
# 1. 机より　雲で働く　夜明け前
# 2. 既読より　精度気になる　上司より
# 3. 会議減り　プロンプト増え　残業増
# 4. 役職が　API鍵で　決まる朝
# 5. 昼寝中　学習進捗　抜かれ気味
# 6. 添削は　AIうなずき　妻は無言
# 7. スキル表　更新欄は　対AI
# 8. 人事評価　バグ修正で　逆転劇
# 9. 休暇届　ボット即決　部長未読
# 10. 明細に　プロンプト料　謎の項目
#
# 保存先: /Users/priscilla/Documents/Priscilla/100ノック/Chapter 5/senryu.txt
#
# ============================================================
