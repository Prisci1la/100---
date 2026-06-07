"""
knock41.py: Few-Shot推論 (Few-Shot Reasoning)

Few-shot推論では、複数の例（example）を与えてから、
その例のパターンに従って新しい問題を解かせます。

このスクリプトでは、問題40の問題文に解答例を追加し、
few-shot推論で解きます。
"""

from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


def few_shot_reasoning():
    """Few-shot推論で歴史問題を解く（4-shot推論）"""

    client = create_openai_client()

    # Few-shot例を含むプロンプト
    prompt = """
    次のような問題と解答例を参考にして、最後の問題に答えてください。

    【例1】日本の近代化に関連するできごと
    ア　府知事・県令からなる地方官会議が設置された。
    イ　廃藩置県が実施され，中央から府知事・県令が派遣される体制になった。
    ウ　すべての藩主が，天皇に領地と領民を返還した。
    解答: ウ→イ→ア

    【例2】江戸幕府の北方での対外的な緊張
    ア　レザノフが長崎に来航したが，幕府が冷淡な対応をしたため，ロシア船が樺太や択捉島を攻撃した。
    イ　ゴローウニンが国後島に上陸し，幕府の役人に捕らえられ抑留された。
    ウ　ラクスマンが根室に来航し，漂流民を届けるとともに通商を求めた。
    解答: ウ→ア→イ

    【例3】中居屋重兵衛の生涯の期間におこったできごと
    ア　アヘン戦争がおこり，清がイギリスに敗北した。
    イ　異国船打払令が出され，外国船を撃退することが命じられた。
    ウ　桜田門外の変がおこり，大老の井伊直弼が暗殺された。
    解答: イ→ア→ウ

    【例4】加藤高明に関連するできごと
    ア　朝鮮半島において，独立を求める大衆運動である三・一独立運動が展開された。
    イ　関東大震災後の混乱のなかで，朝鮮人や中国人に対する殺傷事件がおきた。
    ウ　日本政府が，袁世凱政府に対して二十一カ条の要求を突き付けた。
    解答: ウ→ア→イ

    【問題】9世紀に活躍した人物に関係するできごと
    ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
    イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
    ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。

    上記の例と同じ形式で答えてください。
    解答のみを「ア→イ→ウ」のような形式で1行で出力してください。
    """

    # APIに送信
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

    # 結果を表示
    print("=" * 60)
    print("knock41: Few-Shot推論（4-shot推論）")
    print("=" * 60)
    print("\n【解答】")
    answer = message.choices[0].message.content or ""
    print(answer)
    if not answer.strip():
        print("\n【デバッグ】")
        print(f"finish_reason: {message.choices[0].finish_reason}")
        print(f"usage: {message.usage}")
    print("\n" + "=" * 60)

    return answer


if __name__ == "__main__":
    few_shot_reasoning()

# 実行結果:
# ============================================================
# knock41: Few-Shot推論（4-shot推論）
# ============================================================
#
# 【解答】
# イ→ア→ウ
#
# ============================================================
