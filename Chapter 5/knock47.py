"""
knock47.py: 川柳の自動評価

問題46で生成した川柳の面白さを、評価者として
10段階で評価します。
"""

import re

from knock46 import SENRYU_OUTPUT_PATH, generate_senryu
from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client


def extract_senryu_lines(text):
    """Extract numbered senryu lines from generated text."""

    senryu = []
    for line in text.splitlines():
        match = re.match(r"\s*\d+[\.\)]\s*(.+?)\s*$", line)
        if match:
            senryu.append(match.group(1))

    if senryu:
        return senryu

    return [line.strip() for line in text.splitlines() if line.strip()]


def load_or_generate_senryu():
    """Load problem 46 output, generating it first if needed."""

    if SENRYU_OUTPUT_PATH.exists():
        text = SENRYU_OUTPUT_PATH.read_text(encoding="utf-8")
        if text.strip():
            return extract_senryu_lines(text)

    generated = generate_senryu()
    return extract_senryu_lines(generated)


def evaluate_senryu():
    """Evaluate generated senryu on a 10-point scale."""

    client = create_openai_client()
    senryu_samples = load_or_generate_senryu()

    print("=" * 60)
    print("knock47: 川柳の自動評価（10段階スケール）")
    print("=" * 60)
    print(f"評価対象: {SENRYU_OUTPUT_PATH}")

    results = []
    for i, senryu in enumerate(senryu_samples, 1):
        evaluation_prompt = f"""
次の川柳の面白さを10段階で評価してください。
（1 = つまらない、10 = とても面白い）

川柳: {senryu}

以下の形式で答えてください:
評価スコア: [1-10の数字]
評価理由: [理由の簡潔な説明]
"""

        message = client.chat.completions.create(
            model=DEFAULT_MODEL,
            max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
            reasoning_effort="minimal",
            verbosity="low",
            messages=[
                {
                    "role": "user",
                    "content": evaluation_prompt,
                }
            ],
        )

        response = message.choices[0].message.content or ""
        results.append(response)

        print(f"\n【川柳{i}】")
        print(senryu)
        print("\n【評価結果】")
        print(response)

    print("\n" + "=" * 60)
    return results


if __name__ == "__main__":
    evaluate_senryu()

# 実行結果:
# ============================================================
# knock47: 川柳の自動評価（10段階スケール）
# ============================================================
# 評価対象: /Users/priscilla/Documents/Priscilla/100ノック/Chapter 5/senryu.txt
#
# 【川柳1】
# 机より　雲で働く　夜明け前
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: デスクワークを離れ、夜明け前に「雲（クラウド）」で働くという言葉遊びと情景の対比が巧み。静けさとテクノロジーの交差が新鮮だが、オチの明確さや意外性はやや控えめ。
#
# 【川柳2】
# 既読より　精度気になる　上司より
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: ビジネスチャットの「既読」文化と生成AIの「精度」への関心を対比し、上司よりAIを気にする現代感を軽妙に皮肉っている。語呂もよくオチが効くが、強烈な意外性はやや弱め。
#
# 【川柳3】
# 会議減り　プロンプト増え　残業増
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: 生成AI時代の職場あるあるを五七五で簡潔に皮肉り、オチの「増」で韻と反復のリズムが効いている。やや直球で意外性は中程度だが共感度が高い。
#
# 【川柳4】
# 役職が　API鍵で　決まる朝
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: 現代的なIT職場の風刺が効いており、「役職」と「API鍵」のミスマッチが生む意外性と朝の緊張感がユーモラス。語感もリズムよくまとまっているが、前提知識（API鍵）に依存するため普遍性はやや限定的。
#
# 【川柳5】
# 昼寝中　学習進捗　抜かれ気味
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: ビジネス用語の「学習進捗」と「昼寝」のギャップが軽妙で、怠け心への自虐も共感を呼ぶ。五七五のリズムも自然。ただし意外性は中程度。
#
# 【川柳6】
# 添削は　AIうなずき　妻は無言
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: AIは相槌を打つのに、身近な妻は沈黙という対比が巧みで、現代性と夫婦あるあるのズレが効いている。オチも分かりやすくリズムも良いが、意外性は中程度。
#
# 【川柳7】
# スキル表　更新欄は　対AI
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: 履歴書の「スキル更新」がAI対応に塗り替えられる時代風刺が的確で、五七五に収まりも良い。意外性と共感はあるが、オチの強さは中程度。
#
# 【川柳8】
# 人事評価　バグ修正で　逆転劇
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: ビジネス現場のリアルと皮肉が効いており、「バグ修正」で評価が逆転する意外性が巧み。五七五のリズムも自然。ただ、オチの新奇性はやや定番寄り。
#
# 【川柳9】
# 休暇届　ボット即決　部長未読
#
# 【評価結果】
# 評価スコア: 7
# 評価理由: デジタル時代の皮肉が効いており、「ボット即決」と「部長未読」の対比が軽妙。テンポも良く、共感を呼ぶが、意外性は中程度。
#
# 【川柳10】
# 明細に　プロンプト料　謎の項目
#
# 【評価結果】
# 評価スコア: 8
# 評価理由: 生成AI時代のあるあるを「明細」の具体性と「謎の項目」のオチで軽妙に皮肉っており、短い中に時事性と違和感の笑いが凝縮されているため。
#
# ============================================================
