"""
knock43.py: 应答偏差 / 応答のバイアス
在问题42的JMMLU评估基础上改变实验设置，观察正确率是否变化。 / 問題42のJMMLU評価をもとに実験設定を変え、正解率の変化を観察します。
比较原始选择项配置与将正答项都移至D的配置。 / 元の選択肢配置と正答をすべてDへ移した配置を比較します。
"""

import os  # 导入操作系统模块 / OSモジュールをインポート

from jmmlu_utils import (  # 导入JMMLU工具函数 / JMMLUユーティリティ関数をインポート
    extract_choice,  # 导入从回答中提取选择项的函数 / 回答から選択肢を抽出する関数をインポート
    format_jmmlu_prompt,  # 导入格式化JMMLU提示词的函数 / JMMLUプロンプトを整形する関数をインポート
    load_jmmlu_questions,  # 导入加载JMMLU问题的函数 / JMMLU問題を読み込む関数をインポート
    move_correct_choice_to_d,  # 导入将正确答案移至D的函数 / 正解をDに移す関数をインポート
)
from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client  # 导入配置常量和OpenAI客户端 / 設定定数とOpenAIクライアントをインポート


SUBJECT = os.environ.get("JMMLU_SUBJECT", "japanese_history")  # 从环境变量读取JMMLU科目，默认为日本历史 / 環境変数からJMMLU科目を読み取り、既定値は日本史
LIMIT = int(os.environ.get("JMMLU_LIMIT", "10"))  # 从环境变量读取问题数量限制，默认10题 / 環境変数から問題数上限を読み取り、既定値は10問


def answer_question(client, item):  # 回答一个JMMLU多选题 / JMMLUの多肢選択問題に1問回答する

    message = client.chat.completions.create(  # 调用API生成回答 / APIを呼び出して回答を生成
        model=DEFAULT_MODEL,  # 指定模型 / モデルを指定
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,  # 设置最大token数 / 最大トークン数を設定
        reasoning_effort="minimal",  # 设置推理难度为最小 / 推論の負荷を最小に設定
        verbosity="low",  # 设置详细度为低 / 出力の詳細度を低く設定
        messages=[  # 构建消息列表 / メッセージリストを構築
            {
                "role": "user",  # 消息角色为用户 / メッセージの役割をユーザーに設定
                "content": format_jmmlu_prompt(item),  # 格式化问题提示 / 問題プロンプトを整形
            }
        ],
    )

    response = message.choices[0].message.content or ""  # 从API响应中提取回答文本，如果为空则设为空字符串 / API応答から回答文を取り出し、空なら空文字列にする
    return extract_choice(response)  # 从回答中提取选择项（A、B、C、D） / 回答から選択肢（A、B、C、D）を抽出


def evaluate_setting(client, questions, name, transform):  # 评估一个实验设置 / 1つの実験設定を評価する

    correct_count = 0  # 初始化正确计数器 / 正解数カウンターを初期化
    predictions = []  # 初始化预测列表 / 予測リストを初期化

    for item in questions:  # 遍历每个问题 / 各問題を順に処理
        transformed = transform(item)  # 对问题应用转换函数（如移动正答项） / 問題に変換関数を適用（例: 正答の移動）
        prediction = answer_question(client, transformed)  # 回答转换后的问题 / 変換後の問題に回答
        predictions.append(prediction or "抽出失敗")  # 将预测添加到列表（如果为空则显示"抽出失敗"） / 予測をリストに追加し、空なら「抽出失敗」と表示
        correct_count += int(prediction == transformed["answer"])  # 统计正确答案数（比较预测与转换后问题的正确答案） / 変換後の正解と予測を比べて正解数を加算

    accuracy = (correct_count / len(questions)) * 100 if questions else 0.0  # 计算正确率（百分比），如果没有问题则为0 / 正解率（百分率）を計算し、問題がなければ0にする
    return {  # 返回结果字典 / 結果の辞書を返す
        "name": name,  # 设置名称 / 設定名
        "correct_count": correct_count,  # 正确答案数 / 正解数
        "accuracy": accuracy,  # 正确率百分比 / 正解率の百分率
        "predictions": predictions,  # 所有预测列表 / すべての予測リスト
    }


def investigate_response_bias():  # 比较不同JMMLU实验设置下的正确率 / 異なるJMMLU実験設定の正解率を比較する

    client = create_openai_client()  # 初始化OpenAI客户端 / OpenAIクライアントを初期化
    limit = LIMIT if LIMIT > 0 else None  # 根据LIMIT值确定问题限制数，0表示None（无限制） / LIMIT値から問題数上限を決め、0はNone（無制限）にする
    questions = load_jmmlu_questions(subject=SUBJECT, limit=limit)  # 从JMMLU加载问题 / JMMLUから問題を読み込む

    settings = [  # 定义实验设置列表 / 実験設定リストを定義
        ("元の選択肢配置", lambda item: item),  # 第一个设置：原始选择项配置（使用lambda返回原问题） / 1つ目の設定: 元の選択肢配置（lambdaで元の問題を返す）
        ("正解をすべてDに移動", move_correct_choice_to_d),  # 第二个设置：将正答项移至D选项 / 2つ目の設定: 正答をD選択肢へ移す
    ]

    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print("knock43: JMMLU 応答のバイアス調査")  # 输出任务标题 / タスクのタイトルを出力
    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print(f"科目: {SUBJECT}")  # 输出科目信息 / 科目情報を出力
    print(f"問題数: {len(questions)}")  # 输出问题数 / 問題数を出力

    results = []  # 初始化结果列表 / 結果リストを初期化
    for name, transform in settings:  # 遍历每个实验设置 / 各実験設定を順に処理
        result = evaluate_setting(client, questions, name, transform)  # 评估当前设置 / 現在の設定を評価
        results.append(result)  # 将结果添加到列表 / 結果をリストへ追加

        print(f"\n【{name}】")  # 输出设置名称 / 設定名を出力
        print(f"正解数: {result['correct_count']}/{len(questions)}")  # 输出正确答案数统计 / 正解数の統計を出力
        print(f"正解率: {result['accuracy']:.1f}%")  # 输出正确率百分比 / 正解率の百分率を出力
        print(f"予測: {', '.join(result['predictions'])}")  # 输出所有预测结果 / すべての予測結果を出力

    print("\n" + "=" * 60)  # 输出分割线 / 区切り線を出力
    if len(results) >= 2:
        diff = results[1]["accuracy"] - results[0]["accuracy"]  # 如果至少有两个结果，计算第二个设置和第一个设置的正确率差异 / 結果が2つ以上あれば2つ目と1つ目の正解率差を計算
        print(f"正解率の差: {diff:+.1f}ポイント")  # 输出差异（:+表示显示正负号） / 差を出力（:+で符号を表示）
    print("=" * 60)  # 输出分割线 / 区切り線を出力

    return results  # 返回结果列表 / 結果リストを返す


if __name__ == "__main__":
    investigate_response_bias()

"""
执行结果 / 実行結果:
============================================================
knock43: JMMLU 応答のバイアス調査
============================================================
科目: japanese_history
問題数: 10

【元の選択肢配置】
正解数: 9/10
正解率: 90.0%
予測: B, B, D, D, A, A, D, A, C, A

【正解をすべてDに移動】
正解数: 9/10
正解率: 90.0%
予測: D, D, D, D, D, D, D, D, D, A

============================================================
正解率の差: +0.0ポイント
============================================================
"""
