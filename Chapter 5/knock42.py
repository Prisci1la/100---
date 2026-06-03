"""
knock42.py: 多选题正确率 / 多肢選択問題の正解率
使用JMMLU数据集中某一科目的多选题。 / JMMLUデータセットの特定科目の多肢選択問題を使用します。
计算模型的正确率。 / モデルの正解率を計算します。
"""

import os  # 导入标准库的操作系统模块 / 標準ライブラリのOSモジュールをインポート

from jmmlu_utils import extract_choice, format_jmmlu_prompt, load_jmmlu_questions  # 导入自定义JMMLU工具函数 / 自作のJMMLUユーティリティ関数をインポート
from openai_config import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, create_openai_client  # 导入配置常量和OpenAI客户端创建函数 / 設定定数とOpenAIクライアント作成関数をインポート


SUBJECT = os.environ.get("JMMLU_SUBJECT", "japanese_history")  # 从环境变量读取JMMLU科目，默认为日本历史 / 環境変数からJMMLU科目を読み取り、既定値は日本史
LIMIT = int(os.environ.get("JMMLU_LIMIT", "10"))  # 从环境变量读取问题数量限制，默认10题，0表示不限制 / 環境変数から問題数上限を読み取り、既定値は10問、0は無制限


def answer_question(client, item):  # 回答一个JMMLU多选题 / JMMLUの多肢選択問題に1問回答する

    message = client.chat.completions.create(  # 调用API生成对问题的回答 / APIを呼び出して問題への回答を生成
        model=DEFAULT_MODEL,  # 指定模型 / モデルを指定
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,  # 设置最大token数 / 最大トークン数を設定
        reasoning_effort="minimal",  # 设置推理难度为最小 / 推論の負荷を最小に設定
        verbosity="low",  # 设置详细度为低 / 出力の詳細度を低く設定
        messages=[  # 构建消息列表 / メッセージリストを構築
            {
                "role": "user",  # 消息角色为用户 / メッセージの役割をユーザーに設定
                "content": format_jmmlu_prompt(item),  # 使用JMMLU工具函数格式化问题 / JMMLUユーティリティ関数で問題を整形
            }
        ],
    )

    response = message.choices[0].message.content or ""  # 从API响应中提取回答文本，如果为空则设为空字符串 / API応答から回答文を取り出し、空なら空文字列にする
    return response, extract_choice(response)  # 从回答中提取选择项（A、B、C、D） / 回答から選択肢（A、B、C、D）を抽出


def calculate_accuracy():  # 计算单个JMMLU科目的正确率 / 単一JMMLU科目の正解率を計算する

    client = create_openai_client()  # 初始化OpenAI客户端 / OpenAIクライアントを初期化
    limit = LIMIT if LIMIT > 0 else None  # 根据LIMIT值确定问题限制数，0表示None（无限制） / LIMIT値から問題数上限を決め、0はNone（無制限）にする
    questions = load_jmmlu_questions(subject=SUBJECT, limit=limit)  # 从JMMLU加载问题 / JMMLUから問題を読み込む

    correct_count = 0  # 初始化正确计数器 / 正解数カウンターを初期化

    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print("knock42: JMMLU 多肢選択問題の正解率")  # 输出任务标题 / タスクのタイトルを出力
    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print(f"科目: {SUBJECT}")  # 输出科目信息 / 科目情報を出力
    print(f"問題数: {len(questions)}")  # 输出问题数 / 問題数を出力
    if questions:
        print(f"データ: {questions[0]['source']}")  # 如果有问题，输出数据源 / 問題があればデータソースを出力

    for i, item in enumerate(questions, 1):  # 遍历每个问题 / 各問題を順に処理
        response, prediction = answer_question(client, item)  # 回答问题并获取回答文本和提取的选择项 / 問題に回答し、回答文と抽出した選択肢を取得
        is_correct = prediction == item["answer"]  # 检查预测是否与正确答案一致 / 予測が正解と一致するか確認
        correct_count += int(is_correct)  # 统计正确答案数 / 正解数を加算

        print(f"\n【問題{i}】")  # 输出问题编号 / 問題番号を出力
        print(f"予測: {prediction or '抽出失敗'}")  # 输出模型预测，如果提取失败则显示"抽出失敗" / モデル予測を出力し、抽出失敗時は「抽出失敗」と表示
        print(f"正解: {item['answer']}")  # 输出正确答案 / 正解を出力
        print(f"判定: {'正解' if is_correct else '不正解'}")  # 输出判定结果 / 判定結果を出力
        if not prediction:
            print(f"回答全文: {response}")  # 如果没有成功提取选择项，输出完整回答用于调试 / 選択肢を抽出できない場合はデバッグ用に回答全文を出力

    accuracy = (correct_count / len(questions)) * 100 if questions else 0.0  # 计算正确率（百分比），如果没有问题则为0 / 正解率（百分率）を計算し、問題がなければ0にする

    print("\n" + "=" * 60)  # 输出分割线 / 区切り線を出力
    print(f"正解率: {correct_count}/{len(questions)} = {accuracy:.1f}%")  # 输出正确率统计 / 正解率の統計を出力
    print("=" * 60)  # 输出分割线 / 区切り線を出力

    return accuracy  # 返回计算的正确率 / 計算した正解率を返す


if __name__ == "__main__":
    calculate_accuracy()

"""
执行结果 / 実行結果:
============================================================
knock42: JMMLU 多肢選択問題の正解率
============================================================
科目: japanese_history
問題数: 10
データ: https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/japanese_history.csv

【問題1】
予測: B
正解: B
判定: 正解

【問題2】
予測: B
正解: B
判定: 正解

【問題3】
予測: D
正解: D
判定: 正解

【問題4】
予測: D
正解: D
判定: 正解

【問題5】
予測: A
正解: A
判定: 正解

【問題6】
予測: A
正解: A
判定: 正解

【問題7】
予測: D
正解: D
判定: 正解

【問題8】
予測: A
正解: A
判定: 正解

【問題9】
予測: C
正解: C
判定: 正解

【問題10】
予測: A
正解: D
判定: 不正解

============================================================
正解率: 9/10 = 90.0%
============================================================
"""
