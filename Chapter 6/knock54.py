'''
knock54.py: 在类比数据上实验 / アナロジーデータでの実験

针对 questions-words.txt 中的 capital-common-countries 部分做实验。 / questions-words.txt の capital-common-countries 節で実験する。
记录每条样本的预测词和相似度。 / 各サンプルの予測語と類似度を記録する。
'''

from pathlib import Path  # 导入路径处理类 / パス処理クラスを読み込む
from chapter6_utils import ANALOGY_RESULT_PATH, load_vectors  # 导入输出路径和模型加载函数 / 出力パスとモデル読み込み関数を導入する


def iter_question_sections(path=None):  # 逐行读取类比问题数据 / アナロジー問題データを1行ずつ読む
    if path is None:
        path = Path(__file__).resolve().parent / "questions-words.txt"
    section = None  # 初始化当前节名称 / 現在の節名を初期化する
    with path.open(encoding="utf-8") as file:  # 以UTF-8打开问题文件 / UTF-8で問題ファイルを開く
        for line in file:  # 遍历文件每一行 / ファイルの各行を順に処理する
            line = line.strip()  # 去除首尾空白 / 行頭と行末の空白を除去する
            if not line:  # 如果是空行则跳过 / 空行なら読み飛ばす
                continue  # 继续下一行 / 次の行へ進む
            if line.startswith(":"):  # 如果是节标题行 / 節タイトル行かどうか判定する
                section = line[1:].strip()  # 记录当前节名称 / 現在の節名を記録する
                continue  # 标题行不作为题目返回 / タイトル行は問題として返さない
            yield section, line.split()  # 返回节名和该行单词列表 / 節名とその行の単語列を返す


def extract_capital_common_countries():  # 提取capital-common-countries部分 / capital-common-countries節を抽出する
    rows = []  # 准备保存目标题目行 / 対象問題行を保存するリストを用意する
    for section, words in iter_question_sections():  # 遍历所有节和题目 / すべての節と問題を順に処理する
        if section == "capital-common-countries":  # 只保留目标节 / 対象節だけを残す
            rows.append(words)  # 把题目加入结果列表 / 問題を結果リストへ追加する
        elif rows:  # 如果已经离开目标节 / すでに対象節を抜けたか確認する
            break  # 直接结束循环 / その時点でループを終了する
    return rows  # 返回该节全部题目 / その節の全問題を返す


def main():  # 定义主函数 / メイン関数を定義する
    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    rows = extract_capital_common_countries()  # 提取capital-common-countries部分的全部题目 / capital-common-countries節の全問題を抽出する

    with ANALOGY_RESULT_PATH.open("w", encoding="utf-8") as output:  # 以写入模式打开结果文件 / 書き込みモードで結果ファイルを開く
        for capital_a, country_a, capital_b, country_b in rows:  # 逐条处理类比题 / 各アナロジー問題を順に処理する
            predicted, similarity = model.most_similar(  # 计算最相近的预测结果 / 最も近い予測結果を計算する
                positive=[country_a, capital_b],  # 使用国家A和首都B作为正向向量 / 国Aと首都Bを正のベクトルとして使う
                negative=[capital_a],  # 使用首都A作为负向向量 / 首都Aを負のベクトルとして使う
                topn=1,  # 只保留最接近的1个结果 / 最も近い1件だけを残す
            )[0]
            output.write(  # 将原题、预测词和相似度写入文件 / 元問題と予測語と類似度をファイルへ書き込む
                f"capital-common-countries {capital_a} {country_a} "  # 写入题目前半部分 / 問題前半を書き込む
                f"{capital_b} {country_b} {predicted} {similarity:.6f}\n"  # 写入题目后半部分和预测结果 / 問題後半と予測結果を書き込む
            )

    print(f"wrote {len(rows)} rows to {ANALOGY_RESULT_PATH}")  # 输出写入行数和文件位置 / 書き込み行数と保存先を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
wrote 506 rows to C:/Users/Administrator/Desktop/100ノック/Chapter 6/analogy_predictions.txt
'''
