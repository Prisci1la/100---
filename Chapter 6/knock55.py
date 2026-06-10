'''
knock55.py: 类比任务的正确率 / アナロジータスクでの正解率

重新评估完整的 analogy 数据集。 / analogy データセット全体を再評価する。
分别输出语义类比和文法类比的正确率。 / 意味的アナロジーと文法的アナロジーの正解率をそれぞれ表示する。
'''

from pathlib import Path  # 导入路径处理类 / パス処理クラスを読み込む
from chapter6_utils import load_vectors  # 导入模型加载函数 / モデル読み込み関数を導入する


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


def semantic_section(section):  # 判断某一节是否属于语义类比 / ある節が意味的アナロジーか判定する
    return not section.startswith("gram")  # gram开头的是文法类比 / gramで始まる節は文法アナロジーである


def evaluate_all_sections():  # 评估全部类比节的正确率 / 全アナロジー節の正解率を評価する
    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    counts = {  # 初始化语义类和文法类统计字典 / 意味系と文法系の集計辞書を初期化する
        "semantic": {"total": 0, "correct": 0},  # 准备语义类统计框架 / 意味系の集計枠を用意する
        "syntactic": {"total": 0, "correct": 0},  # 准备文法类统计框架 / 文法系の集計枠を用意する
    }

    questions_path = Path(__file__).resolve().parent / "questions-words.txt"  # 获取问题文件路径 / 問題ファイルパスを取得する
    _overall_score, sections = model.evaluate_word_analogies(str(questions_path))  # 用gensim评估整个问题文件 / gensimで問題ファイル全体を評価する
    for section_data in sections:  # 遍历每一个节的评估结果 / 各節の評価結果を順に処理する
        section_name = section_data["section"]  # 取出当前节名称 / 現在の節名を取り出す
        key = "semantic" if semantic_section(section_name) else "syntactic"  # 判断该节属于语义还是文法 / その節が意味系か文法系か判定する
        counts[key]["correct"] += len(section_data["correct"])  # 累加当前类别的正确数 / 現在の分類の正解数を加算する
        counts[key]["total"] += len(section_data["correct"]) + len(section_data["incorrect"])  # 累加当前类别总题数 / 現在の分類の総問題数を加算する

    return counts  # 返回完整统计结果 / 完全な集計結果を返す


def main():  # 定义主函数 / メイン関数を定義する
    counts = evaluate_all_sections()  # 获取全部节的正确率统计 / 全節の正解率集計を取得する

    for key in ["semantic", "syntactic"]:  # 按语义类和文法类依次输出 / 意味系と文法系を順に出力する
        total = counts[key]["total"]  # 取出该类别总题数 / その分類の総問題数を取り出す
        correct = counts[key]["correct"]  # 取出该类别正确数 / その分類の正解数を取り出す
        accuracy = correct / total if total else 0.0  # 计算该类别正确率 / その分類の正解率を計算する
        print(f"{key}: {correct}/{total} = {accuracy:.4f}")  # 输出当前类别的正确率 / 現在の分類の正解率を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
semantic: 20725/27985 = 0.7406
syntactic: 7889/10675 = 0.7390
'''
