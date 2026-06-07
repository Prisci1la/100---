'''
chapter6_utils.py: 第6章共用工具 / 第6章の共通ユーティリティ

集中管理第6章脚本共通使用的处理逻辑。 / 第6章スクリプトで共通利用する処理をまとめて管理する。
包括词向量模型加载、评价数据读取等功能。 / 単語ベクトルモデルの読み込みや評価データの読み込みなどを扱う。
'''

from __future__ import annotations  # 启用延迟类型注解 / 遅延型注釈を有効にする

import csv  # 导入CSV读取模块 / CSV読み込みモジュールを読み込む
import math  # 导入数学计算模块 / 数学計算モジュールを読み込む
from pathlib import Path  # 导入路径处理类 / パス処理クラスを読み込む


BASE_DIR = Path(__file__).resolve().parent  # 获取当前文件所在目录 / 現在のファイルがあるディレクトリを取得する
MODEL_PATH = BASE_DIR / "GoogleNews-vectors-negative300.bin"  # 词向量模型路径 / 単語ベクトルモデルのパス
QUESTIONS_PATH = BASE_DIR / "questions-words.txt"  # 类比问题数据路径 / アナロジー問題データのパス
ANALOGY_RESULT_PATH = BASE_DIR / "analogy_predictions.txt"  # 54题输出文件路径 / 54問の出力ファイルパス
WORDSIM_PATH = BASE_DIR / "wordsim353" / "combined.csv"  # WordSimilarity-353数据路径 / WordSimilarity-353データのパス


def load_vectors():  # 读取Google News词向量模型 / Google News単語ベクトルモデルを読み込む
    try:  # 尝试导入gensim模型类 / gensimのモデルクラスを読み込む
        from gensim.models import KeyedVectors  # 导入KeyedVectors / KeyedVectorsをインポートする
    except ImportError as exc:  # 如果未安装gensim则进入异常 / gensim未導入なら例外処理へ進む
        raise RuntimeError(  # 抛出更易懂的错误信息 / より分かりやすいエラーを送出する
            "gensim is required. Run: python -m pip install gensim numpy scipy"  # 提示安装依赖 / 依存関係の導入を案内する
        ) from exc  # 保留原始异常链 / 元の例外情報を保持する

    if not MODEL_PATH.exists():  # 检查模型文件是否存在 / モデルファイルの存在を確認する
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")  # 文件不存在则报错 / ファイルが無ければエラーにする

    return KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=True)  # 以binary格式加载模型 / binary形式でモデルを読み込む


def cosine_similarity(vec_a, vec_b):  # 计算两个向量的余弦相似度 / 2つのベクトルのコサイン類似度を計算する
    dot = float(vec_a @ vec_b)  # 计算点积 / 内積を計算する
    norm_a = math.sqrt(float(vec_a @ vec_a))  # 计算第一个向量的范数 / 1つ目のベクトルのノルムを計算する
    norm_b = math.sqrt(float(vec_b @ vec_b))  # 计算第二个向量的范数 / 2つ目のベクトルのノルムを計算する
    return dot / (norm_a * norm_b)  # 返回余弦相似度 / コサイン類似度を返す


def iter_question_sections(path=QUESTIONS_PATH):  # 逐行读取类比问题数据 / アナロジー問題データを1行ずつ読む
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


def build_country_names():  # 从首都类比题中提取国家名称 / 首都アナロジー問題から国名を抽出する
    countries = set()  # 使用集合去重 / 集合で重複を除く
    for section, words in iter_question_sections():  # 遍历所有类比题 / すべてのアナロジー問題を走査する
        if not section.startswith("capital-") or len(words) != 4:  # 只保留首都相关且格式正确的题目 / 首都関連で形式が正しい問題だけを残す
            continue  # 不符合条件就跳过 / 条件外なら読み飛ばす
        _capital_a, country_a, _capital_b, country_b = words  # 按“首都 国家 首都 国家”拆分 / 「首都 国名 首都 国名」の形で分解する
        countries.add(country_a)  # 加入第一个国家名 / 1つ目の国名を追加する
        countries.add(country_b)  # 加入第二个国家名 / 2つ目の国名を追加する

    return sorted(countries)  # 返回排序后的国家名列表 / 並べ替えた国名リストを返す


def load_country_names():  # 获取国家名列表 / 国名リストを取得する
    return build_country_names()  # 直接根据题目数据动态生成 / 問題データから動的に生成する


def load_country_vectors(model):  # 读取模型中存在的国家向量 / モデル内に存在する国名ベクトルを取り出す
    names = [name for name in load_country_names() if name in model]  # 过滤掉模型中不存在的国家名 / モデルに無い国名を除外する
    vectors = [model[name] for name in names]  # 取出对应国家的向量 / 対応する国名ベクトルを取り出す
    return names, vectors  # 返回国家名和向量列表 / 国名とベクトル一覧を返す


def semantic_section(section):  # 判断某一节是否属于语义类比 / ある節が意味的アナロジーか判定する
    return not section.startswith("gram")  # gram开头的是文法类比 / gramで始まる節は文法アナロジーである


def spearman_correlation(xs, ys):  # 计算Spearman秩相关系数 / Spearman順位相関係数を計算する
    def average_ranks(values):  # 计算并列值的平均排名 / 同順位を考慮した平均順位を計算する
        indexed = sorted(enumerate(values), key=lambda item: item[1])  # 按值排序并保留原索引 / 値で並べ替えつつ元の添字を保持する
        ranks = [0.0] * len(values)  # 初始化排名数组 / 順位配列を初期化する
        i = 0  # 初始化起始位置 / 開始位置を初期化する
        while i < len(indexed):  # 处理每一组相同的值 / 同じ値のまとまりごとに処理する
            j = i + 1  # 从下一位开始查找 / 次の位置から探索を始める
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:  # 找到所有并列元素 / 同じ値が続く範囲を見つける
                j += 1  # 扩展并列区间 / 同順位区間を広げる
            rank = (i + 1 + j) / 2  # 计算平均排名 / 平均順位を計算する
            for k in range(i, j):  # 给并列元素赋相同平均排名 / 同順位要素へ同じ平均順位を割り当てる
                ranks[indexed[k][0]] = rank  # 写回对应原索引位置 / 元の添字位置へ書き戻す
            i = j  # 继续处理下一组值 / 次の値のグループへ進む
        return ranks  # 返回排名数组 / 順位配列を返す

    rx = average_ranks(xs)  # 计算第一组数据的排名 / 1つ目のデータの順位を計算する
    ry = average_ranks(ys)  # 计算第二组数据的排名 / 2つ目のデータの順位を計算する
    mean_x = sum(rx) / len(rx)  # 计算第一组排名均值 / 1つ目の順位平均を計算する
    mean_y = sum(ry) / len(ry)  # 计算第二组排名均值 / 2つ目の順位平均を計算する
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))  # 计算协方差分子 / 共分散の分子を計算する
    var_x = sum((x - mean_x) ** 2 for x in rx)  # 计算第一组方差项 / 1つ目の分散項を計算する
    var_y = sum((y - mean_y) ** 2 for y in ry)  # 计算第二组方差项 / 2つ目の分散項を計算する
    return cov / math.sqrt(var_x * var_y)  # 返回Spearman相关系数 / Spearman相関係数を返す


def iter_wordsim353(path=WORDSIM_PATH):  # 逐行读取WordSimilarity-353数据 / WordSimilarity-353データを1行ずつ読む
    with path.open(encoding="utf-8") as file:  # 以UTF-8打开CSV文件 / UTF-8でCSVファイルを開く
        reader = csv.DictReader(file)  # 创建按列名读取的CSV解析器 / 列名ベースのCSV読み取り器を作る
        for row in reader:  # 遍历每一行评分数据 / 各行の評価データを順に処理する
            yield row["Word 1"], row["Word 2"], float(row["Human (mean)"])  # 返回两个单词和人工平均分 / 2語と人手平均スコアを返す
