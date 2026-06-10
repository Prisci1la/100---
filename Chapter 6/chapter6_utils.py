'''
chapter6_utils.py: 第6章共用工具 / 第6章の共通ユーティリティ

集中管理第6章脚本共通使用的处理逻辑。 / 第6章スクリプトで共通利用する処理をまとめて管理する。
包括词向量模型加载、评价数据读取等功能。 / 単語ベクトルモデルの読み込みや評価データの読み込みなどを扱う。
'''

from __future__ import annotations  # 启用延迟类型注解 / 遅延型注釈を有効にする

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
