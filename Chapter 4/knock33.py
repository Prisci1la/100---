"""
33. 係り受け解析 / 依存句法解析
文章 text に対して文節（bunsetsu）単位の簡易係り受けを作る。
对文章 text 进行简易的文节级依存分析。

注意 / 注意：
  Janome は依存句法解析をサポートしていないため、
  文節分割 + 「右の文節に係る」というヒューリスティック規則で近似する。
  本格的な依存解析には GiNZA / CaboCha が必要。
  Janome 不支持依存句法解析，因此用「文节分割 + 依赖右侧最近文节」的启发式规则近似。
  完整依存解析需要 GiNZA / CaboCha。
"""

# 导入 Janome 分词器 / Janome トークナイザーをインポート
from janome.tokenizer import Tokenizer
# 从 knock30 模块导入 load_text 函数 / knock30 モジュールから load_text 関数をインポート
from knock30 import load_text


def split_bunsetsu(text: str) -> list[list]:  # 将文本分割为文节 / テキストを文節に分割
    tokenizer = Tokenizer()
    bunsetsu_list: list[list] = []
    current: list = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(",")[0]  # 词性第一项 / 品詞の最初
        current.append(token)
        if pos in ("助詞", "助動詞", "記号"):  # 文节分界符 / 文節の区切り
            bunsetsu_list.append(current)
            current = []
    if current:
        bunsetsu_list.append(current)
    return bunsetsu_list


def bunsetsu_text(bunsetsu: list) -> str:  # 文节字符串化（去除符号）/ 文節を文字列化
    return "".join(t.surface for t in bunsetsu  # 连接非「符号」的 token / 記号以外のトークンを連結
                   if t.part_of_speech.split(",")[0] != "記号")


def extract_dependencies(text: str) -> list[tuple[str, str]]:  # 提取依存关系 / 依存関係を抽出
    bunsetsu_list = split_bunsetsu(text)
    deps: list[tuple[str, str]] = []
    for i in range(len(bunsetsu_list) - 1):
        child = bunsetsu_text(bunsetsu_list[i])  # 依存元の文節 / 依存来源
        head = bunsetsu_text(bunsetsu_list[i + 1])  # 依存先の文節 / 依存目标
        if child and head:
            deps.append((child, head))
    return deps
    return deps


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 读取文本内容 / テキスト内容を読み込む
    text = load_text()
    # 遍历提取的依存关系 / 抽出した依存関係をループ
    for child, head in extract_dependencies(text):
        # 以制表符分隔打印依存来源和目标 / 係り元と係り先をタブ文字で区切って出力
        print(f"{child}\t{head}")
'''
政治が  わからぬ
村の    牧人で
牧人で  ある
笛を    吹き
吹き    羊と
羊と    遊んで
遊んで  暮して
暮して  来た
けれども邪悪に対して    は
人一倍に        敏感で
敏感で  あっ
あっ    た
'''