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

from janome.tokenizer import Tokenizer
from knock30 import load_text


def split_bunsetsu(text: str) -> list[list]:
    """
    文を文節に分割する（粗い規則）。
    将句子分割为文节（粗略规则）。

    規則 / 规则：
      助詞・助動詞・記号で文節を区切る
      （助词 / 助动词 / 符号作为文节边界）
    """
    tokenizer = Tokenizer()
    bunsetsu_list: list[list] = []
    current: list = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(",")[0]
        current.append(token)
        if pos in ("助詞", "助動詞", "記号"):
            # 文区切り（。）なら文末まで含めて終了
            if token.surface in ("。", "！", "？"):
                bunsetsu_list.append(current)
                current = []
            else:
                bunsetsu_list.append(current)
                current = []
    if current:
        bunsetsu_list.append(current)
    return bunsetsu_list


def bunsetsu_text(bunsetsu: list) -> str:
    """文節を文字列化（記号を除く）/ 文节字符串化（去除符号）"""
    return "".join(
        t.surface for t in bunsetsu
        if t.part_of_speech.split(",")[0] != "記号"
    )


def extract_dependencies(text: str) -> list[tuple[str, str]]:
    """
    (係り元, 係り先) のリストを返す。最後の文節は ROOT として除外。
    返回 (依存来源, 依存目标) 列表，最后一个文节为 ROOT，排除。

    ヒューリスティック / 启发式：
      日本語は「左から右に係る」性質があるので、
      各文節は右隣の文節に係るとみなす。
      日语依赖关系是「从左到右」的，每个文节依赖于右侧最近的文节。
    """
    bunsetsu_list = split_bunsetsu(text)
    deps: list[tuple[str, str]] = []
    for i in range(len(bunsetsu_list) - 1):
        child = bunsetsu_text(bunsetsu_list[i])
        head = bunsetsu_text(bunsetsu_list[i + 1])
        if child and head:
            deps.append((child, head))
    return deps


if __name__ == "__main__":
    text = load_text()
    for child, head in extract_dependencies(text):
        print(f"{child}\t{head}")
'''
メロスは        激怒した
必ず    かの邪智暴虐の
かの邪智暴虐の  王を
王を    除かなけれ
除かなけれ      ば
ば      ならぬ
ならぬ  と
と      決意した
メロスに        は
は      政治が
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