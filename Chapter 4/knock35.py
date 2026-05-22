"""
35. 係り受け木 / 依存树
「メロスは激怒した。」の係り受け木をテキスト形式で可視化する。
将"メロスは激怒した。"的依存树以文本形式可视化。

注意 / 注意：
  Janome は依存解析できないため、文節分割の簡易木で代用する。
  Janome 不支持依存解析，使用文节分割的简易树代替。
  graphviz が不要なため、テキスト出力に変更。
  不需要 graphviz，改为文本输出。
"""

from pathlib import Path
from knock33 import split_bunsetsu, bunsetsu_text


def visualize_dependency(sentence: str, output: str = "knock35_tree.txt") -> str:
    """
    係り受けをテキストファイルとして保存し、パスを返す。
    将依存关系保存为文本文件，返回路径。

    出力例 / 输出例：
      メロスは → 激怒した
    """
    bunsetsu_list = split_bunsetsu(sentence)
    chunks = [bunsetsu_text(b) for b in bunsetsu_list if bunsetsu_text(b)]

    lines: list[str] = []
    # 各文節は右隣に係る（最後を除く） / 每个文节依赖右侧（除最后一个）
    for i in range(len(chunks) - 1):
        lines.append(f"{chunks[i]} → {chunks[i + 1]}")
    # 最後の文節は ROOT / 最后一个文节为 ROOT
    if chunks:
        lines.append(f"{chunks[-1]} → [ROOT]")

    output_text = "\n".join(lines)
    out_path = Path(__file__).parent / output
    out_path.write_text(output_text, encoding="utf-8")

    print(output_text)  # 同時に画面にも表示 / 同时输出到屏幕
    return str(out_path)


if __name__ == "__main__":
    sentence = "メロスは激怒した。"
    path = visualize_dependency(sentence)
    print(f"\n依存树已保存到 / 依存木を保存しました: {path}")
