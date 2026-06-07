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

# 导入 Path 用于文件操作 / Path をインポートしてファイル操作を実施
from pathlib import Path
# 从 knock33 模块导入文节分割和文本转换函数 / knock33 モジュールから文節分割と文字列変換関数をインポート
from knock33 import split_bunsetsu, bunsetsu_text


def visualize_dependency(sentence: str, output: str = "knock35_tree.txt") -> str:  # 保存依存关系为TXT文件 / 依存関係を描写して保存
    bunsetsu_list = split_bunsetsu(sentence)
    chunks = [bunsetsu_text(b) for b in bunsetsu_list if bunsetsu_text(b)]  # 文节不为空
    lines = [f"{chunks[i]} → {chunks[i + 1]}" for i in range(len(chunks) - 1)]  # 构建依存关系
    if chunks:
        lines.append(f"{chunks[-1]} → [ROOT]")  # 最后一个文节为ROOT / 最後は値
    output_text = "\n".join(lines)
    out_path = Path(__file__).parent / output
    out_path.write_text(output_text, encoding="utf-8")
    print(output_text)
    return str(out_path)


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 定义要处理的句子 / 処理対象の文を定義
    sentence = "メロスは激怒した。"
    # 可视化並保存依存树 / 依存木を可視化して保存
    path = visualize_dependency(sentence)
    # 打印保存消息 / 保存メッセージを出力
    print(f"\n依存树已保存到 / 依存木を保存しました: {path}")
'''
メロスは → 激怒した
激怒した → [ROOT]
'''