"""
39. Zipfの法則 / Zipf 定律
コーパスにおける単語の出現頻度順位を横軸、その出現頻度を縦軸として、
両対数グラフをプロットする。
以语料库中单词出现频率的排名为横轴，出现频率为纵轴，
绘制双对数图。
"""

from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

from knock36 import tokenize_corpus


def plot_zipf(output: str = "knock39_zipf.png") -> str:
    """両対数グラフを保存する。/ 保存双对数图。"""
    freq = Counter(tokenize_corpus())
    # 降順にソート / 降序排序
    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = range(1, len(sorted_freq) + 1)

    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, sorted_freq, marker=".", linestyle="none", markersize=2)
    plt.xlabel("Rank (順位 / 排名)")
    plt.ylabel("Frequency (頻度 / 频率)")
    plt.title("Zipf's Law / Zipfの法則")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    out_path = Path(__file__).parent / output
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


if __name__ == "__main__":
    path = plot_zipf()
    print(f"图已保存到 / グラフを保存しました: {path}")
