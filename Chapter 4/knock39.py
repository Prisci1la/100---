"""
39. Zipfの法則 / Zipf 定律
コーパスにおける単語の出現頻度順位を横軸、その出現頻度を縦軸として、
両対数グラフをプロットする。
以语料库中单词出现频率的排名为横轴，出现频率为纵轴，
绘制双对数图。
"""

# 导入 Counter 用于统计频率 / Counter をインポートして頻度を数える
from collections import Counter
# 导入 matplotlib 用于绘图 / matplotlib をインポートして描画する
import matplotlib.pyplot as plt
# 导入 Path 用于文件处理 / Path をインポートしてファイル処理
from pathlib import Path

# 设置字体支持中文和日文 / 中国語と日本語を表示できるフォントを設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'SimHei', 'DejaVu Sans']
# 禁用减号符号的特殊处理 / マイナス記号の特殊処理を無効化
plt.rcParams['axes.unicode_minus'] = False

# 从 knock36 模块导入分词函数 / knock36 モジュールから分かち書き関数をインポート
from knock36 import tokenize_corpus


def plot_zipf(output: str = "knock39_zipf.png") -> str:  # 绘制Zipf定律图 / Zipfの法則グラフ
    freq = Counter(tokenize_corpus())
    sorted_freq = sorted(freq.values(), reverse=True)  # 降序排序 / 降順
    ranks = range(1, len(sorted_freq) + 1)
    plt.figure(figsize=(12, 8))
    plt.loglog(ranks, sorted_freq, marker=".", linestyle="none", markersize=3)  # 对数图 / 対数グラフ
    plt.xlabel("Rank (排名 / 順位)", fontsize=12)
    plt.ylabel("Frequency (频率 / 頻度)", fontsize=12)
    plt.title("Zipf's Law / Zipfの法則", fontsize=14, fontweight="bold")
    plt.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.5)  # 网格线 / グリッド
    out_path = Path(__file__).parent / output
    plt.savefig(out_path, dpi=300, bbox_inches="tight")  # 保存300DPI / 300DPI保存
    plt.close()
    return str(out_path)


# 程序入口 / プログラムエントリポイント
if __name__ == "__main__":
    # 绘制并保存 Zipf 定律图 / Zipf の法則のグラフを描画して保存
    path = plot_zipf()
    # 打印保存消息 / 保存メッセージを出力
    print(f"图已保存到 / グラフを保存しました: {path}")
