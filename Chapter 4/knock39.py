"""
39. Zipfの法則 / Zipf 定律
コーパスにおける単語の出現頻度順位を横軸、その出現頻度を縦軸として、
両対数グラフをプロットする。
以语料库中单词出现频率的排名为横轴，出现频率为纵轴，
绘制双对数图。

注意 / 注意：
  matplotlib / Pillow が Python 3.15 で wheel 未提供のため、
  純 Python で SVG（ベクター画像）を生成する。
  matplotlib / Pillow 在 Python 3.15 没有预编译 wheel，
  改为用纯 Python 生成 SVG（矢量图）。
  SVG はブラウザ・画像ビューア・VSCode で開ける。
  SVG 可用浏览器 / 图像查看器 / VSCode 直接打开。
"""

import math
from collections import Counter
from pathlib import Path

from knock36 import tokenize_corpus


# プロット領域の設定 / 绘图区域设置
WIDTH = 800           # SVG 全幅 / SVG 总宽
HEIGHT = 600          # SVG 全高 / SVG 总高
MARGIN_LEFT = 80
MARGIN_RIGHT = 30
MARGIN_TOP = 50
MARGIN_BOTTOM = 60
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM


def _log_scale(value: float, vmin: float, vmax: float, pixels: int) -> float:
    """log10 を線形ピクセルにマップ / 将 log10 值映射为线性像素"""
    return (math.log10(value) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin)) * pixels


def _build_svg(ranks: list[int], freqs: list[int]) -> str:
    """両対数散布図を SVG 文字列として返す。/ 返回双对数散点图的 SVG 字符串。"""
    xmin, xmax = 1, max(ranks)
    ymin, ymax = 1, max(freqs)

    # 各点の SVG 座標を計算 / 计算每个点的 SVG 坐标
    circles: list[str] = []
    for r, f in zip(ranks, freqs):
        px = MARGIN_LEFT + _log_scale(r, xmin, xmax, PLOT_W)
        py = MARGIN_TOP + PLOT_H - _log_scale(f, ymin, ymax, PLOT_H)
        circles.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.5" fill="#1f77b4"/>')

    # 軸の目盛り（10 の冪） / 轴刻度（10 的幂）
    x_ticks: list[str] = []
    exp = 0
    while 10 ** exp <= xmax:
        tick = 10 ** exp
        if tick >= xmin:
            px = MARGIN_LEFT + _log_scale(tick, xmin, xmax, PLOT_W)
            x_ticks.append(f'<line x1="{px:.1f}" y1="{MARGIN_TOP}" x2="{px:.1f}" y2="{MARGIN_TOP + PLOT_H}" stroke="#e0e0e0"/>')
            x_ticks.append(f'<text x="{px:.1f}" y="{MARGIN_TOP + PLOT_H + 18}" text-anchor="middle" font-size="11" fill="#444">10^{exp}</text>')
        exp += 1

    y_ticks: list[str] = []
    exp = 0
    while 10 ** exp <= ymax:
        tick = 10 ** exp
        if tick >= ymin:
            py = MARGIN_TOP + PLOT_H - _log_scale(tick, ymin, ymax, PLOT_H)
            y_ticks.append(f'<line x1="{MARGIN_LEFT}" y1="{py:.1f}" x2="{MARGIN_LEFT + PLOT_W}" y2="{py:.1f}" stroke="#e0e0e0"/>')
            y_ticks.append(f'<text x="{MARGIN_LEFT - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#444">10^{exp}</text>')
        exp += 1

    # 軸の枠 / 坐标轴边框
    frame = (
        f'<rect x="{MARGIN_LEFT}" y="{MARGIN_TOP}" '
        f'width="{PLOT_W}" height="{PLOT_H}" '
        f'fill="white" stroke="#333" stroke-width="1"/>'
    )

    # 軸ラベル・タイトル / 轴标签和标题
    title = f'<text x="{WIDTH / 2}" y="25" text-anchor="middle" font-size="18" font-weight="bold">Zipf\'s Law / Zipfの法則</text>'
    xlabel = f'<text x="{MARGIN_LEFT + PLOT_W / 2}" y="{HEIGHT - 15}" text-anchor="middle" font-size="13">Rank (順位 / 排名)</text>'
    ylabel = (
        f'<text x="20" y="{MARGIN_TOP + PLOT_H / 2}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 20 {MARGIN_TOP + PLOT_H / 2})">Frequency (頻度 / 频率)</text>'
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        frame,
        *x_ticks,
        *y_ticks,
        *circles,
        title,
        xlabel,
        ylabel,
        "</svg>",
    ]
    return "\n".join(parts)


def plot_zipf(output: str = "knock39_zipf.svg") -> str:
    """両対数グラフを SVG として保存する。/ 将双对数图保存为 SVG。"""
    freq = Counter(tokenize_corpus())
    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = list(range(1, len(sorted_freq) + 1))

    svg = _build_svg(ranks, sorted_freq)
    out_path = Path(__file__).parent / output
    out_path.write_text(svg, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = plot_zipf()
    print(f"图已保存到 / グラフを保存しました: {path}")
