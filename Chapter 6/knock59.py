'''
knock59.py: t-SNE可视化 / t-SNEによる可視化

将国家名词向量降到二维空间。 / 国名ベクトルを2次元空間に圧縮する。
并把结果保存为散点图图片。 / その結果を散布図画像として保存する。
'''

from chapter6_utils import BASE_DIR, load_country_vectors, load_vectors  # 导入输出目录、国家向量加载函数和模型加载函数 / 出力先ディレクトリと国名ベクトル読み込み関数とモデル読み込み関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    try:  # 尝试导入绘图、数值和t-SNE库 / 描画と数値計算とt-SNE用ライブラリを読み込む
        import matplotlib  # 导入matplotlib主模块 / matplotlib本体をインポートする
        matplotlib.use("Agg")  # 使用无界面后端防止弹窗 / 画面不要のAggバックエンドを使う
        import matplotlib.pyplot as plt  # 导入绘图接口 / 描画インターフェースをインポートする
        import numpy as np  # 导入numpy / numpyをインポートする
        from sklearn.manifold import TSNE  # 导入t-SNE降维器 / t-SNE次元削減器をインポートする
    except ImportError as exc:  # 如果依赖缺失则进入异常 / 依存関係不足なら例外処理へ進む
        raise RuntimeError("scikit-learn and matplotlib are required. Run: python -m pip install scikit-learn matplotlib") from exc  # 抛出更明确的安装提示 / より明確な導入案内エラーを出す

    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    countries, vectors = load_country_vectors(model)  # 读取国家名列表和对应向量 / 国名一覧と対応ベクトルを読み込む
    if len(vectors) < 2:  # 如果向量数量不足以降维则报错 / ベクトル数が少なすぎて次元削減できなければエラーにする
        raise RuntimeError("Need at least two country vectors for t-SNE.")  # 抛出向量不足错误 / ベクトル不足エラーを送出する
    vectors = np.asarray(vectors)  # 把向量列表转成numpy数组 / ベクトル一覧をnumpy配列へ変換する

    points = TSNE(  # 构造t-SNE降维器并执行降维 / t-SNE次元削減器を構築して次元削減を実行する
        n_components=2,  # 压缩到二维空间 / 2次元空間へ圧縮する
        perplexity=min(30, max(5, len(countries) // 3)),  # 根据样本数量设置困惑度 / サンプル数に応じてperplexityを設定する
        random_state=0,  # 固定随机种子保证可复现 / 乱数シードを固定して再現可能にする
        init="pca",  # 使用PCA初始化 / PCAで初期化する
        learning_rate="auto",  # 让学习率自动选择 / 学習率を自動選択にする
    ).fit_transform(vectors)  # 对全部国家向量执行降维 / 全国家ベクトルに対して次元削減を実行する

    plt.figure(figsize=(12, 10))  # 创建画布并设置尺寸 / 描画キャンバスを作成してサイズを設定する
    plt.scatter(points[:, 0], points[:, 1])  # 绘制二维散点图 / 2次元散布図を描く
    for country, (x, y) in zip(countries, points):  # 遍历每个国家及其二维坐标 / 各国名とその2次元座標を順に処理する
        plt.annotate(country, (x, y), fontsize=8)  # 在对应点旁边标注国家名称 / 対応点のそばに国名ラベルを付ける
    plt.tight_layout()  # 自动调整布局避免重叠 / 重なりを避けるためにレイアウトを自動調整する

    output_path = BASE_DIR / "knock59_tsne.png"  # 设置输出图片路径 / 出力画像パスを設定する
    plt.savefig(output_path, dpi=200, bbox_inches="tight")  # 保存t-SNE图像 / t-SNE画像を保存する
    print(f"saved t-SNE plot: {output_path}")  # 输出保存路径 / 保存先パスを表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
saved t-SNE plot: C:/Users/Administrator/Desktop/100ノック/Chapter 6/knock59_tsne.png
'''
