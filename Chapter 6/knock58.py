'''
knock58.py: Ward法聚类 / Ward法によるクラスタリング

对国家名词向量做Ward层次聚类。 / 国名ベクトルに対してWard法の階層クラスタリングを行う。
并将结果保存为树状图图片。 / さらに結果をデンドログラム画像として保存する。
'''

from chapter6_utils import BASE_DIR, load_country_vectors, load_vectors  # 导入输出目录、国家向量加载函数和模型加载函数 / 出力先ディレクトリと国名ベクトル読み込み関数とモデル読み込み関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    try:  # 尝试导入绘图、数值和层次聚类库 / 描画と数値計算と階層クラスタリング用ライブラリを読み込む
        import matplotlib  # 导入matplotlib主模块 / matplotlib本体をインポートする
        matplotlib.use("Agg")  # 使用无界面后端防止弹窗 / 画面不要のAggバックエンドを使う
        import matplotlib.pyplot as plt  # 导入绘图接口 / 描画インターフェースをインポートする
        import numpy as np  # 导入numpy / numpyをインポートする
        from scipy.cluster.hierarchy import dendrogram, linkage  # 导入树状图和linkage函数 / デンドログラム関数とlinkage関数をインポートする
    except ImportError as exc:  # 如果依赖缺失则进入异常 / 依存関係不足なら例外処理へ進む
        raise RuntimeError("scipy and matplotlib are required. Run: python -m pip install scipy matplotlib") from exc  # 抛出更明确的安装提示 / より明確な導入案内エラーを出す

    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    countries, vectors = load_country_vectors(model)  # 读取国家名列表和对应向量 / 国名一覧と対応ベクトルを読み込む
    if len(vectors) < 2:  # 如果向量数量不足以聚类则报错 / ベクトル数が少なすぎてクラスタリングできなければエラーにする
        raise RuntimeError("Need at least two country vectors for hierarchical clustering.")  # 抛出向量不足错误 / ベクトル不足エラーを送出する
    vectors = np.asarray(vectors)  # 把向量列表转成numpy数组 / ベクトル一覧をnumpy配列へ変換する

    linkage_matrix = linkage(vectors, method="ward")  # 用Ward法构建层次聚类矩阵 / Ward法で階層クラスタ行列を構築する

    plt.figure(figsize=(16, 8))  # 创建画布并设置尺寸 / 描画キャンバスを作成してサイズを設定する
    dendrogram(linkage_matrix, labels=countries, leaf_rotation=90)  # 绘制带国家标签的树状图 / 国名ラベル付きのデンドログラムを描く
    plt.tight_layout()  # 自动调整布局避免重叠 / 重なりを避けるためにレイアウトを自動調整する

    output_path = BASE_DIR / "knock58_dendrogram.png"  # 设置输出图片路径 / 出力画像パスを設定する
    plt.savefig(output_path, dpi=200, bbox_inches="tight")  # 保存树状图图片 / デンドログラム画像を保存する
    print(f"saved dendrogram: {output_path}")  # 输出保存路径 / 保存先パスを表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
saved dendrogram: C:/Users/Administrator/Desktop/100ノック/Chapter 6/knock58_dendrogram.png
'''
