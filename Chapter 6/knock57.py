'''
knock57.py: k-means聚类 / k-meansクラスタリング

提取国家名对应的词向量。 / 国名に対応する単語ベクトルを抽出する。
使用 k=5 进行 k-means 聚类。 / k=5 で k-means クラスタリングを行う。
'''

from chapter6_utils import load_country_vectors, load_vectors  # 导入国家向量加载函数和模型加载函数 / 国名ベクトル読み込み関数とモデル読み込み関数を導入する


def main():  # 定义主函数 / メイン関数を定義する
    try:  # 尝试导入数值和聚类库 / 数値計算とクラスタリング用ライブラリを読み込む
        import numpy as np  # 导入numpy / numpyをインポートする
        from sklearn.cluster import KMeans  # 导入KMeans聚类器 / KMeansクラスタリング器をインポートする
    except ImportError as exc:  # 如果依赖缺失则进入异常 / 依存関係不足なら例外処理へ進む
        raise RuntimeError("scikit-learn is required. Run: python -m pip install scikit-learn") from exc  # 抛出更明确的安装提示 / より明確な導入案内エラーを出す

    model = load_vectors()  # 读取预训练词向量模型 / 学習済み単語ベクトルモデルを読み込む
    countries, vectors = load_country_vectors(model)  # 读取国家名列表和对应向量 / 国名一覧と対応ベクトルを読み込む
    if not vectors:  # 如果一个国家向量都没有则报错 / 国名ベクトルが1つも無ければエラーにする
        raise RuntimeError("No country vectors were found in the model.")  # 抛出无数据错误 / データなしエラーを送出する
    vectors = np.asarray(vectors)  # 把向量列表转成numpy数组 / ベクトル一覧をnumpy配列へ変換する

    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")  # 构造k=5且可复现的KMeans模型 / k=5で再現可能なKMeansモデルを作成する
    labels = kmeans.fit_predict(vectors)  # 对全部国家向量进行聚类并得到标签 / 全国名ベクトルをクラスタリングしてラベルを得る

    for cluster_id in range(5):  # 依次输出5个簇 / 5つのクラスタを順に出力する
        members = [country for country, label in zip(countries, labels) if label == cluster_id]  # 收集属于当前簇的国家名 / 現在のクラスタに属する国名を集める
        print(f"cluster {cluster_id}: {', '.join(members)}")  # 输出当前簇包含的国家 / 現在のクラスタに含まれる国名を表示する


if __name__ == "__main__":  # 只有直接运行文件时才执行 / ファイルを直接実行した場合のみ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
运行结果: / 実行結果:
cluster 0: Albania, Armenia, Azerbaijan, Belarus, Bulgaria, Cyprus, Georgia, Kazakhstan, Kyrgyzstan, Macedonia, Moldova, Montenegro, Russia, Serbia, Tajikistan, Turkey, Turkmenistan, Ukraine, Uzbekistan
cluster 1: Afghanistan, Algeria, Bahamas, Bahrain, Belize, Chile, Cuba, Dominica, Ecuador, Egypt, Honduras, Iran, Iraq, Jamaica, Jordan, Lebanon, Libya, Morocco, Nicaragua, Oman, Peru, Qatar, Somalia, Syria, Tunisia, Uruguay, Venezuela
cluster 2: Australia, Bangladesh, Bhutan, Canada, China, Fiji, Greenland, Indonesia, Japan, Laos, Nepal, Pakistan, Philippines, Samoa, Taiwan, Thailand, Tuvalu, Vietnam
cluster 3: Austria, Belgium, Croatia, Denmark, England, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Liechtenstein, Lithuania, Malta, Norway, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden, Switzerland
cluster 4: Angola, Botswana, Burundi, Eritrea, Gabon, Gambia, Ghana, Guinea, Guyana, Kenya, Liberia, Madagascar, Malawi, Mali, Mauritania, Mozambique, Namibia, Niger, Nigeria, Rwanda, Senegal, Sudan, Suriname, Uganda, Zambia, Zimbabwe
'''
