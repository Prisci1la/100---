'''
knock68.py: 特徴量の重みの確認 / 特徴量の重みの確認

学習したモデルの特徴量の重みを分析し、重要度の高い特徴を抽出する
/ 学習したモデルの特徴量の重みを分析し、重要度の高い特徴を抽出する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签

    model = LogisticRegression(max_iter=1000, random_state=42)  # ロジスティック回帰モデルを初期化 / 初始化逻辑回归模型
    model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

    feature_names = vectorizer.get_feature_names_out()  # 特徴名を取得 / 获取特征名
    coefficients = model.coef_[0]  # 係数を取得 / 获取系数

    feature_importance = list(zip(feature_names, coefficients))  # 特徴と係数をペアにする / 将特征和系数配对

    print("\n[Top 20 Positive Features (Contribute to Positive Sentiment)]:")  # ポジティブ特徴を表示 / 显示积极特征
    positive_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:20]  # 上位20の正例特徴を取得 / 获取前20个正例特征
    for i, (feature, weight) in enumerate(positive_features):  # 各特徴について反復 / 对每个特征进行迭代
        print(f"{i+1:2d}. {feature:20s} : {weight:8.6f}")  # 特徴と重みを出力 / 输出特征和权重

    print("\n[Top 20 Negative Features (Contribute to Negative Sentiment)]:")  # ネガティブ特徴を表示 / 显示消极特征
    negative_features = sorted(feature_importance, key=lambda x: x[1], reverse=False)[:20]  # 上位20の負例特徴を取得 / 获取前20个负例特征
    for i, (feature, weight) in enumerate(negative_features):  # 各特徴について反復 / 对每个特征进行迭代
        print(f"{i+1:2d}. {feature:20s} : {weight:8.6f}")  # 特徴と重みを出力 / 输出特征和权重


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:

[Top 20 Positive Features (Contribute to Positive Sentiment)]:
 1. powerful             : 4.944601
 2. best                 : 4.733084
 3. hilarious            : 4.660514
 4. beautiful            : 4.581303
 5. remarkable           : 4.467454
 6. wonderful            : 4.170156
 7. fascinating          : 4.111473
 8. heart                : 4.066851
 9. terrific             : 3.998004
10. solid                : 3.994147
11. enjoyable            : 3.755532
12. appealing            : 3.748306
13. intriguing           : 3.741796
14. beautifully          : 3.672932
15. honest               : 3.656981
16. touching             : 3.574992
17. refreshing           : 3.571270
18. strong               : 3.561211
19. engrossing           : 3.457578
20. moving               : 3.442334

[Top 20 Negative Features (Contribute to Negative Sentiment)]:
 1. worst                : -5.944002
 2. lacks                : -5.395745
 3. lacking              : -4.964813
 4. bad                  : -4.830297
 5. lack                 : -4.571109
 6. mess                 : -4.500906
 7. devoid               : -4.337681
 8. stupid               : -4.198786
 9. failure              : -4.049106
10. waste                : -3.990846
11. flat                 : -3.888347
12. poor                 : -3.835918
13. depressing           : -3.820501
14. pretentious          : -3.783732
15. pointless            : -3.723219
16. poorly               : -3.701463
17. ugly                 : -3.655088
18. dull                 : -3.560590
19. clichés              : -3.556712
20. unfunny              : -3.538886
'''
