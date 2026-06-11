'''
knock64.py: 条件付き確率 / 条件付き確率

各クラスの条件付き確率を計算する
/ 各クラスの条件付き確率を計算する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Conditional Probabilities")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签

    model = LogisticRegression(max_iter=1000, random_state=42)  # ロジスティック回帰モデルを初期化 / 初始化逻辑回归模型
    model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

    print("\nCalculating conditional probabilities on validation set...")  # 条件付き確率計算開始を出力 / 输出条件概率计算开始

    val_probabilities = model.predict_proba(val_features)  # 予測確率を計算 / 计算预测概率

    print(f"\nClass labels: {model.classes_}")  # クラスラベルを出力 / 输出类标签
    print(f"\nClass 0 (Negative) mean probability: {val_probabilities[:, 0].mean():.6f}")  # クラス0の平均確率を出力 / 输出类0的平均概率
    print(f"Class 1 (Positive) mean probability: {val_probabilities[:, 1].mean():.6f}")  # クラス1の平均確率を出力 / 输出类1的平均概率

    print("\n[Probability Distribution Statistics]:")  # 確率分布統計を表示 / 显示概率分布统计

    for class_idx, class_name in enumerate(['Negative', 'Positive']):  # 各クラスについて反復 / 对每个类进行迭代
        probs = val_probabilities[:, class_idx]  # クラスの確率を取得 / 获取类的概率
        print(f"\n{class_name} (Class {class_idx}):")  # クラス名を出力 / 输出类名
        print(f"  Mean: {probs.mean():.6f}")  # 平均を出力 / 输出平均值
        print(f"  Std: {probs.std():.6f}")  # 標準偏差を出力 / 输出标准差
        print(f"  Min: {probs.min():.6f}")  # 最小値を出力 / 输出最小值
        print(f"  Max: {probs.max():.6f}")  # 最大値を出力 / 输出最大值
        print(f"  Median: {np.median(probs):.6f}")  # 中央値を出力 / 输出中位数

    print("\n[Sample Probabilities]:")  # サンプル確率を表示 / 显示样本概率
    for i in range(min(5, len(val_df))):  # 最初の5サンプルについて反復 / 对前5个样本进行迭代
        print(f"\nSample {i+1}:")  # サンプル番号を出力 / 输出样本号
        print(f"  P(Negative): {val_probabilities[i, 0]:.6f}")  # 負例確率を出力 / 输出负例概率
        print(f"  P(Positive): {val_probabilities[i, 1]:.6f}")  # 正例確率を出力 / 输出正例概率


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Conditional Probabilities
==================================================

Calculating conditional probabilities on validation set...

Class labels: [0 1]

Class 0 (Negative) mean probability: 0.433713
Class 1 (Positive) mean probability: 0.566287

[Probability Distribution Statistics]:

Negative (Class 0):
  Mean: 0.433713
  Std: 0.308407
  Min: 0.004984
  Max: 0.993270
  Median: 0.385616

Positive (Class 1):
  Mean: 0.566287
  Std: 0.308407
  Min: 0.006730
  Max: 0.995016
  Median: 0.614384

[Sample Probabilities]:

Sample 1:
  P(Negative): 0.026931
  P(Positive): 0.973069

Sample 2:
  P(Negative): 0.923937
  P(Positive): 0.076063

Sample 3:
  P(Negative): 0.095501
  P(Positive): 0.904499

Sample 4:
  P(Negative): 0.226094
  P(Positive): 0.773906

Sample 5:
  P(Negative): 0.908623
  P(Positive): 0.091377
'''
