'''
knock63.py: 予測 / 予測

学習したモデルを使用して検証データに対する予測を行う
/ 学習したモデルを使用して検証データに対する予測を行う
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
from sklearn.metrics import accuracy_score, confusion_matrix  # メトリクス関数を導入 / 导入指标函数
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Model Prediction")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签
    val_labels = val_df['label'].values  # 検証ラベルを取得 / 获取验证标签

    model = LogisticRegression(max_iter=1000, random_state=42)  # ロジスティック回帰モデルを初期化 / 初始化逻辑回归模型
    model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

    print("\nMaking predictions on validation set...")  # 予測開始を出力 / 输出预测开始

    val_predictions = model.predict(val_features)  # 検証データに対する予測を行う / 对验证数据进行预测
    val_probabilities = model.predict_proba(val_features)  # 予測確率を計算 / 计算预测概率

    accuracy = accuracy_score(val_labels, val_predictions)  # 精度を計算 / 计算精度

    print(f"\nValidation Accuracy: {accuracy:.6f}")  # 精度を出力 / 输出精度
    print(f"Correct predictions: {(val_predictions == val_labels).sum()}/{len(val_labels)}")  # 正解数を出力 / 输出正确预测数

    print("\n[Sample Predictions]:")  # サンプル予測を表示 / 显示样本预测
    for i in range(min(5, len(val_df))):  # 最初の5サンプルについて反復 / 对前5个样本进行迭代
        text = val_df['sentence'].iloc[i]  # テキストを取得 / 获取文本
        actual = val_labels[i]  # 実際のラベルを取得 / 获取实际标签
        predicted = val_predictions[i]  # 予測ラベルを取得 / 获取预测标签
        prob_negative, prob_positive = val_probabilities[i]  # 確率を取得 / 获取概率
        print(f"\nSample {i+1}:")  # サンプル番号を出力 / 输出样本号
        print(f"  Text: {text[:80]}...")  # テキストを出力 / 输出文本
        print(f"  Actual: {actual}, Predicted: {predicted}")  # ラベルを出力 / 输出标签
        print(f"  Probabilities: Negative={prob_negative:.4f}, Positive={prob_positive:.4f}")  # 確率を出力 / 输出概率


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Model Prediction
==================================================

Making predictions on validation set...

Validation Accuracy: 0.798165
Correct predictions: 696/872

[Sample Predictions]:

Sample 1:
  Text: it 's a charming and often affecting journey . ...
  Actual: 1, Predicted: 1
  Probabilities: Negative=0.0269, Positive=0.9731

Sample 2:
  Text: unflinchingly bleak and desperate ...
  Actual: 0, Predicted: 0
  Probabilities: Negative=0.9239, Positive=0.0761

Sample 3:
  Text: allows us to hope that nolan is poised to embark a major career as a commercial ...
  Actual: 1, Predicted: 1
  Probabilities: Negative=0.0955, Positive=0.9045

Sample 4:
  Text: the acting , costumes , music , cinematography and sound are all astounding give...
  Actual: 1, Predicted: 1
  Probabilities: Negative=0.2261, Positive=0.7739

Sample 5:
  Text: it 's slow -- very , very slow . ...
  Actual: 0, Predicted: 0
  Probabilities: Negative=0.9086, Positive=0.0914
'''
