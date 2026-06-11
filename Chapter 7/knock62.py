'''
knock62.py: 学習 / 学習

特徴ベクトルを用いてロジスティック回帰モデルを学習する
/ 特徴ベクトルを用いてロジスティック回帰モデルを学習する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Logistic Regression Training")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签

    print("\nTraining Logistic Regression model...")  # トレーニング開始を出力 / 输出训练开始

    model = LogisticRegression(max_iter=1000, random_state=42)  # ロジスティック回帰モデルを初期化 / 初始化逻辑回归模型
    model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

    print("Model trained successfully!")  # トレーニング完了を出力 / 输出训练完成
    print(f"\nModel parameters:")  # モデルパラメータを出力 / 输出模型参数
    print(f"- Intercept: {model.intercept_}")  # インターセプトを出力 / 输出截距
    print(f"- Number of features: {len(model.coef_[0])}")  # 特徴数を出力 / 输出特征数
    print(f"- Classes: {model.classes_}")  # クラスを出力 / 输出类别

    train_score = model.score(train_features, train_labels)  # 訓練精度を計算 / 计算训练精度
    val_score = model.score(val_features, val_df['label'].values)  # 検証精度を計算 / 计算验证精度

    print(f"\nTraining accuracy: {train_score:.6f}")  # 訓練精度を出力 / 输出训练精度
    print(f"Validation accuracy: {val_score:.6f}")  # 検証精度を出力 / 输出验证精度


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Logistic Regression Training
==================================================

Training Logistic Regression model...
Model trained successfully!

Model parameters:
- Intercept: [0.11912149]
- Number of features: 10000
- Classes: [0 1]

Training accuracy: 0.901840
Validation accuracy: 0.798165
'''
