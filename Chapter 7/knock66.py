'''
knock66.py: 混同行列の作成 / 混同行列の作成

混同行列を生成して分類性能を評価する
/ 混同行列を生成して分類性能を評価する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
from sklearn.metrics import confusion_matrix  # 混同行列関数を導入 / 导入混淆矩阵函数
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


def print_confusion_matrix(cm, labels=None):  # 混同行列を表示する関数を定義 / 定义打印混淆矩阵的函数
    '''混同行列を整形して表示する / 混同行列を整形して表示する'''
    if labels is None:  # ラベルが指定されない場合 / 如果没有指定标签
        labels = ['Class 0', 'Class 1']  # デフォルトラベルを使用 / 使用默认标签
    print("\n" + "=" * 40)  # 区切り線を出力 / 输出分隔线
    print("Confusion Matrix")  # タイトルを出力 / 输出标题
    print("=" * 40)  # 区切り線を出力 / 输出分隔线
    print("\n" + " " * 20 + "Predicted")  # ヘッダを出力 / 输出头部
    print(" " * 15 + f"{labels[0]:>12} {labels[1]:>12}")  # ラベルヘッダを出力 / 输出标签头部
    for i, label in enumerate(labels):  # 各ラベルについて反復 / 对每个标签进行迭代
        print(f"{label:>15} {cm[i, 0]:>12} {cm[i, 1]:>12}")  # 行を出力 / 输出行
    print()


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Confusion Matrix Analysis")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签
    val_labels = val_df['label'].values  # 検証ラベルを取得 / 获取验证标签

    model = LogisticRegression(max_iter=1000, random_state=42)  # ロジスティック回帰モデルを初期化 / 初始化逻辑回归模型
    model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

    print("\nGenerating confusion matrix for validation set...")  # 混同行列生成開始を出力 / 输出混淆矩阵生成开始

    val_predictions = model.predict(val_features)  # 予測を行う / 进行预测
    cm = confusion_matrix(val_labels, val_predictions)  # 混同行列を計算 / 计算混淆矩阵

    print_confusion_matrix(cm, labels=['Negative (0)', 'Positive (1)'])  # 混同行列を表示 / 显示混淆矩阵

    tn, fp, fn, tp = cm.ravel()  # 各要素を取得 / 获取各个元素

    print("\n[Matrix Elements]:")  # 行列要素を表示 / 显示矩阵元素
    print(f"True Negatives (TN): {tn}")  # 真陰性を出力 / 输出真负例
    print(f"False Positives (FP): {fp}")  # 偽陽性を出力 / 输出假正例
    print(f"False Negatives (FN): {fn}")  # 偽陰性を出力 / 输出假负例
    print(f"True Positives (TP): {tp}")  # 真陽性を出力 / 输出真正例

    print("\n[Prediction Distribution]:")  # 予測分布を表示 / 显示预测分布
    unique, counts = np.unique(val_predictions, return_counts=True)  # ユニーク値とカウントを取得 / 获取唯一值和计数
    for label, count in zip(unique, counts):  # 各ラベルについて反復 / 对每个标签进行迭代
        print(f"Class {label}: {count} predictions")  # ラベルのカウントを出力 / 输出标签的计数


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Confusion Matrix Analysis
==================================================

Generating confusion matrix for validation set...

========================================
Confusion Matrix
========================================

                    Predicted
               Negative (0) Positive (1)
   Negative (0)          308          120
   Positive (1)           56          388


[Matrix Elements]:
True Negatives (TN): 308
False Positives (FP): 120
False Negatives (FN): 56
True Positives (TP): 388

[Prediction Distribution]:
Class 0: 364 predictions
Class 1: 508 predictions
'''
