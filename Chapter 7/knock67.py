'''
knock67.py: 精度の計測 / 精度の計測

精度、適合率、再現率、F1スコアなどの評価指標を計算する
/ 精度、適合率、再現率、F1スコアなどの評価指標を計算する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
from sklearn.metrics import (  # メトリクス関数を導入 / 导入指标函数
    accuracy_score,  # 精度スコア / 精度分数
    precision_score,  # 適合率スコア / 精确率分数
    recall_score,  # 再現率スコア / 召回率分数
    f1_score,  # F1スコア / F1分数
    classification_report  # 分類レポート / 分类报告
)
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Evaluation Metrics")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签
    val_labels = val_df['label'].values  # 検証ラベルを取得 / 获取验证标签

    model = LogisticRegression(max_iter=1000, random_state=42)  # ロジスティック回帰モデルを初期化 / 初始化逻辑回归模型
    model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

    print("\nCalculating evaluation metrics...")  # メトリクス計算開始を出力 / 输出指标计算开始

    val_predictions = model.predict(val_features)  # 予測を行う / 进行预测

    accuracy = accuracy_score(val_labels, val_predictions)  # 精度を計算 / 计算精度
    precision = precision_score(val_labels, val_predictions)  # 適合率を計算 / 计算精确率
    recall = recall_score(val_labels, val_predictions)  # 再現率を計算 / 计算召回率
    f1 = f1_score(val_labels, val_predictions)  # F1スコアを計算 / 计算F1分数

    print("\n[Main Metrics]:")  # メインメトリクスを表示 / 显示主要指标
    print(f"Accuracy:  {accuracy:.6f}")  # 精度を出力 / 输出精度
    print(f"Precision: {precision:.6f}")  # 適合率を出力 / 输出精确率
    print(f"Recall:    {recall:.6f}")  # 再現率を出力 / 输出召回率
    print(f"F1-Score:  {f1:.6f}")  # F1スコアを出力 / 输出F1分数

    print("\n[Macro Averages]:")  # マクロ平均を表示 / 显示宏平均
    print(f"Macro Precision: {precision_score(val_labels, val_predictions, average='macro'):.6f}")  # マクロ精度を出力 / 输出宏精确率
    print(f"Macro Recall:    {recall_score(val_labels, val_predictions, average='macro'):.6f}")  # マクロ再現率を出力 / 输出宏召回率
    print(f"Macro F1-Score:  {f1_score(val_labels, val_predictions, average='macro'):.6f}")  # マクロF1を出力 / 输出宏F1

    print("\n[Weighted Averages]:")  # 重み付き平均を表示 / 显示加权平均
    print(f"Weighted Precision: {precision_score(val_labels, val_predictions, average='weighted'):.6f}")  # 重み付き精度を出力 / 输出加权精确率
    print(f"Weighted Recall:    {recall_score(val_labels, val_predictions, average='weighted'):.6f}")  # 重み付き再現率を出力 / 输出加权召回率
    print(f"Weighted F1-Score:  {f1_score(val_labels, val_predictions, average='weighted'):.6f}")  # 重み付きF1を出力 / 输出加权F1

    print("\n[Detailed Classification Report]:")  # 詳細分類レポートを表示 / 显示详细分类报告
    print(classification_report(val_labels, val_predictions, target_names=['Negative', 'Positive']))  # レポートを出力 / 输出报告


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Evaluation Metrics
==================================================

Calculating evaluation metrics...

[Main Metrics]:
Accuracy:  0.798165
Precision: 0.763780
Recall:    0.873874
F1-Score:  0.815126

[Macro Averages]:
Macro Precision: 0.804967
Macro Recall:    0.796750
Macro F1-Score:  0.796452

[Weighted Averages]:
Weighted Precision: 0.804211
Weighted Recall:    0.798165
Weighted F1-Score:  0.796795

[Detailed Classification Report]:
              precision    recall  f1-score   support

    Negative       0.85      0.72      0.78       428
    Positive       0.76      0.87      0.82       444

    accuracy                           0.80       872
   macro avg       0.80      0.80      0.80       872
weighted avg       0.80      0.80      0.80       872
'''
