'''
knock61.py: 特徴ベクトル / 特徴ベクトル

テキストを特徴ベクトル（TF-IDF）に変換する
/ テキストを特徴ベクトル（TF-IDF）に変換する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Feature Vectorization (TF-IDF)")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    print(f"\nVectorizer type: {type(vectorizer).__name__}")  # ベクトル化器の種類を出力 / 输出向量化器类型
    print(f"Number of features: {train_features.shape[1]}")  # 特徴数を出力 / 输出特征数
    print(f"Train set shape: {train_features.shape}")  # 訓練データの形状を出力 / 输出训练数据形状
    print(f"Validation set shape: {val_features.shape}")  # 検証データの形状を出力 / 输出验证数据形状

    print(f"\nTrain data sparsity: {1 - (train_features.nnz / (train_features.shape[0] * train_features.shape[1])):.4f}")  # 疎度を出力 / 输出稀疏性
    print(f"Validation data sparsity: {1 - (val_features.nnz / (val_features.shape[0] * val_features.shape[1])):.4f}")  # 疎度を出力 / 输出稀疏性

    print("\n[Top 20 features (vocabulary)]:")  # 上位20の特徴を表示 / 显示前20个特征
    feature_names = vectorizer.get_feature_names_out()  # 特徴名を取得 / 获取特征名
    print(f"Total vocabulary size: {len(feature_names)}")  # ボキャブラリサイズを出力 / 输出词汇表大小
    print("First 20 features:", list(feature_names[:20]))  # 最初の20の特徴を出力 / 输出前20个特征


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Feature Vectorization (TF-IDF)
==================================================

Vectorizer type: TfidfVectorizer
Number of features: 10000
Train set shape: (67349, 10000)
Validation set shape: (872, 10000)

Train data sparsity: 0.9995
Validation data sparsity: 0.9992

[Top 20 features (vocabulary)]:
Total vocabulary size: 10000
First 20 features: ['000', '10', '100', '101', '103', '105', '10th', '11', '110', '112', '12', '12th', '13', '13th', '146', '15', '16', '163', '18', '19']
'''
