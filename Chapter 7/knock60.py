'''
knock60.py: データの入手・整形 / データの入手・整形

SST-2データセットの統計情報を取得し、表示する
/ SST-2データセットの統計情報を取得して表示する
'''

import pandas as pd  # データフレーム操作ライブラリを導入 / 导入数据框操作库
from chapter7_utils import load_sst2_data, get_dataset_stats  # データセット読み込み関数を導入 / 导入数据集读取函数


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("SST-2 Dataset Statistics")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_stats = get_dataset_stats(train_df)  # 訓練データの統計情報を取得 / 获取训练数据的统计信息
    val_stats = get_dataset_stats(val_df)  # 検証データの統計情報を取得 / 获取验证数据的统计信息

    print("\n[Training Data]")  # 訓練データセクション / 训练数据部分
    print(f"Total samples: {train_stats['total_samples']}")  # 総サンプル数を出力 / 输出总样本数
    print(f"Number of labels: {train_stats['num_labels']}")  # ラベル数を出力 / 输出标签数
    print(f"Label distribution: {train_stats['label_distribution']}")  # ラベル分布を出力 / 输出标签分布
    print(f"Average text length: {train_stats['avg_text_length']:.2f}")  # テキスト平均長を出力 / 输出文本平均长度
    print(f"Max text length: {train_stats['max_text_length']}")  # テキスト最大長を出力 / 输出文本最大长度
    print(f"Min text length: {train_stats['min_text_length']}")  # テキスト最小長を出力 / 输出文本最小长度

    print("\n[Validation Data]")  # 検証データセクション / 验证数据部分
    print(f"Total samples: {val_stats['total_samples']}")  # 総サンプル数を出力 / 输出总样本数
    print(f"Number of labels: {val_stats['num_labels']}")  # ラベル数を出力 / 输出标签数
    print(f"Label distribution: {val_stats['label_distribution']}")  # ラベル分布を出力 / 输出标签分布
    print(f"Average text length: {val_stats['avg_text_length']:.2f}")  # テキスト平均長を出力 / 输出文本平均长度
    print(f"Max text length: {val_stats['max_text_length']}")  # テキスト最大長を出力 / 输出文本最大长度
    print(f"Min text length: {val_stats['min_text_length']}")  # テキスト最小長を出力 / 输出文本最小长度

    print("\n[Sample Data]")  # サンプルデータセクション / 样本数据部分
    print("\nFirst 3 training samples:")  # 最初の3つの訓練サンプルを表示 / 显示前3个训练样本
    print(train_df[['sentence', 'label']].head(3))  # データサンプルを表示 / 显示数据样本


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
SST-2 Dataset Statistics
==================================================

[Training Data]
Total samples: 67349
Number of labels: 2
Label distribution: {1: 37569, 0: 29780}
Average text length: 53.51
Max text length: 268
Min text length: 2

[Validation Data]
Total samples: 872
Number of labels: 2
Label distribution: {1: 444, 0: 428}
Average text length: 105.84
Max text length: 244
Min text length: 6

[Sample Data]

First 3 training samples:
                                            sentence  label
0       hide new secretions from the parental units       0
1               contains no wit , only labored gags       0
2  that loves its characters and communicates som...      1
'''
