'''
knock69.py: 正則化パラメータの変更 / 正則化パラメータの変更

異なる正則化係数（C値）でモデルを学習し、精度曲線を描画する
/ 異なる正則化係数（C値）でモデルを学習し、精度曲線を描画する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
from sklearn.metrics import accuracy_score  # 精度スコア関数を導入 / 导入精度分数函数
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库
import matplotlib.pyplot as plt  # グラフ描画ライブラリを導入 / 导入绘图库


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Regularization Parameter Tuning")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    train_features, val_features, vectorizer = create_bow_features(  # 特徴ベクトルを作成 / 创建特征向量
        train_df['sentence'].values,  # 訓練テキスト / 训练文本
        val_df['sentence'].values  # 検証テキスト / 验证文本
    )

    train_labels = train_df['label'].values  # 訓練ラベルを取得 / 获取训练标签
    val_labels = val_df['label'].values  # 検証ラベルを取得 / 获取验证标签

    c_values = np.logspace(-4, 2, 13)  # C値のログスケール配列を作成 / 创建C值的对数刻度数组
    train_accuracies = []  # 訓練精度リストを初期化 / 初始化训练精度列表
    val_accuracies = []  # 検証精度リストを初期化 / 初始化验证精度列表

    print(f"\nTesting {len(c_values)} different C values...")  # テスト数を出力 / 输出测试数

    for c in c_values:  # 各C値について反復 / 对每个C值进行迭代
        model = LogisticRegression(C=c, max_iter=1000, random_state=42)  # モデルを初期化 / 初始化模型
        model.fit(train_features, train_labels)  # モデルを学習 / 训练模型

        train_pred = model.predict(train_features)  # 訓練予測を行う / 进行训练预测
        val_pred = model.predict(val_features)  # 検証予測を行う / 进行验证预测

        train_acc = accuracy_score(train_labels, train_pred)  # 訓練精度を計算 / 计算训练精度
        val_acc = accuracy_score(val_labels, val_pred)  # 検証精度を計算 / 计算验证精度

        train_accuracies.append(train_acc)  # 訓練精度を追加 / 添加训练精度
        val_accuracies.append(val_acc)  # 検証精度を追加 / 添加验证精度

        print(f"C={c:8.4f}: Train Acc={train_acc:.6f}, Val Acc={val_acc:.6f}")  # 結果を出力 / 输出结果

    print("\n[Best Performance]:")  # 最高性能を表示 / 显示最高性能
    best_val_idx = np.argmax(val_accuracies)  # 最高検証精度のインデックスを取得 / 获取最高验证精度的索引
    print(f"Best C value: {c_values[best_val_idx]:.4f}")  # 最高C値を出力 / 输出最高C值
    print(f"Best validation accuracy: {val_accuracies[best_val_idx]:.6f}")  # 最高検証精度を出力 / 输出最高验证精度
    print(f"Corresponding train accuracy: {train_accuracies[best_val_idx]:.6f}")  # 対応する訓練精度を出力 / 输出对应的训练精度

    print("\nGenerating accuracy curve graph...")  # グラフ生成開始を出力 / 输出图表生成开始

    plt.figure(figsize=(10, 6))  # 図のサイズを設定 / 设置图表大小
    plt.semilogx(c_values, train_accuracies, marker='o', label='Train Accuracy', linewidth=2)  # 訓練精度曲線を描画 / 绘制训练精度曲线
    plt.semilogx(c_values, val_accuracies, marker='s', label='Validation Accuracy', linewidth=2)  # 検証精度曲線を描画 / 绘制验证精度曲线
    plt.xlabel('Regularization Parameter (C)', fontsize=12)  # X軸ラベルを設定 / 设置X轴标签
    plt.ylabel('Accuracy', fontsize=12)  # Y軸ラベルを設定 / 设置Y轴标签
    plt.title('Model Accuracy vs Regularization Parameter', fontsize=14)  # タイトルを設定 / 设置标题
    plt.legend(fontsize=10)  # 凡例を表示 / 显示图例
    plt.grid(True, alpha=0.3)  # グリッドを表示 / 显示网格
    plt.tight_layout()  # レイアウトを調整 / 调整布局
    plt.savefig('knock69_accuracy_curve.png', dpi=100)  # グラフを保存 / 保存图表
    print("Graph saved as 'knock69_accuracy_curve.png'")  # 保存完了を出力 / 输出保存完成
    plt.close()  # グラフを閉じる / 关闭图表


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Regularization Parameter Tuning
==================================================

Testing 13 different C values...
C=  0.0001: Train Acc=0.557826, Val Acc=0.509174
C=  0.0003: Train Acc=0.557826, Val Acc=0.509174
C=  0.0010: Train Acc=0.557826, Val Acc=0.509174
C=  0.0032: Train Acc=0.562993, Val Acc=0.514908
C=  0.0100: Train Acc=0.627017, Val Acc=0.591743
C=  0.0316: Train Acc=0.771251, Val Acc=0.706422
C=  0.1000: Train Acc=0.853287, Val Acc=0.770642
C=  0.3162: Train Acc=0.881394, Val Acc=0.791284
C=  1.0000: Train Acc=0.901840, Val Acc=0.798165
C=  3.1623: Train Acc=0.916495, Val Acc=0.800459
C= 10.0000: Train Acc=0.922894, Val Acc=0.795872
C= 31.6228: Train Acc=0.929561, Val Acc=0.783257
C=100.0000: Train Acc=0.931640, Val Acc=0.780963

[Best Performance]:
Best C value: 3.1623
Best validation accuracy: 0.800459
Corresponding train accuracy: 0.916495

Generating accuracy curve graph...
Graph saved as 'knock69_accuracy_curve.png'
'''
