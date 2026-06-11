'''
knock65.py: テキストのポジネガの予測 / テキストのポジネガの予測

テキスト入力を受け取って正負を予測するシステムを構築する
/ テキスト入力を受け取って正負を予測するシステムを構築する
'''

from chapter7_utils import load_sst2_data, create_bow_features  # データセット読み込み関数を導入 / 导入数据集读取函数
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデルを導入 / 导入逻辑回归模型
import numpy as np  # 数値計算ライブラリを導入 / 导入数值计算库


class SentimentAnalyzer:  # センチメント分析クラスを定義 / 定义情感分析类
    def __init__(self):  # 初期化メソッドを定義 / 定义初始化方法
        self.model = None  # モデルを初期化 / 初始化模型
        self.vectorizer = None  # ベクトル化器を初期化 / 初始化向量化器

    def train(self, texts, labels):  # モデルを学習するメソッドを定義 / 定义模型学习方法
        from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF特征提取器を導入 / 导入TF-IDF特征提取器
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')  # ベクトル化器を初期化 / 初始化向量化器
        features = self.vectorizer.fit_transform(texts)  # 特徴を抽出 / 提取特征
        self.model = LogisticRegression(max_iter=1000, random_state=42)  # モデルを初期化 / 初始化模型
        self.model.fit(features, labels)  # モデルを学習 / 训练模型

    def predict(self, text):  # テキストの感情を予測するメソッドを定義 / 定义文本情感预测方法
        features = self.vectorizer.transform([text])  # テキストを特徴に変換 / 将文本转换为特征
        prediction = self.model.predict(features)[0]  # 予測を取得 / 获取预测
        probability = self.model.predict_proba(features)[0]  # 確率を取得 / 获取概率
        return prediction, probability  # 予測と確率を返す / 返回预测和概率

    def get_sentiment_label(self, prediction):  # 予測ラベルを取得するメソッドを定義 / 定义获取预测标签方法
        return "Positive" if prediction == 1 else "Negative"  # ラベルを返す / 返回标签


def main():  # メイン関数を定義する / 定义主函数
    train_df, val_df = load_sst2_data()  # SST-2データセットを読み込む / 读取SST-2数据集

    print("=" * 50)  # 区切り線を出力 / 输出分隔线
    print("Sentiment Analysis System")  # タイトルを出力 / 输出标题
    print("=" * 50)  # 区切り線を出力 / 输出分隔线

    analyzer = SentimentAnalyzer()  # 分析器を初期化 / 初始化分析器
    analyzer.train(train_df['sentence'].values, train_df['label'].values)  # モデルを学習 / 训练模型

    print("\nModel trained successfully!")  # トレーニング完了を出力 / 输出训练完成

    print("\n[Sample Predictions from Validation Set]:")  # サンプル予測を表示 / 显示样本预测

    test_samples = [  # テストサンプルを定義 / 定义测试样本
        "This movie is absolutely fantastic! I loved every minute of it.",  # ポジティブサンプル / 积极样本
        "The plot was boring and the acting was terrible.",  # ネガティブサンプル / 消极样本
        "It was okay, nothing special.",  # ニュートラルサンプル / 中性样本
    ]

    for text in test_samples:  # 各テストサンプルについて反復 / 对每个测试样本进行迭代
        prediction, probability = analyzer.predict(text)  # 予測を行う / 进行预测
        sentiment = analyzer.get_sentiment_label(prediction)  # センチメントラベルを取得 / 获取情感标签
        print(f"\nText: {text}")  # テキストを出力 / 输出文本
        print(f"Prediction: {sentiment}")  # 予測を出力 / 输出预测
        print(f"Confidence: {max(probability):.4f}")  # 信頼度を出力 / 输出置信度
        print(f"Probabilities: Negative={probability[0]:.4f}, Positive={probability[1]:.4f}")  # 確率を出力 / 输出概率

    print("\n[Validation Set Performance]:")  # 検証セット性能を表示 / 显示验证集性能
    correct = 0  # 正解数を初期化 / 初始化正确数
    for i in range(len(val_df)):  # 各検証サンプルについて反復 / 对每个验证样本进行迭代
        prediction, _ = analyzer.predict(val_df['sentence'].iloc[i])  # 予測を行う / 进行预测
        if prediction == val_df['label'].iloc[i]:  # 予測が正解と一致する場合 / 如果预测与实际标签匹配
            correct += 1  # 正解数をインクリメント / 正确数加1
    accuracy = correct / len(val_df)  # 精度を計算 / 计算精度
    print(f"Validation Accuracy: {accuracy:.6f}")  # 精度を出力 / 输出精度


if __name__ == "__main__":  # ファイルを直接実行した場合のみ動かす / 只有直接运行文件时才执行
    main()  # メイン関数を呼び出す / 调用主函数

'''
运行结果: / 実行結果:
==================================================
Sentiment Analysis System
==================================================

Model trained successfully!

[Sample Predictions from Validation Set]:

Text: This movie is absolutely fantastic! I loved every minute of it.
Prediction: Positive
Confidence: 0.9479
Probabilities: Negative=0.0521, Positive=0.9479

Text: The plot was boring and the acting was terrible.
Prediction: Negative
Confidence: 0.9796
Probabilities: Negative=0.9796, Positive=0.0204

Text: It was okay, nothing special.
Prediction: Positive
Confidence: 0.7867
Probabilities: Negative=0.2133, Positive=0.7867

[Validation Set Performance]:
Validation Accuracy: 0.798165
'''
