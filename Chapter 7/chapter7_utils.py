'''
chapter7_utils.py: 第7章通用工具函数 / 第7章の共通ユーティリティ関数

提供数据读取、特征向量转换等共同功能的工具库
/ データの読み込みや特徴ベクトル変換など共通機能を提供するユーティリティライブラリ
'''

import pandas as pd  # データフレーム操作ライブラリを導入 / 数据框操作库
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF特征提取器を導入 / TF-IDF特征提取器
from sklearn.feature_extraction.text import CountVectorizer  # 词袋模型を導入 / 词袋模型


def load_sst2_data(train_path='data/train.csv', val_path='data/val.csv'):  # SST-2データセットを読み込む / 读取SST-2数据集
    '''データセットの読み込み / データの読み込み'''
    train_df = pd.read_csv(train_path)  # 訓練データを読み込む / 读取训练数据
    val_df = pd.read_csv(val_path)  # 検証データを読み込む / 读取验证数据
    return train_df, val_df  # 訓練データと検証データを返す / 返回训练和验证数据


def create_bow_features(train_texts, val_texts, max_features=10000):  # 词袋特征を作成する / 创建词袋特征
    '''TF-IDFベクトル化器を作成してテキストを特徴ベクトルに変換する / 用TF-IDF向量化器将文本转换为特征向量'''
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')  # ベクトル化器を初期化 / 初始化向量化器
    train_features = vectorizer.fit_transform(train_texts)  # 訓練データで学習して変換する / 用训练数据拟合并转换
    val_features = vectorizer.transform(val_texts)  # 検証データを変換する / 转换验证数据
    return train_features, val_features, vectorizer  # 特徴ベクトルとベクトル化器を返す / 返回特征向量和向量化器


def get_dataset_stats(df):  # データセットの統計情報を取得する / 获取数据集的统计信息
    '''データセットの基本統計量を計算する / 计算数据集的基本统计量'''
    stats = {  # 統計情報を辞書として格納 / 将统计信息存储为字典
        'total_samples': len(df),  # 総サンプル数 / 总样本数
        'num_labels': df['label'].nunique(),  # ラベル数 / 标签数
        'label_distribution': df['label'].value_counts().to_dict(),  # ラベル分布 / 标签分布
        'avg_text_length': df['sentence'].str.len().mean(),  # テキスト平均長 / 文本平均长度
        'max_text_length': df['sentence'].str.len().max(),  # テキスト最大長 / 文本最大长度
        'min_text_length': df['sentence'].str.len().min(),  # テキスト最小長 / 文本最小长度
    }
    return stats  # 統計情報を返す / 返回统计信息
