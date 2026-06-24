'''
tutorial02_datasets_dataloaders.py: PyTorch入门[2]Dataset与DataLoader / PyTorch入門[2]DatasetとDataLoader

使用自定义Dataset类和DataLoader按batch取出数据。
/ カスタムDatasetクラスとDataLoaderでbatch単位のデータを取り出す。
'''

import torch  # 导入PyTorch / PyTorchを導入する
from torch.utils.data import DataLoader, Dataset  # 导入Dataset和DataLoader / DatasetとDataLoaderを導入する


class ToyDataset(Dataset):  # 定义玩具数据集 / 小さなデータセットを定義する
    def __init__(self):  # 初始化数据集 / データセットを初期化する
        self.x = torch.arange(20, dtype=torch.float32).reshape(10, 2)  # 创建10个二维样本 / 10個の2次元サンプルを作る
        self.y = (self.x.sum(dim=1) > 10).long()  # 根据和是否大于10生成标签 / 合計が10より大きいかでラベルを作る

    def __len__(self):  # 返回样本数量 / サンプル数を返す
        return len(self.x)  # 返回x的长度 / xの長さを返す

    def __getitem__(self, index):  # 返回单个样本 / 1つのサンプルを返す
        return self.x[index], self.y[index]  # 返回特征和标签 / 特徴量とラベルを返す


def main():  # 定义主函数 / メイン関数を定義する
    dataset = ToyDataset()  # 创建数据集 / データセットを作る
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)  # 创建DataLoader / DataLoaderを作る
    for batch_id, (x, y) in enumerate(dataloader):  # 遍历batch / batchを走査する
        print(f"batch {batch_id}")  # 输出batch编号 / batch番号を出力する
        print(f"x shape: {x.shape}")  # 输出特征形状 / 特徴量の形状を出力する
        print(f"y: {y}")  # 输出标签 / ラベルを出力する
        break  # 只显示一个batch / 1つのbatchだけ表示する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
运行结果: / 実行結果:
batch 0
x shape: torch.Size([4, 2])
y: tensor([0, 0, 0, 1])
'''
