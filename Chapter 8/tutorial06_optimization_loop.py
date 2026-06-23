'''
tutorial06_optimization_loop.py: PyTorch入门[6]优化循环 / PyTorch入門[6]最適化ループ

在命令行中用DataLoader训练一个小型分类模型。
/ コマンドラインでDataLoaderを使い、小さな分類モデルを学習する。
'''

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 导入神经网络模块 / ニューラルネットワークモジュールを導入する
from torch.utils.data import DataLoader, TensorDataset  # 导入数据工具 / データツールを導入する


class TinyClassifier(nn.Module):  # 定义小型分类器 / 小さな分類器を定義する
    def __init__(self):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))  # 定义两层网络 / 2層ネットワークを定義する

    def forward(self, x):  # 定义前向传播 / 順伝播を定義する
        return self.net(x)  # 返回分类logit / 分類logitを返す


def main():  # 定义主函数 / メイン関数を定義する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备 / デバイスを自動選択する
    x = torch.randn(256, 2)  # 创建随机二维数据 / 乱数の2次元データを作る
    y = (x[:, 0] + x[:, 1] > 0).long()  # 根据直线生成标签 / 直線でラベルを作る
    dataloader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)  # 创建DataLoader / DataLoaderを作る
    model = TinyClassifier().to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    loss_fn = nn.CrossEntropyLoss()  # 创建交叉熵损失 / 交差エントロピー損失を作る
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 创建SGD优化器 / SGD最適化器を作る

    print(f"Using device: {device}")  # 输出设备 / デバイスを出力する
    for epoch in range(1, 6):  # 训练5轮 / 5epoch学習する
        total_loss = 0.0  # 初始化损失总和 / 損失合計を初期化する
        for batch_x, batch_y in dataloader:  # 遍历batch / batchを走査する
            batch_x = batch_x.to(device)  # 特征移动到设备 / 特徴量をデバイスへ移す
            batch_y = batch_y.to(device)  # 标签移动到设备 / ラベルをデバイスへ移す
            optimizer.zero_grad()  # 清空梯度 / 勾配を消す
            pred = model(batch_x)  # 前向计算 / 順伝播を行う
            loss = loss_fn(pred, batch_y)  # 计算损失 / 損失を計算する
            loss.backward()  # 反向传播 / 逆伝播を行う
            optimizer.step()  # 更新参数 / パラメータを更新する
            total_loss += loss.item() * batch_x.size(0)  # 累加损失 / 損失を加算する
        print(f"epoch {epoch}: loss={total_loss / len(dataloader.dataset):.6f}")  # 输出每轮损失 / 各epochの損失を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
