'''
tutorial08_quickstart.py: PyTorch入门[8]快速开始 / PyTorch入門[8]クイックスタート

整合Dataset、DataLoader、模型、训练、评估与保存流程。
/ Dataset、DataLoader、モデル、学習、評価、保存の流れをまとめて実行する。
'''

from pathlib import Path  # 导入路径处理类 / パス処理クラスを導入する

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 导入神经网络模块 / ニューラルネットワークモジュールを導入する
from torch.utils.data import DataLoader, TensorDataset  # 导入数据工具 / データツールを導入する


class QuickstartModel(nn.Module):  # 定义快速开始模型 / クイックスタート用モデルを定義する
    def __init__(self):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))  # 定义分类网络 / 分類ネットワークを定義する

    def forward(self, x):  # 定义前向传播 / 順伝播を定義する
        return self.net(x)  # 返回logit / logitを返す


def make_data():  # 创建玩具数据 / 小さなデータを作る
    x = torch.randn(512, 4)  # 创建随机特征 / 乱数特徴量を作る
    y = (x[:, 0] - x[:, 1] + x[:, 2] > 0).long()  # 根据线性规则生成标签 / 線形規則でラベルを作る
    return TensorDataset(x[:400], y[:400]), TensorDataset(x[400:], y[400:])  # 返回训练和测试数据 / 訓練とテストデータを返す


def train(dataloader, model, loss_fn, optimizer, device):  # 定义训练函数 / 学習関数を定義する
    model.train()  # 切换到训练模式 / 学習モードに切り替える
    for x, y in dataloader:  # 遍历训练batch / 訓練batchを走査する
        x = x.to(device)  # 特征移动到设备 / 特徴量をデバイスへ移す
        y = y.to(device)  # 标签移动到设备 / ラベルをデバイスへ移す
        pred = model(x)  # 计算预测 / 予測を計算する
        loss = loss_fn(pred, y)  # 计算损失 / 損失を計算する
        optimizer.zero_grad()  # 清空梯度 / 勾配を消す
        loss.backward()  # 反向传播 / 逆伝播を行う
        optimizer.step()  # 更新参数 / パラメータを更新する


def test(dataloader, model, loss_fn, device):  # 定义测试函数 / テスト関数を定義する
    model.eval()  # 切换到评估模式 / 評価モードに切り替える
    loss_total = 0.0  # 初始化损失总和 / 損失合計を初期化する
    correct = 0  # 初始化正确数 / 正解数を初期化する
    with torch.no_grad():  # 测试时不计算梯度 / テスト時は勾配を計算しない
        for x, y in dataloader:  # 遍历测试batch / テストbatchを走査する
            x = x.to(device)  # 特征移动到设备 / 特徴量をデバイスへ移す
            y = y.to(device)  # 标签移动到设备 / ラベルをデバイスへ移す
            pred = model(x)  # 计算预测 / 予測を計算する
            loss_total += loss_fn(pred, y).item() * x.size(0)  # 累加损失 / 損失を加算する
            correct += (pred.argmax(dim=1) == y).sum().item()  # 累加正确数 / 正解数を加算する
    size = len(dataloader.dataset)  # 获取测试样本数 / テストサンプル数を取得する
    print(f"accuracy={(correct / size):.6f}, avg_loss={(loss_total / size):.6f}")  # 输出测试结果 / テスト結果を出力する


def main():  # 定义主函数 / メイン関数を定義する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备 / デバイスを自動選択する
    train_dataset, test_dataset = make_data()  # 创建数据集 / データセットを作る
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练DataLoader / 訓練DataLoaderを作る
    test_loader = DataLoader(test_dataset, batch_size=32)  # 创建测试DataLoader / テストDataLoaderを作る
    model = QuickstartModel().to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    loss_fn = nn.CrossEntropyLoss()  # 创建损失函数 / 損失関数を作る
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # 创建优化器 / 最適化器を作る
    save_path = Path(__file__).resolve().parent / "tutorial08_model.pth"  # 设置保存路径 / 保存パスを設定する

    print(f"Using device: {device}")  # 输出设备 / デバイスを出力する
    for epoch in range(1, 6):  # 训练5轮 / 5epoch学習する
        print(f"epoch {epoch}")  # 输出epoch编号 / epoch番号を出力する
        train(train_loader, model, loss_fn, optimizer, device)  # 执行训练 / 学習を実行する
        test(test_loader, model, loss_fn, device)  # 执行测试 / テストを実行する
    torch.save(model.state_dict(), save_path)  # 保存模型参数 / モデルパラメータを保存する
    print(f"saved to: {save_path}")  # 输出保存路径 / 保存パスを出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
