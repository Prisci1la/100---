'''
tutorial04_build_model.py: PyTorch入门[4]模型构建 / PyTorch入門[4]モデル構築

通过继承torch.nn.Module定义神经网络模型，并移动到GPU或CPU。
/ torch.nn.Moduleを継承してニューラルネットワークを定義し、GPUまたはCPUへ移す。
'''

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 导入神经网络模块 / ニューラルネットワークモジュールを導入する


class NeuralNetwork(nn.Module):  # 定义神经网络类 / ニューラルネットワーククラスを定義する
    def __init__(self):  # 初始化模型层 / モデル層を初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        self.flatten = nn.Flatten()  # 创建展平层 / 平坦化層を作る
        self.linear_relu_stack = nn.Sequential(  # 创建顺序网络 / 順序ネットワークを作る
            nn.Linear(28 * 28, 512),  # 第一层全连接 / 1層目の全結合
            nn.ReLU(),  # ReLU激活函数 / ReLU活性化関数
            nn.Linear(512, 512),  # 第二层全连接 / 2層目の全結合
            nn.ReLU(),  # ReLU激活函数 / ReLU活性化関数
            nn.Linear(512, 10),  # 输出10类logit / 10クラスlogitを出力する
        )

    def forward(self, x):  # 定义前向传播 / 順伝播を定義する
        x = self.flatten(x)  # 展平输入图像 / 入力画像を平坦化する
        logits = self.linear_relu_stack(x)  # 通过全连接网络 / 全結合ネットワークに通す
        return logits  # 返回logit / logitを返す


def main():  # 定义主函数 / メイン関数を定義する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备 / デバイスを自動選択する
    model = NeuralNetwork().to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    x = torch.rand(1, 28, 28, device=device)  # 创建一张随机图像 / 乱数画像を1枚作る
    logits = model(x)  # 执行前向计算 / 順伝播を実行する
    prediction = logits.argmax(dim=1)  # 取最大logit作为预测类别 / 最大logitを予測クラスにする
    print(f"Using device: {device}")  # 输出设备 / デバイスを出力する
    print(model)  # 输出模型结构 / モデル構造を出力する
    print(f"predicted class: {prediction.item()}")  # 输出预测类别 / 予測クラスを出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
