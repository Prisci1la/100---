'''
tutorial05_autograd.py: PyTorch入门[5]自动微分 / PyTorch入門[5]自動微分

演示requires_grad、反向传播和梯度确认。
/ requires_grad、逆伝播、勾配確認を実演する。
'''

import torch  # 导入PyTorch / PyTorchを導入する


def main():  # 定义主函数 / メイン関数を定義する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备 / デバイスを自動選択する
    x = torch.ones(5, device=device)  # 创建输入张量 / 入力テンソルを作る
    y = torch.zeros(3, device=device)  # 创建目标张量 / 目標テンソルを作る
    w = torch.randn(5, 3, requires_grad=True, device=device)  # 创建需要梯度的权重 / 勾配が必要な重みを作る
    b = torch.randn(3, requires_grad=True, device=device)  # 创建需要梯度的偏置 / 勾配が必要なバイアスを作る
    z = torch.matmul(x, w) + b  # 计算线性输出 / 線形出力を計算する
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # 计算损失 / 損失を計算する
    loss.backward()  # 执行反向传播 / 逆伝播を実行する

    print(f"Using device: {device}")  # 输出设备 / デバイスを出力する
    print(f"loss: {loss.item():.6f}")  # 输出损失 / 損失を出力する
    print(f"w.grad:\n{w.grad}")  # 输出权重梯度 / 重みの勾配を出力する
    print(f"b.grad:\n{b.grad}")  # 输出偏置梯度 / バイアスの勾配を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
