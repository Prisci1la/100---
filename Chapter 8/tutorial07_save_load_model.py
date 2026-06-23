'''
tutorial07_save_load_model.py: PyTorch入门[7]模型保存与读取 / PyTorch入門[7]モデル保存と読み込み

保存state_dict，并重新读取到同结构模型中。
/ state_dictを保存し、同じ構造のモデルへ読み戻す。
'''

from pathlib import Path  # 导入路径处理类 / パス処理クラスを導入する

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 导入神经网络模块 / ニューラルネットワークモジュールを導入する


class TinyModel(nn.Module):  # 定义小型模型 / 小さなモデルを定義する
    def __init__(self):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        self.linear = nn.Linear(4, 2)  # 创建线性层 / 線形層を作る

    def forward(self, x):  # 定义前向传播 / 順伝播を定義する
        return self.linear(x)  # 返回线性输出 / 線形出力を返す


def main():  # 定义主函数 / メイン関数を定義する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备 / デバイスを自動選択する
    save_path = Path(__file__).resolve().parent / "tutorial07_model.pth"  # 设置保存路径 / 保存パスを設定する
    model = TinyModel().to(device)  # 创建模型并移动到设备 / モデルを作ってデバイスへ移す
    torch.save(model.state_dict(), save_path)  # 保存模型参数 / モデルパラメータを保存する
    loaded_model = TinyModel().to(device)  # 创建同结构模型 / 同じ構造のモデルを作る
    loaded_model.load_state_dict(torch.load(save_path, map_location=device))  # 读取参数 / パラメータを読み込む
    loaded_model.eval()  # 切换到评估模式 / 評価モードに切り替える
    x = torch.rand(1, 4, device=device)  # 创建测试输入 / テスト入力を作る
    y = loaded_model(x)  # 用读取后的模型预测 / 読み込んだモデルで予測する

    print(f"Using device: {device}")  # 输出设备 / デバイスを出力する
    print(f"saved to: {save_path}")  # 输出保存路径 / 保存パスを出力する
    print(f"output: {y}")  # 输出预测结果 / 予測結果を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
