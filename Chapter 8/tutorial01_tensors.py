'''
tutorial01_tensors.py: PyTorch入门[1]张量 / PyTorch入門[1]テンソル

演示张量创建、属性确认、GPU移动和基本运算。
/ テンソル作成、属性確認、GPU移動、基本演算を確認する。
'''

import torch  # 导入PyTorch / PyTorchを導入する


def main():  # 定义主函数 / メイン関数を定義する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU / GPUまたはCPUを自動選択する
    data = [[1, 2], [3, 4]]  # 准备Python列表 / Pythonリストを用意する
    tensor = torch.tensor(data, dtype=torch.float32)  # 从列表创建张量 / リストからテンソルを作る
    ones = torch.ones_like(tensor)  # 创建同形状全1张量 / 同じ形の1テンソルを作る
    random_tensor = torch.rand_like(tensor)  # 创建同形状随机张量 / 同じ形の乱数テンソルを作る
    gpu_tensor = tensor.to(device)  # 将张量移动到设备 / テンソルをデバイスへ移す
    product = gpu_tensor @ gpu_tensor.T  # 计算矩阵乘法 / 行列積を計算する

    print(f"Using device: {device}")  # 输出设备 / デバイスを出力する
    print(f"shape: {tensor.shape}, dtype: {tensor.dtype}")  # 输出形状和类型 / 形状と型を出力する
    print(f"ones shape: {ones.shape}")  # 输出全1张量形状 / 1テンソルの形状を出力する
    print(f"random shape: {random_tensor.shape}")  # 输出随机张量形状 / 乱数テンソルの形状を出力する
    print(f"product device: {product.device}")  # 输出运算设备 / 演算デバイスを出力する
    print(f"product shape: {product.shape}")  # 输出矩阵乘法结果形状 / 行列積の形状を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
运行结果: / 実行結果:
Using device: cuda
shape: torch.Size([2, 2]), dtype: torch.float32
ones shape: torch.Size([2, 2])
random shape: torch.Size([2, 2])
product device: cuda:0
product shape: torch.Size([2, 2])
'''
