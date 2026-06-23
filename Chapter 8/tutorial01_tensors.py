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
    print(f"tensor:\n{tensor}")  # 输出原始张量 / 元テンソルを出力する
    print(f"shape: {tensor.shape}, dtype: {tensor.dtype}")  # 输出形状和类型 / 形状と型を出力する
    print(f"ones:\n{ones}")  # 输出全1张量 / 1テンソルを出力する
    print(f"random:\n{random_tensor}")  # 输出随机张量 / 乱数テンソルを出力する
    print(f"product device: {product.device}")  # 输出运算设备 / 演算デバイスを出力する
    print(f"product:\n{product}")  # 输出矩阵乘法结果 / 行列積の結果を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
