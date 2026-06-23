'''
tutorial03_transforms.py: PyTorch入门[3]数据变换 / PyTorch入門[3]データ変換

演示将原始样本转换为Tensor，并对数值进行标准化。
/ 元サンプルをTensorへ変換し、数値を標準化する流れを確認する。
'''

import torch  # 导入PyTorch / PyTorchを導入する


class NormalizeTransform:  # 定义标准化变换类 / 標準化変換クラスを定義する
    def __init__(self, mean, std):  # 初始化平均值和标准差 / 平均と標準偏差を初期化する
        self.mean = torch.tensor(mean, dtype=torch.float32)  # 保存平均值 / 平均を保存する
        self.std = torch.tensor(std, dtype=torch.float32)  # 保存标准差 / 標準偏差を保存する

    def __call__(self, sample):  # 使实例可以像函数一样调用 / インスタンスを関数のように呼べるようにする
        tensor = torch.tensor(sample, dtype=torch.float32)  # 将输入转换为Tensor / 入力をTensorへ変換する
        return (tensor - self.mean) / self.std  # 返回标准化结果 / 標準化結果を返す


def main():  # 定义主函数 / メイン関数を定義する
    sample = [10.0, 20.0, 30.0]  # 准备原始样本 / 元サンプルを用意する
    transform = NormalizeTransform(mean=[10.0, 10.0, 10.0], std=[10.0, 10.0, 10.0])  # 创建变换器 / 変換器を作る
    transformed = transform(sample)  # 执行变换 / 変換を実行する
    print(f"original: {sample}")  # 输出原始数据 / 元データを出力する
    print(f"transformed: {transformed}")  # 输出变换结果 / 変換結果を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す
