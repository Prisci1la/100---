'''
knock72.py: Bag of words模型的构建 / Bag of wordsモデルの構築

使用torch.nn.Module继承类，基于平均词向量实现二分类模型。
/ torch.nn.Moduleを継承し、平均単語ベクトルによる二値分類モデルを実装する。
'''

import torch  # 导入PyTorch / PyTorchを導入する
from torch import nn  # 导入神经网络模块 / ニューラルネットワークモジュールを導入する

from chapter8_utils import build_embedding_resources, parse_common_args  # 导入embedding工具 / embeddingツールを導入する


class BoWClassifier(nn.Module):  # BoW二分类模型 / BoW二値分類モデル
    def __init__(self, embedding_matrix, freeze_embeddings=True):  # 初始化模型 / モデルを初期化する
        super().__init__()  # 调用父类初始化 / 親クラスを初期化する
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)  # 将numpy矩阵转为Tensor / numpy行列をTensorへ変換する
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=freeze_embeddings, padding_idx=0)  # 创建embedding层 / embedding層を作る
        self.linear = nn.Linear(embedding_tensor.shape[1], 1)  # 创建线性分类层 / 線形分類層を作る

    def forward(self, input_ids, lengths=None):  # 前向计算 / 順伝播を行う
        embedded = self.embedding(input_ids)  # 将token ID转换为词向量 / token IDを単語ベクトルへ変換する
        mask = (input_ids != 0).unsqueeze(-1)  # 创建PAD以外的掩码 / PAD以外のマスクを作る
        summed = (embedded * mask).sum(dim=1)  # 对有效词向量求和 / 有効な単語ベクトルを合計する
        if lengths is None:  # 如果没有传入长度 / 長さが渡されていない場合
            lengths = mask.sum(dim=1).clamp(min=1).squeeze(-1)  # 从掩码计算长度 / マスクから長さを計算する
        averaged = summed / lengths.to(embedded.device).unsqueeze(-1).clamp(min=1)  # 计算平均词向量 / 平均単語ベクトルを計算する
        return self.linear(averaged)  # 输出logit / logitを出力する


def main():  # 定义主函数 / メイン関数を定義する
    args = parse_common_args("knock72: build BoW classifier")  # 解析命令行参数 / コマンドライン引数を解析する
    embedding_matrix, _token_to_id, _id_to_token = build_embedding_resources(  # 构建embedding矩阵 / embedding行列を構築する
        vector_path=args.vector_path,  # 指定词向量路径 / 単語ベクトルパスを指定する
        binary=not args.text_vectors,  # 指定词向量格式 / 単語ベクトル形式を指定する
        max_vocab=args.max_vocab,  # 限制词表大小 / 語彙数を制限する
    )
    model = BoWClassifier(embedding_matrix, freeze_embeddings=True)  # 创建冻结embedding的BoW模型 / embeddingを固定したBoWモデルを作る
    dummy_input = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)  # 创建测试输入 / テスト入力を作る
    dummy_lengths = torch.tensor([3, 2], dtype=torch.long)  # 创建真实长度 / 実際の長さを作る
    logits = model(dummy_input, dummy_lengths)  # 前向计算 / 順伝播を行う

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 72: BoW Classifier")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(model)  # 输出模型结构 / モデル構造を出力する
    print(f"logits shape: {logits.shape}")  # 输出logit形状 / logit形状を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

r'''
运行结果: / 実行結果:
==================================================
Knock 72: BoW Classifier
==================================================
BoWClassifier(
  (embedding): Embedding(50001, 300, padding_idx=0)
  (linear): Linear(in_features=300, out_features=1, bias=True)
)
logits shape: torch.Size([2, 1])
'''
