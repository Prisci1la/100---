'''
knock83.py: CLSトークンによる文ベクトル / CLSトークンによる文ベクトル

最終層の[CLS]トークン埋め込みで文ベクトルを作り、全ペアのコサイン類似度を求める。
/ 最終層の[CLS]トークン埋め込みで文ベクトルを作り、全ペアのコサイン類似度を求める。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ
from itertools import combinations  # 组合生成工具 / 組み合わせ生成ツール

import torch  # 导入PyTorch / PyTorchを導入する
from torch.nn import functional as F  # 导入函数式API / 関数型APIを導入する
from transformers import AutoModel, AutoTokenizer  # 导入BERT主体模型和tokenizer / BERT本体モデルとtokenizerを導入する


MODEL_NAME = "google-bert/bert-base-uncased"  # 指定使用的BERT模型 / 使用するBERTモデルを指定する
SIMILARITY_SENTENCES = [  # 问题83和84的句子列表 / 問題83と84の文リスト
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def move_encoding_to_device(encoding, device):  # 将tokenizer输出移动到设备 / tokenizer出力をデバイスへ移す
    return {key: value.to(device) for key, value in encoding.items()}  # 移动字典中的Tensor / 辞書内Tensorを移動する


def encode_sentences(tokenizer, sentences=SIMILARITY_SENTENCES, device=None):  # 批量编码句子 / 複数文をエンコードする
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )  # 调用tokenizer批量编码 / tokenizerでバッチ符号化する
    return move_encoding_to_device(encoding, device)  # 编码结果移动到设备 / 符号化結果をデバイスへ移す


def get_cls_embeddings(tokenizer, model, sentences=SIMILARITY_SENTENCES, device=None):  # 取得[CLS]句向量 / [CLS]文ベクトルを取得する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = encode_sentences(tokenizer, sentences, device=device)  # 编码句子 / 文を符号化する
    with torch.no_grad():  # 不计算梯度 / 勾配を計算しない
        outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])  # 运行BERT / BERTを実行する
    return outputs.last_hidden_state[:, 0, :]  # 返回最后一层[CLS]向量 / 最終層の[CLS]ベクトルを返す


def cosine_similarity_pairs(embeddings, sentences=SIMILARITY_SENTENCES):  # 计算所有句子对的余弦相似度 / 全ペアのcos類似度を計算する
    normalized = F.normalize(embeddings, p=2, dim=1)  # L2归一化 / L2正規化する
    results = []  # 初始化结果列表 / 結果リストを初期化する
    for left, right in combinations(range(len(sentences)), 2):  # 遍历所有句子组合 / 全組み合わせを走査する
        score = torch.dot(normalized[left], normalized[right]).item()  # 计算两个向量点积 / 2ベクトルの内積を計算する
        results.append((sentences[left], sentences[right], score))  # 保存句子对和分数 / 文ペアとスコアを保存する
    return results  # 返回相似度结果 / 類似度結果を返す


def print_similarity_results(results):  # 打印相似度结果 / 類似度結果を表示する
    for left_sentence, right_sentence, score in results:  # 遍历结果 / 結果を走査する
        print(f"{score:.6f}\t{left_sentence}\t|\t{right_sentence}")  # 输出分数和句子对 / スコアと文ペアを出力する


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock83: sentence vectors from [CLS] token")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名称 / モデル名
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备 / デバイスを取得する
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 读取tokenizer / tokenizerを読み込む
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()  # 读取BERT编码器 / BERTエンコーダを読み込む
    embeddings = get_cls_embeddings(tokenizer, model, SIMILARITY_SENTENCES, device=device)  # 取得[CLS]向量 / [CLS]ベクトルを取得する
    results = cosine_similarity_pairs(embeddings, SIMILARITY_SENTENCES)  # 计算全组合相似度 / 全組み合わせの類似度を計算する

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 83: CLS Sentence Vectors")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"device: {device}")  # 输出设备 / デバイスを出力する
    print_similarity_results(results)  # 打印相似度结果 / 類似度結果を表示する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
実行結果 / 运行结果:
==================================================
Knock 83: CLS Sentence Vectors
==================================================
device: cuda
0.988061	The movie was full of fun.	|	The movie was full of excitement.
0.955766	The movie was full of fun.	|	The movie was full of crap.
0.947532	The movie was full of fun.	|	The movie was full of rubbish.
0.954127	The movie was full of excitement.	|	The movie was full of crap.
0.948664	The movie was full of excitement.	|	The movie was full of rubbish.
0.980693	The movie was full of crap.	|	The movie was full of rubbish.
'''

