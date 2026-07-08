'''
knock81.py: マスクの予測 / マスクの予測

"The movie was full of [MASK]."の[MASK]に入る最適なtokenを求める。
/ "The movie was full of [MASK]."の[MASK]に入る最適なtokenを求める。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # 导入PyTorch / PyTorchを導入する
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 导入BERT模型和tokenizer / BERTモデルとtokenizerを導入する


MODEL_NAME = "google-bert/bert-base-uncased"  # 指定使用的BERT模型 / 使用するBERTモデルを指定する
MASK_SENTENCE = "The movie was full of [MASK]."  # 问题81和82的句子 / 問題81と82の文


def move_encoding_to_device(encoding, device):  # 将tokenizer输出移动到设备 / tokenizer出力をデバイスへ移す
    return {key: value.to(device) for key, value in encoding.items()}  # 移动字典中的Tensor / 辞書内Tensorを移動する


def predict_mask_tokens(tokenizer, model, sentence=MASK_SENTENCE, top_k=1, device=None):  # 预测[MASK]位置的token / [MASK]位置を予測する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device  # 自动选择设备 / デバイスを自動選択する
    encoding = tokenizer(sentence, return_tensors="pt")  # 将句子转为模型输入 / 文をモデル入力へ変換する
    input_ids = encoding["input_ids"]  # 取出token ID / token IDを取り出す
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)  # 找到[MASK]位置 / [MASK]位置を探す
    if len(mask_positions) != 1:  # 本题要求句子中只有一个[MASK] / この課題では[MASK]は1つ
        raise ValueError("The sentence must contain exactly one [MASK] token.")  # 数量不对时抛错 / 数が違う場合はエラー
    encoding = move_encoding_to_device(encoding, device)  # 输入移动到设备 / 入力をデバイスへ移動する
    mask_index = mask_positions[0, 1].item()  # 取得[MASK]在序列中的位置 / 系列内の[MASK]位置を得る
    with torch.no_grad():  # 预测时不计算梯度 / 予測時は勾配を計算しない
        logits = model(**encoding).logits[0, mask_index]  # 取出[MASK]位置的logit / [MASK]位置のlogitを取る
        probabilities = torch.softmax(logits, dim=-1)  # 转换为概率 / 確率へ変換する
        top_probs, top_ids = torch.topk(probabilities, k=top_k)  # 取概率最高的token / 上位tokenを取る
    results = []  # 初始化结果列表 / 結果リストを初期化する
    for token_id, probability in zip(top_ids.tolist(), top_probs.tolist()):  # 遍历top-k结果 / top-k結果を走査する
        token = tokenizer.convert_ids_to_tokens(token_id)  # ID转换为token / IDをtokenへ変換する
        results.append((token, probability))  # 保存token和概率 / tokenと確率を保存する
    return results  # 返回预测结果 / 予測結果を返す


def main():  # 定义主函数 / メイン関数を定義する
    parser = argparse.ArgumentParser(description="knock81: predict one masked token")  # 创建参数解析器 / 引数パーサーを作る
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name")  # 模型名称 / モデル名
    parser.add_argument("--sentence", default=MASK_SENTENCE, help="sentence containing one [MASK]")  # 含[MASK]的句子 / [MASK]を含む文
    args = parser.parse_args()  # 解析命令行参数 / コマンドライン引数を解析する

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备 / デバイスを取得する
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 读取tokenizer / tokenizerを読み込む
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device).eval()  # 读取Masked LM / Masked LMを読み込む
    token, probability = predict_mask_tokens(tokenizer, model, args.sentence, top_k=1, device=device)[0]  # 预测概率最高的token / 確率最大のtokenを予測する

    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print("Knock 81: Mask Prediction")  # 输出标题 / タイトルを出力する
    print("=" * 50)  # 输出分隔线 / 区切り線を出力する
    print(f"device: {device}")  # 输出设备 / デバイスを出力する
    print(f"sentence: {args.sentence}")  # 输出输入句子 / 入力文を出力する
    print(f"prediction: {token}")  # 输出预测token / 予測tokenを出力する
    print(f"probability: {probability:.6f}")  # 输出概率 / 確率を出力する


if __name__ == "__main__":  # 直接运行时执行 / 直接実行された場合だけ動かす
    main()  # 调用主函数 / メイン関数を呼び出す

'''
実行結果 / 运行结果:
==================================================
Knock 81: Mask Prediction
==================================================
device: cuda
sentence: The movie was full of [MASK].
prediction: fun
probability: 0.107119
'''

