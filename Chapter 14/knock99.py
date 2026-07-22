'''
knock99.py: 翻訳サーバの構築 / 翻訳サーバの構築

Flaskで、日本語文を入力すると翻訳結果を表示するデモサーバを構築する。
/ Flaskで、日本語文を入力すると翻訳結果を表示するデモサーバを構築する。
'''

import argparse  # 命令行参数解析库 / コマンドライン引数解析ライブラリ

import torch  # PyTorch / PyTorch
from flask import Flask, request  # Flask工具 / Flaskツール

from chapter14_utils import BASE_CHECKPOINT, beam_search_decode, encode, get_device, get_ja_tagger, load_checkpoint, tokenize_ja  # 共用工具 / 共通ツール


HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Translation Demo</title></head>
<body>
  <h1>Japanese-English Translation</h1>
  <form method="post">
    <textarea name="sentence" rows="4" cols="80">{sentence}</textarea><br>
    <button type="submit">Translate</button>
  </form>
  <h2>Result</h2>
  <pre>{result}</pre>
</body>
</html>
"""  # 简单HTML模板 / 簡単なHTMLテンプレート


def create_app(checkpoint, beam_size=5, max_len=80):  # 创建Flask app / Flask appを作る
    device = get_device()  # 设备 / デバイス
    model, src_vocab, tgt_vocab, _config = load_checkpoint(checkpoint, device)  # 读取模型 / モデルを読む
    tagger = get_ja_tagger()  # 日语tagger / 日本語tagger
    app = Flask(__name__)  # 创建app / appを作る

    @app.route("/", methods=["GET", "POST"])  # 主页route / トップroute
    def index():  # 处理请求 / リクエストを処理する
        sentence = request.form.get("sentence", "")  # 取得输入 / 入力を取得する
        result = ""  # 初始化结果 / 結果を初期化
        if sentence:  # 有输入时翻译 / 入力がある場合は翻訳
            tokens = tokenize_ja(sentence, tagger)  # 分词 / 分かち書き
            src_ids = torch.tensor(encode(tokens, src_vocab), dtype=torch.long).unsqueeze(1)  # 编码 / 符号化
            out_tokens = beam_search_decode(model, src_ids, src_vocab, tgt_vocab, beam_size=beam_size, max_len=max_len, device=device)  # 翻译 / 翻訳
            result = " ".join(out_tokens)  # 连接输出 / 出力を結合
        return HTML.format(sentence=sentence, result=result)  # 返回页面 / ページを返す

    return app  # 返回app / appを返す


def main():  # 主函数 / メイン関数
    parser = argparse.ArgumentParser(description="knock99: Flask translation server")  # 参数解析 / 引数解析
    parser.add_argument("--checkpoint", default=str(BASE_CHECKPOINT), help="model checkpoint")  # checkpoint / checkpoint
    parser.add_argument("--beam-size", type=int, default=5, help="beam size")  # beam大小 / beamサイズ
    parser.add_argument("--host", default="127.0.0.1", help="host")  # host / host
    parser.add_argument("--port", type=int, default=5000, help="port")  # port / port
    args = parser.parse_args()  # 解析 / 解析する
    app = create_app(args.checkpoint, beam_size=args.beam_size)  # 创建app / appを作る
    app.run(host=args.host, port=args.port)  # 启动服务器 / サーバを起動する


if __name__ == "__main__":  # 直接运行 / 直接実行
    main()  # 调用主函数 / メイン関数を呼ぶ


# AIX実行結果メモ (2026-07-19, log: ~/100knock/logs/ch14_knock99_smoke_g24.log)
# Flask app smoke test hit the local service and verified a result was returned.
# smoke test: status_code=200, contains_result=True.
# checkpoint used: Chapter 14/checkpoints/domain_adapted_mt_limited.pt.
