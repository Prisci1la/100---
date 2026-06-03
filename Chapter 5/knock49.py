"""
knock49.py: 分词/标记化 / トークン化

对给定文本进行实际标记化，并确认标记数和标记ID。 / 与えられたテキストを実際にトークン化し、トークン数とトークンIDを確認します。
"""

from openai_config import DEFAULT_MODEL  # 导入默认模型名 / 既定のモデル名をインポート


TEXT = (  # 定义用于token计数的示例文本 / トークン数計測用のサンプルテキストを定義
    """吾輩は猫である。名前はまだ無い。

どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"""
)


def get_encoding(model):  # 返回当前设置对应的分词器编码 / 設定に対応するトークナイザーのエンコーディングを返す

    try:  # 尝试导入分词库 / トークナイザーライブラリのインポートを試行
        import warnings  # 导入警告控制模块 / 警告制御モジュールをインポート

        warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")  # 忽略urllib3的OpenSSL警告 / urllib3のOpenSSL警告を無視
        import tiktoken  # 导入tiktoken分词库 / tiktokenトークナイザーをインポート
    except ImportError as exc:
        raise RuntimeError(  # tiktoken未安装时抛出易懂错误 / tiktoken未インストール時に分かりやすいエラーを投げる
            "tiktokenがインストールされていません。"
            " `.venv/bin/python -m pip install tiktoken` を実行してください。"
        ) from exc

    try:  # 尝试获取指定模型对应的编码 / 指定モデルに対応するエンコーディングを取得
        return tiktoken.encoding_for_model(model)  # 返回模型专用编码 / モデル専用エンコーディングを返す
    except KeyError:
        return tiktoken.get_encoding("o200k_base")  # 找不到模型时使用备用编码 / モデルが見つからない場合は代替エンコーディングを使う


def count_tokens():  # 标记化文本并输出详细信息 / テキストをトークン化して詳細情報を出力する

    encoding = get_encoding(DEFAULT_MODEL)  # 获取默认模型对应的分词编码 / 既定モデルに対応するトークンエンコーディングを取得
    token_ids = encoding.encode(TEXT)  # 将文本编码为token ID列表 / テキストをトークンIDリストにエンコード

    print("=" * 60)  # 输出分割线 / 区切り線を出力
    print("knock49: トークン化（トークン数計測）")  # 输出任务标题 / タスクのタイトルを出力
    print("=" * 60)  # 输出分割线 / 区切り線を出力

    print("\n【テキスト情報】")  # 输出文本信息标签 / テキスト情報ラベルを出力
    print(f"文字数: {len(TEXT)}")  # 输出字符数 / 文字数を出力
    print(f"トークン数: {len(token_ids)}")  # 输出token数 / トークン数を出力

    print("\n【先頭20トークンID】")  # 输出前20个token ID标签 / 先頭20トークンIDラベルを出力
    print(token_ids[:20])  # 输出前20个token ID / 先頭20個のトークンIDを出力

    print("\n【先頭20トークンの復元】")  # 输出token复原标签 / トークン復元ラベルを出力
    for token_id in token_ids[:20]:  # 遍历前20个token ID / 先頭20個のトークンIDを順に処理
        token_bytes = encoding.decode_single_token_bytes(token_id)  # 将单个token ID解码为字节 / 単一トークンIDをバイト列にデコード
        token_text = token_bytes.decode("utf-8", errors="replace")  # 将字节尽量解码为文本 / バイト列を可能な限りテキストにデコード
        print(f"{token_id}: {token_text!r} bytes={token_bytes!r}")  # 输出token ID、文本和字节 / トークンID、テキスト、バイト列を出力

    print("\n" + "=" * 60)  # 输出分割线 / 区切り線を出力

    return len(token_ids)  # 返回token总数 / トークン総数を返す


if __name__ == "__main__":
    count_tokens()

"""
执行结果 / 実行結果:
============================================================
knock49: トークン化（トークン数計測）
============================================================

【テキスト情報】
文字数: 450
トークン数: 364

【先頭20トークンID】
[129857, 20000, 102, 5205, 48091, 4344, 73977, 788, 118857, 5205, 166599, 10205, 3826, 1497, 18524, 8468, 4344, 5883, 9472, 5598]

【先頭20トークンの復元】
129857: '吾' bytes=b'\xe5\x90\xbe'
20000: '�' bytes=b'\xe8\xbc'
102: '�' bytes=b'\xa9'
5205: 'は' bytes=b'\xe3\x81\xaf'
48091: '猫' bytes=b'\xe7\x8c\xab'
4344: 'で' bytes=b'\xe3\x81\xa7'
73977: 'ある' bytes=b'\xe3\x81\x82\xe3\x82\x8b'
788: '。' bytes=b'\xe3\x80\x82'
118857: '名前' bytes=b'\xe5\x90\x8d\xe5\x89\x8d'
5205: 'は' bytes=b'\xe3\x81\xaf'
166599: 'まだ' bytes=b'\xe3\x81\xbe\xe3\x81\xa0'
10205: '無' bytes=b'\xe7\x84\xa1'
3826: 'い' bytes=b'\xe3\x81\x84'
1497: '。\n\n' bytes=b'\xe3\x80\x82\n\n'
18524: 'ど' bytes=b'\xe3\x81\xa9'
8468: 'こ' bytes=b'\xe3\x81\x93'
4344: 'で' bytes=b'\xe3\x81\xa7'
5883: '生' bytes=b'\xe7\x94\x9f'
9472: 'れ' bytes=b'\xe3\x82\x8c'
5598: 'た' bytes=b'\xe3\x81\x9f'

============================================================
"""
