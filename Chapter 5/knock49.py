"""
knock49.py: 分词/标记化 / トークン化

对给定文本进行实际标记化，并确认标记数和标记ID。 / 与えられたテキストを実際にトークン化し、トークン数とトークンIDを確認します。
"""

from openai_config import DEFAULT_MODEL


TEXT = """吾輩は猫である。名前はまだ無い。

どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"""


def get_encoding(model):  # 返回当前设置对应的分词器编码 / 設定に対応するトークナイザーのエンコーディングを返す

    try:
        import warnings

        warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktokenがインストールされていません。"
            " `.venv/bin/python -m pip install tiktoken` を実行してください。"
        ) from exc

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens():  # 标记化文本并输出详细信息 / テキストをトークン化して詳細情報を出力する

    encoding = get_encoding(DEFAULT_MODEL)
    token_ids = encoding.encode(TEXT)

    print("=" * 60)
    print("knock49: トークン化（トークン数計測）")
    print("=" * 60)

    print("\n【テキスト情報】")
    print(f"文字数: {len(TEXT)}")
    print(f"トークン数: {len(token_ids)}")

    print("\n【先頭20トークンID】")
    print(token_ids[:20])

    print("\n【先頭20トークンの復元】")
    for token_id in token_ids[:20]:
        token_bytes = encoding.decode_single_token_bytes(token_id)
        token_text = token_bytes.decode("utf-8", errors="replace")
        print(f"{token_id}: {token_text!r} bytes={token_bytes!r}")

    print("\n" + "=" * 60)

    return len(token_ids)


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
