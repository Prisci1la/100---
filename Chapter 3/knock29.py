"""
29. 国旗画像のURLを取得する / 获取国旗图片的 URL
テンプレートの内容を利用し、国旗画像の URL を取得する。
利用模板的内容获取国旗图片的 URL。

MediaWiki API の imageinfo を呼び出して、ファイル参照を URL に変換する。
通过调用 MediaWiki API 的 imageinfo，将文件引用转换为 URL。

API エンドポイント / API 端点：
  https://commons.wikimedia.org/w/api.php
"""

import urllib.parse
import urllib.request
import json

from knock20 import load_uk_article
from knock28 import clean_basic_info


# MediaWiki Commons API のエンドポイント / MediaWiki Commons API 端点
API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"


def get_image_url(filename: str) -> str:
    """
    ファイル名から画像 URL を取得する。
    根据文件名获取图片 URL。
    """
    # クエリパラメータ / 查询参数
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        # 「File:」プレフィックスが必要 / 需要加上 "File:" 前缀
        "titles": f"File:{filename}",
    }

    # URL を構築 / 构建 URL
    url = f"{API_ENDPOINT}?{urllib.parse.urlencode(params)}"

    # User-Agent はマナーとして設定 / 设置 User-Agent 是基本礼仪
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NLP100-Knock29/1.0 (educational use)"},
    )

    # API を呼び出し / 调用 API
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # 応答から URL を取り出す / 从响应中取出 URL
    # pages は辞書で、キーは page id（不定）/ pages 是字典，键是 page id（不定）
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page["imageinfo"][0]["url"]


if __name__ == "__main__":
    # 本文取得 / 取得正文
    text = load_uk_article()

    # 基礎情報を取得（28 のクリーン版）/ 取得基础信息（28 的清理版）
    info = clean_basic_info(text)

    # 「国旗画像」フィールド / "国旗画像" 字段
    flag_filename = info.get("国旗画像")
    if not flag_filename:
        raise ValueError("国旗画像フィールドが見つかりません / 没找到国旗画像字段")

    print(f"ファイル名 / 文件名: {flag_filename}")

    # API で URL に変換 / 通过 API 转换为 URL
    url = get_image_url(flag_filename)
    print(f"URL: {url}")

'''
ファイル名 / 文件名: Flag of the United Kingdom.svg
URL: https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg
'''