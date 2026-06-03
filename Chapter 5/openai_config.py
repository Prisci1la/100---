"""
Chapter 5脚本的共享配置。 / Chapter 5スクリプトの共有設定。

创建客户端前，从本目录的.env文件加载环境变量。 / クライアント作成前に、このディレクトリの.envファイルから環境変数を読み込みます。
"""

import os
from pathlib import Path

from openai import OpenAI


DEFAULT_MODEL = "gpt-5"
DEFAULT_MAX_COMPLETION_TOKENS = 8192


def load_local_env():  # 从Chapter 5/.env读取简单的KEY=VALUE配置 / Chapter 5/.envから単純なKEY=VALUE設定を読み込む

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ.setdefault(key, value)


def create_openai_client():  # 加载Chapter 5/.env后创建客户端 / Chapter 5/.envを読み込んでからクライアントを作成する

    load_local_env()
    return OpenAI()
