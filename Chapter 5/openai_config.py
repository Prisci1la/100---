"""
Chapter 5脚本的共享配置。 / Chapter 5スクリプトの共有設定。

创建客户端前，从本目录的.env文件加载环境变量。 / クライアント作成前に、このディレクトリの.envファイルから環境変数を読み込みます。
"""

import os  # 导入环境变量模块 / 環境変数モジュールをインポート
from pathlib import Path  # 导入路径处理类 / パス処理用クラスをインポート

from openai import OpenAI  # 导入OpenAI客户端类 / OpenAIクライアントクラスをインポート


DEFAULT_MODEL = "gpt-5"  # 定义默认使用模型 / 既定で使用するモデルを定義
DEFAULT_MAX_COMPLETION_TOKENS = 8192  # 定义默认最大生成token数 / 既定の最大生成トークン数を定義


def load_local_env():  # 从Chapter 5/.env读取简单的KEY=VALUE配置 / Chapter 5/.envから単純なKEY=VALUE設定を読み込む

    env_path = Path(__file__).with_name(".env")  # 定位Chapter 5目录下的.env文件 / Chapter 5ディレクトリ内の.envファイルを特定
    if not env_path.exists():  # 如果.env不存在则直接返回 / .envが存在しなければそのまま返す
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():  # 按行读取.env内容 / .env内容を行ごとに読み込む
        line = line.strip()  # 去掉行首行尾空白 / 行頭行末の空白を除去
        if not line or line.startswith("#") or "=" not in line:  # 跳过空行、注释行和非键值行 / 空行、コメント行、キー値形式でない行をスキップ
            continue

        key, value = line.split("=", 1)  # 只按第一个等号拆分键和值 / 最初の等号だけでキーと値に分割
        key = key.strip()  # 清理键名空白 / キー名の空白を除去
        value = value.strip().strip('"').strip("'")  # 清理值的空白和外层引号 / 値の空白と外側の引用符を除去

        if key:  # 键名非空时写入环境变量 / キー名が空でなければ環境変数に設定
            os.environ.setdefault(key, value)  # 只在尚未存在时设置，避免覆盖外部环境变量 / 既存の外部環境変数を上書きしないよう未設定時だけ設定


def create_openai_client():  # 加载Chapter 5/.env后创建客户端 / Chapter 5/.envを読み込んでからクライアントを作成する

    load_local_env()  # 创建客户端前加载本地.env / クライアント作成前にローカル.envを読み込む
    return OpenAI()  # 返回OpenAI客户端实例 / OpenAIクライアントインスタンスを返す
