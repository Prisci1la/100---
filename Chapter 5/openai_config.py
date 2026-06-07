"""
Shared configuration for Chapter 5 scripts.

Loads environment variables from the .env file in this directory before
creating a client.
"""

import os
from pathlib import Path

from openai import OpenAI


DEFAULT_MODEL = "gpt-5"
DEFAULT_MAX_COMPLETION_TOKENS = 8192


def load_local_env():
    """Load simple KEY=VALUE pairs from Chapter 5/.env."""

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


def create_openai_client():
    """Create a client after loading Chapter 5/.env."""

    load_local_env()
    return OpenAI()
