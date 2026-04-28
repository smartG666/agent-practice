from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

BASE_DIR = Path(__file__).parent

FLASH_MODEL = "deepseek-v4-flash"
PRO_MODEL = "deepseek-v4-pro"


def create_client() -> OpenAI:
    """
    创建 DeepSeek API 客户端。

    要求：
    - 项目根目录下存在 .env
    - .env 中包含 DEEPSEEK_API_KEY
    """
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(f"未找到 DEEPSEEK_API_KEY，请检查：{env_path}")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
) -> str:
    """
    调用 DeepSeek Chat Completion 接口，并返回文本内容。

    说明：
    - OpenAI SDK 的类型定义要求 messages 是 Iterable[ChatCompletionMessageParam]
    - 但项目内部为了简单，统一使用 list[dict[str, str]]
    - 这里用 cast 告诉 Pylance：这些 dict 符合 ChatCompletionMessageParam 结构
    """
    typed_messages = cast(
        Iterable[ChatCompletionMessageParam],
        messages,
    )

    response: Any = client.chat.completions.create(
        model=model,
        messages=typed_messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    return content or ""
