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
    print(f"[DEBUG][llm_client] 准备调用模型：{model}")
    print(f"[DEBUG][llm_client] messages 数量：{len(messages)}")
    print(f"[DEBUG][llm_client] temperature：{temperature}")

    typed_messages = cast(
        Iterable[ChatCompletionMessageParam],
        messages,
    )

    response: Any = client.chat.completions.create(
        model=model,
        messages=typed_messages,
        temperature=temperature,
    )

    print(f"[DEBUG][llm_client] 模型调用完成：{model}")

    content = response.choices[0].message.content
    print(f"[DEBUG][llm_client] 返回内容长度：{len(content or '')}")

    return content or ""