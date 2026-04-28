from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class ContextManager:
    """
    一个最小上下文管理器。

    功能：
    1. 维护 system / user / assistant 消息列表
    2. 支持保存和加载历史会话
    3. 支持按字符数做简单截断，避免上下文无限变长
    """

    system_prompt: str
    max_chars: int = 12000
    keep_last_messages: int = 12
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

    def add_user(self, content: str) -> None:
        self.messages.append({
            "role": "user",
            "content": content,
        })

    def add_assistant(self, content: str) -> None:
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

    def build_messages(self) -> list[Message]:
        """
        返回真正发送给模型的 messages。

        策略：
        - system prompt 永远保留
        - 优先保留最近 keep_last_messages 条消息
        - 如果仍然超过 max_chars，就继续从旧到新删除
        """
        if not self.messages:
            return [{"role": "system", "content": self.system_prompt}]

        system_msg = self.messages[0]
        recent_msgs = self.messages[1:][-self.keep_last_messages:]

        result = [system_msg] + recent_msgs

        while self._total_chars(result) > self.max_chars and len(result) > 2:
            # 保留 system，删除最旧的一条普通消息
            del result[1]

        return result

    def clear(self) -> None:
        self.messages = [{
            "role": "system",
            "content": self.system_prompt,
        }]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "system_prompt": self.system_prompt,
            "max_chars": self.max_chars,
            "keep_last_messages": self.keep_last_messages,
            "messages": self.messages,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        system_prompt: str,
        max_chars: int = 12000,
        keep_last_messages: int = 12,
    ) -> "ContextManager":
        path = Path(path)

        if not path.exists():
            return cls(
                system_prompt=system_prompt,
                max_chars=max_chars,
                keep_last_messages=keep_last_messages,
            )

        data = json.loads(path.read_text(encoding="utf-8"))

        return cls(
            system_prompt=data.get("system_prompt", system_prompt),
            max_chars=data.get("max_chars", max_chars),
            keep_last_messages=data.get("keep_last_messages", keep_last_messages),
            messages=data.get("messages", []),
        )

    @staticmethod
    def _total_chars(messages: list[Message]) -> int:
        return sum(len(m["content"]) for m in messages)