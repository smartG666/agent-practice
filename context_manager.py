from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


# 是否开启 context_manager 调试日志
# 不想看日志时，改成 False 即可
DEBUG = True


def debug_log(message: str) -> None:
    """
    打印 context_manager 模块的调试日志。
    """
    if DEBUG:
        print(f"[DEBUG][context_manager] {message}")


def preview_text(text: str, limit: int = 80) -> str:
    """
    生成文本预览，避免日志输出太长。
    """
    return text.replace("\n", " ")[:limit]


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
        debug_log("__post_init__ 开始")
        debug_log(f"当前 messages 数量：{len(self.messages)}")
        debug_log(f"max_chars：{self.max_chars}")
        debug_log(f"keep_last_messages：{self.keep_last_messages}")

        if not self.messages:
            debug_log("messages 为空，自动加入 system prompt")
            self.messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        else:
            debug_log("messages 非空，使用已有历史上下文")

        debug_log(f"__post_init__ 完成，messages 数量：{len(self.messages)}")

    def add_user(self, content: str) -> None:
        debug_log("准备加入 user 消息")
        debug_log(f"user 消息长度：{len(content)}")
        debug_log(f"user 消息预览：{preview_text(content)!r}")

        self.messages.append({
            "role": "user",
            "content": content,
        })

        debug_log(f"user 消息加入完成，当前 messages 数量：{len(self.messages)}")

    def add_assistant(self, content: str) -> None:
        debug_log("准备加入 assistant 消息")
        debug_log(f"assistant 消息长度：{len(content)}")
        debug_log(f"assistant 消息预览：{preview_text(content)!r}")

        self.messages.append({
            "role": "assistant",
            "content": content,
        })

        debug_log(f"assistant 消息加入完成，当前 messages 数量：{len(self.messages)}")

    def build_messages(self) -> list[Message]:
        """
        返回真正发送给模型的 messages。

        策略：
        - system prompt 永远保留
        - 优先保留最近 keep_last_messages 条消息
        - 如果仍然超过 max_chars，就继续从旧到新删除
        """
        debug_log("开始 build_messages")
        debug_log(f"原始 messages 数量：{len(self.messages)}")

        if not self.messages:
            debug_log("messages 为空，返回单独的 system prompt")
            return [{"role": "system", "content": self.system_prompt}]

        system_msg = self.messages[0]
        normal_msgs = self.messages[1:]

        debug_log(f"普通消息数量，不含 system：{len(normal_msgs)}")

        recent_msgs = normal_msgs[-self.keep_last_messages:]

        debug_log(
            "根据 keep_last_messages 截断后，"
            f"保留普通消息数量：{len(recent_msgs)}"
        )

        result = [system_msg] + recent_msgs

        before_chars = self._total_chars(result)
        debug_log(f"按条数截断后 messages 数量：{len(result)}")
        debug_log(f"按条数截断后总字符数：{before_chars}")

        removed_count = 0

        while self._total_chars(result) > self.max_chars and len(result) > 2:
            removed_message = result[1]
            debug_log(
                "总字符数超过 max_chars，删除最旧普通消息："
                f"role={removed_message['role']}, "
                f"content_length={len(removed_message['content'])}, "
                f"preview={preview_text(removed_message['content'])!r}"
            )

            # 保留 system，删除最旧的一条普通消息
            del result[1]
            removed_count += 1

        after_chars = self._total_chars(result)

        debug_log(f"build_messages 删除消息数量：{removed_count}")
        debug_log(f"build_messages 最终 messages 数量：{len(result)}")
        debug_log(f"build_messages 最终总字符数：{after_chars}")

        for index, message in enumerate(result):
            debug_log(
                f"最终 message #{index}: "
                f"role={message['role']}, "
                f"content_length={len(message['content'])}, "
                f"preview={preview_text(message['content'])!r}"
            )

        return result

    def clear(self) -> None:
        debug_log("准备清空短期上下文")
        debug_log(f"清空前 messages 数量：{len(self.messages)}")

        self.messages = [{
            "role": "system",
            "content": self.system_prompt,
        }]

        debug_log(f"清空后 messages 数量：{len(self.messages)}")

    def save(self, path: str | Path) -> None:
        path = Path(path)

        debug_log("准备保存短期上下文")
        debug_log(f"保存路径：{path}")
        debug_log(f"保存前 messages 数量：{len(self.messages)}")
        debug_log(f"保存前总字符数：{self._total_chars(self.messages)}")

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

        debug_log("短期上下文保存完成")

    @classmethod
    def load(
        cls,
        path: str | Path,
        system_prompt: str,
        max_chars: int = 12000,
        keep_last_messages: int = 12,
    ) -> "ContextManager":
        path = Path(path)

        debug_log("准备加载短期上下文")
        debug_log(f"加载路径：{path}")

        if not path.exists():
            debug_log("短期上下文文件不存在，将创建新的 ContextManager")
            return cls(
                system_prompt=system_prompt,
                max_chars=max_chars,
                keep_last_messages=keep_last_messages,
            )

        debug_log("短期上下文文件存在，开始读取 JSON")
        data = json.loads(path.read_text(encoding="utf-8"))

        loaded_messages = data.get("messages", [])

        debug_log(f"文件中 messages 数量：{len(loaded_messages)}")
        debug_log(f"文件中 max_chars：{data.get('max_chars', max_chars)}")
        debug_log(
            "文件中 keep_last_messages："
            f"{data.get('keep_last_messages', keep_last_messages)}"
        )

        context = cls(
            system_prompt=data.get("system_prompt", system_prompt),
            max_chars=data.get("max_chars", max_chars),
            keep_last_messages=data.get("keep_last_messages", keep_last_messages),
            messages=loaded_messages,
        )

        debug_log("短期上下文加载完成")

        return context

    @staticmethod
    def _total_chars(messages: list[Message]) -> int:
        return sum(len(m["content"]) for m in messages)